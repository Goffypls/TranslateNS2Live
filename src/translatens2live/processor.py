"""Motor de traducción por frame: detección + OCR + traducción + tracking/caché.

Es el "cerebro" pesado (modelos de ML) del proyecto. No sabe nada de video en
vivo ni de ventanas: recibe un frame y devuelve los recuadros traducidos. Lo
usa el servidor (`server/main.py`), que lo expone por HTTP para que un
cliente liviano en la PC del usuario no tenga que instalar PaddleOCR,
manga-ocr ni argos-translate.

La "inteligencia" de no retraducir todo cada vez tiene dos niveles:

1. A nivel de frame completo: si la pantalla no cambió respecto al frame
   anterior (dentro de un margen de ruido de la captura), no se corre nada
   del pipeline — ni detector, ni OCR, ni traductor — y se devuelve
   exactamente el mismo resultado ya calculado. Esto es lo que evita el
   "titileo": el detector no es determinístico al 100% entre corridas (el
   ruido normal de una captura de video hace que las cajas tiemblen unos
   píxeles de un frame a otro), así que si ni siquiera hace falta volver a
   correrlo, no hay tembladera posible.
2. A nivel de recuadro individual (cuando el frame sí cambió en algún
   lado): se le pregunta al `RegionTracker` si ese mismo recuadro (misma
   posición + mismo contenido, vía hash) ya se tradujo antes; si es así se
   reusa la traducción sin tocar el OCR ni el traductor. Si es texto nuevo
   pero coincide con algo ya traducido antes (mismo diálogo repetido en
   otra parte de la pantalla), se reusa desde la `TranslationCache` en vez
   de volver a llamar al traductor.
"""

from __future__ import annotations

import cv2
import numpy as np

from .cache import TranslationCache
from .config import AppConfig
from .detection import TextDetector, get_detector
from .layout import merge_line_boxes
from .ocr import JapaneseOcr, get_ocr
from .tracker import RegionTracker, average_hash
from .translator import Translator, get_translator
from .types import BBox, TranslatedBox


class FrameProcessor:
    def __init__(
        self,
        config: AppConfig,
        detector: TextDetector | None = None,
        ocr: JapaneseOcr | None = None,
        translator: Translator | None = None,
    ) -> None:
        self._config = config
        self._detector = detector or get_detector(config.detection)
        self._ocr = ocr or get_ocr(config.ocr)
        self._translator = translator or get_translator(config.translation)
        self._cache = TranslationCache(config.cache.persist_path)
        self._tracker = RegionTracker(
            iou_match_threshold=config.tracker.iou_match_threshold,
            content_change_threshold=config.tracker.content_change_threshold,
            stable_frames_to_lock=config.tracker.stable_frames_to_lock,
        )
        self._last_frame_signature: np.ndarray | None = None
        self._last_results: list[TranslatedBox] = []

    def warmup(self) -> None:
        """Carga los modelos por adelantado (detector, OCR, traductor).

        Se llama al arrancar el servidor para que el primer request real de
        un cliente no tenga que esperar la descarga/inicialización de los
        modelos (eso podía tardar más que el timeout HTTP del cliente).
        Cada componente se intenta por separado: si el traductor todavía no
        tiene los paquetes de argos-translate instalados, el servidor igual
        levanta (el detector/OCR quedan listos) y el error se ve recién al
        traducir, con un mensaje que indica correr `setup_models`.
        """

        for name, component in (
            ("detector", self._detector),
            ("ocr", self._ocr),
            ("translator", self._translator),
        ):
            try:
                component.warmup()
            except Exception as exc:
                print(f"[warmup] no se pudo precargar {name}: {exc}")

    def process(self, frame_bgr: np.ndarray) -> list[TranslatedBox]:
        signature = _frame_signature(frame_bgr)
        if self._last_frame_signature is not None and not _frame_changed(
            self._last_frame_signature, signature, self._config.pipeline.frame_change_threshold
        ):
            # La pantalla no cambió (dentro del margen de ruido de la
            # captura): no se toca el detector/OCR/traductor, se devuelve la
            # misma traducción de siempre. Congela el overlay en vez de
            # dejarlo temblar por corridas del detector levemente distintas
            # sobre una imagen que en la práctica es la misma.
            return list(self._last_results)
        self._last_frame_signature = signature

        boxes = self._detector.detect(frame_bgr)
        boxes = merge_line_boxes(boxes, self._config.detection.line_merge_gap_factor)
        boxes = boxes[: self._config.pipeline.max_boxes_per_frame]

        results: list[TranslatedBox] = []
        for bbox in boxes:
            crop = _safe_crop(frame_bgr, bbox)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else crop
            content_hash = average_hash(gray)

            if not self._tracker.needs_reprocessing(bbox, content_hash):
                reused = self._tracker.update(bbox, content_hash, translated=None)
                if reused is not None:
                    results.append(reused)
                continue

            source_text = self._ocr.recognize(crop)
            if not source_text:
                continue

            cached = self._cache.get(source_text)
            if cached is None:
                cached = self._translator.translate(source_text)
                self._cache.put(source_text, cached)

            translated = TranslatedBox(
                bbox=bbox, source_text=source_text, translated_text=cached
            )
            updated = self._tracker.update(bbox, content_hash, translated=translated)
            if updated is not None:
                results.append(updated)

        self._tracker.prune(boxes)
        self._last_results = results
        return results

    def clear_cache(self) -> None:
        self._cache.clear()

    def save_cache(self) -> None:
        self._cache.save()

    def cache_size(self) -> int:
        return len(self._cache)


_SIGNATURE_SIZE = (160, 90)  # chico y grosero a propósito: solo para detectar "cambió o no"


def _frame_signature(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, _SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.int16)


def _frame_changed(prev: np.ndarray, current: np.ndarray, threshold: float) -> bool:
    diff = float(np.abs(prev - current).mean())
    return diff > threshold


def _safe_crop(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
    x2, y2 = min(w, bbox.x2), min(h, bbox.y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype)
    return frame_bgr[y1:y2, x1:x2]
