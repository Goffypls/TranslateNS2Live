"""Motor de traducción por frame: detección + OCR + traducción + tracking/caché.

Es el "cerebro" pesado (modelos de ML) del proyecto. No sabe nada de video en
vivo ni de ventanas: recibe un frame y devuelve los recuadros traducidos. Lo
usa el servidor (`server/main.py`), que lo expone por HTTP para que un
cliente liviano en la PC del usuario no tenga que instalar PaddleOCR,
manga-ocr ni argos-translate.

La "inteligencia" de no retraducir todo cada vez vive acá: por cada recuadro
detectado se le pregunta al `RegionTracker` si ese mismo recuadro (misma
posición + mismo contenido, vía hash) ya se tradujo en un frame anterior; si
es así se reusa la traducción sin tocar el OCR ni el traductor. Si es texto
nuevo pero coincide con algo ya traducido antes (mismo diálogo repetido en
otra parte de la pantalla), se reusa desde la `TranslationCache` en vez de
volver a llamar al traductor.
"""

from __future__ import annotations

import cv2
import numpy as np

from .cache import TranslationCache
from .config import AppConfig
from .detection import TextDetector, get_detector
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

    def process(self, frame_bgr: np.ndarray) -> list[TranslatedBox]:
        boxes = self._detector.detect(frame_bgr)
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
        return results

    def clear_cache(self) -> None:
        self._cache.clear()

    def save_cache(self) -> None:
        self._cache.save()

    def cache_size(self) -> int:
        return len(self._cache)


def _safe_crop(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
    x2, y2 = min(w, bbox.x2), min(h, bbox.y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype)
    return frame_bgr[y1:y2, x1:x2]
