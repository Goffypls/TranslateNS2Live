"""Orquesta captura + detección + OCR + traducción en hilos separados.

El hilo de procesamiento corre a su propio ritmo (`process_every_ms`) sobre
el frame más reciente disponible, y para cada recuadro detectado:

1. Calcula un hash de contenido del recorte.
2. Le pregunta al `RegionTracker` si ese recuadro (por posición+contenido)
   ya se tradujo antes y no cambió -> si es así, reusa la traducción sin
   tocar el OCR ni el traductor.
3. Si es nuevo o cambió: corre OCR, busca en la `TranslationCache` por texto
   normalizado (puede ya estar cacheado aunque el recuadro se haya movido de
   posición), y si tampoco está ahí, llama al `Translator`.

El resultado (`list[TranslatedBox]`) queda disponible para que el hilo de
render lo dibuje sobre cada frame sin tener que esperar al procesamiento.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np

from .cache import TranslationCache
from .config import AppConfig
from .detection import TextDetector, get_detector
from .ocr import JapaneseOcr, get_ocr
from .tracker import RegionTracker, average_hash
from .translator import Translator, get_translator
from .types import BBox, TranslatedBox


class ProcessingThread:
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

        self._latest_boxes: list[TranslatedBox] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._get_frame: Callable[[], np.ndarray | None] | None = None  # inyectado en start()

    def start(self, get_frame: Callable[[], np.ndarray | None]) -> None:
        """`get_frame` es un callable sin argumentos que devuelve el último frame (o None)."""

        self._get_frame = get_frame
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = self._config.pipeline.process_every_ms / 1000.0
        while not self._stop_event.is_set():
            start = time.monotonic()
            frame = self._get_frame() if self._get_frame else None
            if frame is not None:
                try:
                    self._process_frame(frame)
                except Exception as exc:  # nunca tirar abajo el hilo de procesamiento
                    print(f"[processing] error: {exc}")
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def _process_frame(self, frame_bgr: np.ndarray) -> None:
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
        with self._lock:
            self._latest_boxes = results

    def get_latest_boxes(self) -> list[TranslatedBox]:
        with self._lock:
            return list(self._latest_boxes)

    def clear_cache(self) -> None:
        self._cache.clear()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cache.save()


def _safe_crop(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
    x2, y2 = min(w, bbox.x2), min(h, bbox.y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype)
    return frame_bgr[y1:y2, x1:x2]
