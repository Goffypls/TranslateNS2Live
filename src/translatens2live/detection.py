"""Detección de regiones de texto en un frame.

El detector solo encuentra *dónde* hay texto (bounding boxes); el
reconocimiento del contenido japonés lo hace `ocr.py` por separado. Separar
ambas etapas permite usar un detector rápido y genérico (no necesita saber
japonés) y un reconocedor especializado en la tipografía de videojuegos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import DetectionConfig
from .types import BBox


class TextDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[BBox]:
        """Devuelve las bounding boxes de texto encontradas en el frame (BGR, uint8)."""


class PaddleTextDetector(TextDetector):
    """Detector de regiones de texto basado en PaddleOCR (solo la etapa `det`)."""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._ocr = None  # carga perezosa: paddleocr es pesado de importar

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR  # import perezoso

        self._ocr = PaddleOCR(
            lang="japan",
            use_angle_cls=False,
            det_db_box_thresh=self._config.det_db_box_thresh,
            show_log=False,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[BBox]:
        self._ensure_loaded()
        assert self._ocr is not None
        result = self._ocr.ocr(frame_bgr, det=True, rec=False, cls=False)

        boxes: list[BBox] = []
        for quad in result[0] or []:
            bbox = _quad_to_bbox(quad)
            if bbox.area >= self._config.min_box_area:
                boxes.append(bbox)
        return boxes


def _quad_to_bbox(quad: list[list[float]]) -> BBox:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return BBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def get_detector(config: DetectionConfig) -> TextDetector:
    if config.backend == "paddleocr":
        return PaddleTextDetector(config)
    raise ValueError(f"Backend de detección desconocido: {config.backend}")
