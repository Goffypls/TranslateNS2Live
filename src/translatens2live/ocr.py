"""Reconocimiento (OCR) del texto japonés dentro de un recuadro ya detectado.

`manga-ocr` es un modelo transformer entrenado específicamente en texto de
manga/videojuegos japoneses (tipografías estilizadas, texto vertical/curvo,
furigana, etc.), mucho más confiable que un OCR genérico para este caso de
uso que un motor OCR de propósito general.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from .config import OcrConfig


class JapaneseOcr(ABC):
    @abstractmethod
    def recognize(self, crop_bgr: np.ndarray) -> str:
        """Reconoce el texto japonés en un recorte (BGR, uint8). Puede devolver ''."""


class MangaOcrRecognizer(JapaneseOcr):
    def __init__(self, config: OcrConfig) -> None:
        self._config = config
        self._model = None  # carga perezosa: descarga/inicializa un modelo transformer

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from manga_ocr import MangaOcr  # import perezoso

        self._model = MangaOcr()

    def recognize(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr.size == 0:
            return ""
        self._ensure_loaded()
        assert self._model is not None
        image = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR -> RGB
        text = self._model(image)
        return text.strip()


def get_ocr(config: OcrConfig) -> JapaneseOcr:
    if config.backend == "manga_ocr":
        return MangaOcrRecognizer(config)
    raise ValueError(f"Backend de OCR desconocido: {config.backend}")
