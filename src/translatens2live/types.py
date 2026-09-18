"""Tipos de datos compartidos por el pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Caja delimitadora en coordenadas de píxel (origen arriba-izquierda)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectedRegion:
    """Región de texto detectada en un frame, antes de OCR/traducción."""

    bbox: BBox
    content_hash: int | None = None  # hash perceptual del recorte, para el tracker


@dataclass
class TranslatedBox:
    """Resultado final listo para dibujar en el overlay."""

    bbox: BBox
    source_text: str
    translated_text: str
    stable: bool = False
