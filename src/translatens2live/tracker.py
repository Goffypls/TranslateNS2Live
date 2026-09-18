"""Tracking de recuadros de texto entre frames.

Objetivo: no volver a correr OCR+traducción en un recuadro que no cambió, y
emparejar recuadros entre frames aunque el detector los devuelva con
coordenadas levemente distintas (jitter normal de un detector por frame).

Estrategia:
- Cada recuadro detectado se identifica por posición (bbox) + un hash
  perceptual simple de su contenido (average hash).
- Un recuadro "trackeado" del frame anterior se empareja con uno nuevo si el
  IoU entre ambos supera `iou_match_threshold`.
- Si el hash de contenido del match no cambió (o cambió por debajo del
  umbral), el recuadro se considera igual y se reutiliza la traducción ya
  calculada en vez de pedir OCR/traducción de nuevo.
- Un recuadro necesita `stable_frames_to_lock` coincidencias seguidas antes
  de marcarse "stable" (evita traducir texto que todavía se está animando,
  tipo diálogo con efecto máquina de escribir).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import BBox, TranslatedBox


def average_hash(gray_crop: np.ndarray, hash_size: int = 8) -> int:
    """Average hash (aHash) de un recorte en escala de grises, como entero de 64 bits."""

    if gray_crop.size == 0:
        return 0
    small = _resize_nearest(gray_crop, hash_size, hash_size)
    mean = small.mean()
    bits = (small > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _resize_nearest(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize nearest-neighbor sin depender de OpenCV, para poder testear sin esa dependencia."""

    h, w = arr.shape[:2]
    row_idx = (np.linspace(0, h - 1, height)).astype(int)
    col_idx = (np.linspace(0, w - 1, width)).astype(int)
    return arr[row_idx][:, col_idx]


@dataclass
class _TrackedRegion:
    bbox: BBox
    content_hash: int
    translated: TranslatedBox | None = None
    stable_streak: int = 0


@dataclass
class RegionTracker:
    iou_match_threshold: float = 0.4
    content_change_threshold: int = 6
    stable_frames_to_lock: int = 2
    _tracked: list[_TrackedRegion] = field(default_factory=list)

    def match(
        self, bbox: BBox, content_hash: int
    ) -> tuple[_TrackedRegion | None, bool]:
        """Busca un recuadro trackeado que corresponda a `bbox`.

        Devuelve (región_trackeada_o_None, contenido_sin_cambios).
        `contenido_sin_cambios=True` significa que se puede reusar la
        traducción cacheada en esa región sin volver a correr OCR.
        """

        best: _TrackedRegion | None = None
        best_iou = 0.0
        for region in self._tracked:
            iou = region.bbox.iou(bbox)
            if iou > best_iou:
                best_iou = iou
                best = region

        if best is None or best_iou < self.iou_match_threshold:
            return None, False

        unchanged = (
            hamming_distance(best.content_hash, content_hash)
            <= self.content_change_threshold
        )
        return best, unchanged

    def update(
        self,
        bbox: BBox,
        content_hash: int,
        translated: TranslatedBox | None,
    ) -> TranslatedBox | None:
        """Registra/actualiza el estado de un recuadro para el próximo frame.

        Si `translated` es None se reusa la traducción previa del match (caso
        "contenido sin cambios"). Devuelve el TranslatedBox con `stable`
        actualizado según la racha de frames sin cambios.
        """

        matched, unchanged = self.match(bbox, content_hash)

        if matched is not None and unchanged:
            matched.bbox = bbox
            matched.stable_streak += 1
            if translated is not None:
                matched.translated = translated
            if matched.translated is not None:
                matched.translated = TranslatedBox(
                    bbox=bbox,
                    source_text=matched.translated.source_text,
                    translated_text=matched.translated.translated_text,
                    stable=matched.stable_streak >= self.stable_frames_to_lock,
                )
            return matched.translated

        # Contenido nuevo o cambiado: reemplaza el registro trackeado.
        if matched is not None:
            self._tracked.remove(matched)

        new_region = _TrackedRegion(
            bbox=bbox, content_hash=content_hash, translated=translated, stable_streak=1
        )
        self._tracked.append(new_region)
        if translated is not None:
            new_region.translated = TranslatedBox(
                bbox=bbox,
                source_text=translated.source_text,
                translated_text=translated.translated_text,
                stable=self.stable_frames_to_lock <= 1,
            )
        return new_region.translated

    def prune(self, current_bboxes: list[BBox]) -> None:
        """Elimina recuadros trackeados que ya no aparecen en el frame (texto que desapareció)."""

        if not current_bboxes:
            self._tracked.clear()
            return
        self._tracked = [
            r
            for r in self._tracked
            if any(r.bbox.iou(b) >= self.iou_match_threshold for b in current_bboxes)
        ]

    def needs_reprocessing(self, bbox: BBox, content_hash: int) -> bool:
        """True si este recuadro no tiene traducción cacheada reusable todavía."""

        matched, unchanged = self.match(bbox, content_hash)
        return matched is None or not unchanged or matched.translated is None
