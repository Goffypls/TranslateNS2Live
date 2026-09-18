"""Agrupa recuadros de texto detectados que en realidad son la misma línea.

PaddleOCR a veces detecta una sola oración larga como varios recuadros
chicos separados (sobre todo con texto japonés espaciado). Traducir cada
fragmento por separado da traducciones incoherentes y un overlay lleno de
pedacitos sueltos. Esto agrupa recuadros que se solapan verticalmente y
están horizontalmente cerca en un solo recuadro (la unión de todos), para
que el OCR y la traducción corran sobre la línea completa.
"""

from __future__ import annotations

from .types import BBox


def merge_line_boxes(boxes: list[BBox], gap_factor: float = 1.5) -> list[BBox]:
    """Junta recuadros de la misma línea en uno solo.

    `gap_factor` controla qué tan separados (relativo a la altura de los
    recuadros) pueden estar dos fragmentos para seguir considerándose parte
    de la misma línea. 0 (o negativo) desactiva el merge.
    """

    if not boxes or gap_factor <= 0:
        return list(boxes)

    lines: list[list[BBox]] = []
    for box in sorted(boxes, key=lambda b: (b.y1, b.x1)):
        placed = False
        for line in lines:
            line_y1 = min(b.y1 for b in line)
            line_y2 = max(b.y2 for b in line)
            overlap = max(0, min(line_y2, box.y2) - max(line_y1, box.y1))
            min_height = min(line_y2 - line_y1, box.height) or 1
            if overlap / min_height >= 0.5:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    merged: list[BBox] = []
    for line in lines:
        line.sort(key=lambda b: b.x1)
        cluster = [line[0]]
        for box in line[1:]:
            prev = cluster[-1]
            gap = box.x1 - prev.x2
            avg_height = (prev.height + box.height) / 2
            if gap <= avg_height * gap_factor:
                cluster.append(box)
            else:
                merged.append(_union(cluster))
                cluster = [box]
        merged.append(_union(cluster))

    return merged


def _union(cluster: list[BBox]) -> BBox:
    return BBox(
        min(b.x1 for b in cluster),
        min(b.y1 for b in cluster),
        max(b.x2 for b in cluster),
        max(b.y2 for b in cluster),
    )
