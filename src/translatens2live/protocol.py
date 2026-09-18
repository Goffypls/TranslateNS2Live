"""Serialización JSON de `TranslatedBox` para el protocolo cliente <-> servidor."""

from __future__ import annotations

from typing import Any

from .types import BBox, TranslatedBox


def boxes_to_json(boxes: list[TranslatedBox]) -> list[dict[str, Any]]:
    return [
        {
            "bbox": [b.bbox.x1, b.bbox.y1, b.bbox.x2, b.bbox.y2],
            "source_text": b.source_text,
            "translated_text": b.translated_text,
            "stable": b.stable,
        }
        for b in boxes
    ]


def boxes_from_json(data: list[dict[str, Any]]) -> list[TranslatedBox]:
    result = []
    for item in data:
        x1, y1, x2, y2 = item["bbox"]
        result.append(
            TranslatedBox(
                bbox=BBox(int(x1), int(y1), int(x2), int(y2)),
                source_text=item["source_text"],
                translated_text=item["translated_text"],
                stable=bool(item.get("stable", False)),
            )
        )
    return result
