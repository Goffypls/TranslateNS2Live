import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translatens2live.protocol import boxes_from_json, boxes_to_json
from translatens2live.types import BBox, TranslatedBox


def test_roundtrip():
    boxes = [
        TranslatedBox(
            bbox=BBox(1, 2, 30, 40),
            source_text="こんにちは",
            translated_text="Hola",
            stable=True,
        )
    ]
    data = boxes_to_json(boxes)
    restored = boxes_from_json(data)
    assert restored == boxes
