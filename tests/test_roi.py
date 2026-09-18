import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translatens2live.client.remote_pipeline import _clamp_roi, _offset_box
from translatens2live.types import BBox, TranslatedBox


def test_clamp_roi_within_bounds_unchanged():
    roi = BBox(10, 10, 100, 100)
    assert _clamp_roi(roi, frame_w=200, frame_h=200) == roi


def test_clamp_roi_clips_to_frame():
    roi = BBox(-50, -50, 500, 500)
    clamped = _clamp_roi(roi, frame_w=200, frame_h=150)
    assert clamped == BBox(0, 0, 200, 150)


def test_offset_box_shifts_bbox_and_keeps_text():
    box = TranslatedBox(bbox=BBox(0, 0, 10, 10), source_text="a", translated_text="b", stable=True)
    shifted = _offset_box(box, dx=50, dy=20)
    assert shifted.bbox == BBox(50, 20, 60, 30)
    assert shifted.source_text == "a"
    assert shifted.translated_text == "b"
    assert shifted.stable is True
