import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translatens2live.tracker import RegionTracker, average_hash, hamming_distance
from translatens2live.types import BBox, TranslatedBox


def test_average_hash_stable_for_identical_crop():
    crop = np.random.default_rng(0).integers(0, 255, size=(20, 40), dtype=np.uint8)
    assert average_hash(crop) == average_hash(crop.copy())


def test_average_hash_differs_for_very_different_crops():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, size=(20, 40), dtype=np.uint8)
    b = 255 - a
    assert hamming_distance(average_hash(a), average_hash(b)) > 30


def test_needs_reprocessing_true_for_unseen_box():
    tracker = RegionTracker()
    bbox = BBox(0, 0, 100, 30)
    assert tracker.needs_reprocessing(bbox, content_hash=123)


def test_reuses_translation_when_box_and_content_unchanged():
    tracker = RegionTracker(iou_match_threshold=0.4, content_change_threshold=6)
    bbox = BBox(0, 0, 100, 30)
    translated = TranslatedBox(bbox=bbox, source_text="こんにちは", translated_text="Hola")

    tracker.update(bbox, content_hash=0b1010, translated=translated)
    assert not tracker.needs_reprocessing(bbox, content_hash=0b1010)

    result = tracker.update(bbox, content_hash=0b1010, translated=None)
    assert result is not None
    assert result.translated_text == "Hola"


def test_flags_reprocessing_when_content_hash_changes_a_lot():
    tracker = RegionTracker(iou_match_threshold=0.4, content_change_threshold=2)
    bbox = BBox(0, 0, 100, 30)
    translated = TranslatedBox(bbox=bbox, source_text="A", translated_text="A-es")
    tracker.update(bbox, content_hash=0b0000, translated=translated)

    # Hash muy distinto (todos los bits difieren) -> hay que reprocesar.
    assert tracker.needs_reprocessing(bbox, content_hash=0xFFFFFFFFFFFFFFFF)


def test_stable_flag_requires_consecutive_matches():
    tracker = RegionTracker(stable_frames_to_lock=3)
    bbox = BBox(0, 0, 100, 30)
    translated = TranslatedBox(bbox=bbox, source_text="A", translated_text="A-es")

    r1 = tracker.update(bbox, content_hash=0, translated=translated)
    assert r1 is not None and not r1.stable

    tracker.update(bbox, content_hash=0, translated=None)
    r3 = tracker.update(bbox, content_hash=0, translated=None)
    assert r3 is not None and r3.stable


def test_prune_removes_boxes_not_present_anymore():
    tracker = RegionTracker()
    bbox = BBox(0, 0, 100, 30)
    translated = TranslatedBox(bbox=bbox, source_text="A", translated_text="A-es")
    tracker.update(bbox, content_hash=0, translated=translated)

    tracker.prune(current_bboxes=[])
    assert tracker.needs_reprocessing(bbox, content_hash=0)


def test_bbox_iou():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    c = BBox(100, 100, 110, 110)
    assert 0 < a.iou(b) < 1
    assert a.iou(c) == 0.0
    assert a.iou(a) == 1.0
