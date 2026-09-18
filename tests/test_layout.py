import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translatens2live.layout import merge_line_boxes
from translatens2live.types import BBox


def test_merges_fragments_on_the_same_line():
    boxes = [
        BBox(0, 0, 40, 20),
        BBox(45, 2, 90, 22),
        BBox(95, 1, 150, 21),
    ]
    merged = merge_line_boxes(boxes, gap_factor=1.5)
    assert merged == [BBox(0, 0, 150, 22)]


def test_keeps_separate_lines_apart():
    line1 = [BBox(0, 0, 40, 20), BBox(45, 0, 90, 20)]
    line2 = [BBox(0, 100, 40, 120), BBox(45, 100, 90, 120)]
    merged = merge_line_boxes(line1 + line2, gap_factor=1.5)
    assert len(merged) == 2
    assert BBox(0, 0, 90, 20) in merged
    assert BBox(0, 100, 90, 120) in merged


def test_does_not_merge_across_large_gaps():
    boxes = [BBox(0, 0, 40, 20), BBox(500, 0, 540, 20)]
    merged = merge_line_boxes(boxes, gap_factor=1.5)
    assert len(merged) == 2


def test_gap_factor_zero_disables_merge():
    boxes = [BBox(0, 0, 40, 20), BBox(45, 0, 90, 20)]
    merged = merge_line_boxes(boxes, gap_factor=0)
    assert merged == boxes


def test_empty_input():
    assert merge_line_boxes([]) == []
