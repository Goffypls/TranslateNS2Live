import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translatens2live.processor import _frame_changed, _frame_signature


def test_identical_frames_are_not_changed():
    frame = np.random.default_rng(0).integers(0, 255, size=(200, 300, 3), dtype=np.uint8)
    sig_a = _frame_signature(frame)
    sig_b = _frame_signature(frame.copy())
    assert not _frame_changed(sig_a, sig_b, threshold=2.5)


def test_tiny_noise_is_not_changed():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, size=(200, 300, 3), dtype=np.uint8)
    noisy = frame.astype(np.int16) + rng.integers(-2, 3, size=frame.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    sig_a = _frame_signature(frame)
    sig_b = _frame_signature(noisy)
    assert not _frame_changed(sig_a, sig_b, threshold=2.5)


def test_a_real_change_is_detected():
    frame_a = np.zeros((200, 300, 3), dtype=np.uint8)
    frame_b = np.full((200, 300, 3), 255, dtype=np.uint8)
    sig_a = _frame_signature(frame_a)
    sig_b = _frame_signature(frame_b)
    assert _frame_changed(sig_a, sig_b, threshold=2.5)
