"""Tests for structure segmentation."""

import numpy as np

from extraction.segmentation import segment_track


def test_segment_track_tiles_the_whole_track() -> None:
    sr = 24000
    t = np.arange(sr * 6) / sr
    # a clear timbral change at the 3-second mark
    tone = np.where(t < 3, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t))
    spans = segment_track(tone.astype(np.float32), sr, n_sections=4)

    assert spans[0][0] == 0.0
    assert abs(spans[-1][1] - 6.0) < 0.3
    for current, following in zip(spans, spans[1:], strict=False):
        assert current[1] == following[0]  # sections are contiguous
    assert all(end > start for start, end in spans)


def test_segment_short_track_is_one_section() -> None:
    spans = segment_track(np.ones(12000, dtype=np.float32), 24000)
    assert len(spans) == 1
    assert spans[0][0] == 0.0
