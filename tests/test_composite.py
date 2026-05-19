"""Tests for composite (MERT + CLAP) score blending."""

import numpy as np

from recommend.composite import blend


def test_blend_combines_z_normalised_signals() -> None:
    mert = np.array([1.0, 2.0, 3.0])
    clap = np.array([3.0, 2.0, 1.0])
    has = np.array([True, True, True])
    out = blend(mert, clap, has, mert_weight=1.0, clap_weight=1.0)
    # equal weights, exactly opposite orders -> the z-scores cancel
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-9)


def test_blend_uses_mert_only_without_clap() -> None:
    mert = np.array([1.0, 2.0, 3.0])
    out = blend(mert, np.zeros(3), np.array([False, False, False]), 0.6, 0.4)
    np.testing.assert_allclose(out, 0.6 * (mert - mert.mean()) / mert.std())


def test_blend_follows_clap_when_mert_is_flat() -> None:
    mert = np.array([1.0, 1.0, 1.0])  # no MERT signal at all
    clap = np.array([1.0, 5.0, 9.0])
    out = blend(mert, clap, np.array([True, True, True]), 0.5, 0.5)
    assert out[2] > out[1] > out[0]
