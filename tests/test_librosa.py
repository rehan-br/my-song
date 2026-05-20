"""Tests for Librosa interpretable-feature extraction."""

import numpy as np

from extraction.interpretable.librosa_extract import (
    _MAJOR,
    estimate_key,
    extract_interpretable,
)


def test_estimate_key_recovers_profile_root() -> None:
    # the major profile correlates best with C major at zero rotation
    assert estimate_key(_MAJOR) == ("C", "major")
    # the same profile rotated up 7 semitones -> G major
    assert estimate_key(np.roll(_MAJOR, 7)) == ("G", "major")


def test_estimate_key_flat_chroma_falls_back_to_c_major() -> None:
    # a variance-free chroma gives undefined correlation -> default
    assert estimate_key(np.ones(12)) == ("C", "major")


def test_extract_interpretable_on_synthetic_tone() -> None:
    sr = 24000
    t = np.arange(sr * 3) / sr
    tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    feats = extract_interpretable(tone, sr)

    assert set(feats) == {
        "bpm",
        "key",
        "mode",
        "energy",
        "loudness_db",
        "dyn_range_db",
        "spectral_centroid",
        "zero_crossing_rate",
        "danceability",
        "valence",
        "arousal",
        "instrumentalness",
        "acousticness",
    }
    assert isinstance(feats["bpm"], float)
    assert feats["mode"] in ("major", "minor")
    assert isinstance(feats["spectral_centroid"], float)
    assert feats["spectral_centroid"] > 0
    # Librosa cannot produce these — they must be left for Essentia/Spotify
    assert feats["danceability"] is None
    assert feats["valence"] is None
