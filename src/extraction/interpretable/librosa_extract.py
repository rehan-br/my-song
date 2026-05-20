"""Interpretable features via Librosa.

These describe music's *surface* — tempo, key, brightness, dynamics. Per
CLAUDE.md their job is explainability and steerability, not driving the ranker
(that is the learned MERT/CLAP embeddings' job).

Columns that need Essentia ML models or Spotify audio features (danceability,
valence, arousal, instrumentalness, acousticness) are returned as ``None`` —
Librosa cannot produce them. They are filled in once Essentia is wired in.
"""

import librosa
import numpy as np
from librosa.feature.rhythm import tempo as _tempo  # lazy-loaded submodule

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (tonal hierarchy weights per pitch class).
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str]:
    """Estimate ``(key, mode)`` from a 12-d mean chroma vector.

    Correlates the chroma against the major/minor K-S profiles at all 12
    rotations and returns the best match. Falls back to C major if the chroma
    has no variance (correlation undefined).
    """
    chroma_mean = np.asarray(chroma_mean, dtype=np.float64)
    best_corr, best_key, best_mode = -2.0, 0, "major"
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            corr = float(np.corrcoef(rolled, profile)[0, 1])
            if np.isfinite(corr) and corr > best_corr:
                best_corr, best_key, best_mode = corr, shift, mode
    return _KEY_NAMES[best_key], best_mode


def extract_interpretable(waveform: np.ndarray, sr: int) -> dict[str, object]:
    """Compute Librosa-derived interpretable features for a mono waveform."""
    y = np.asarray(waveform, dtype=np.float32)

    tempo = float(_tempo(y=y, sr=sr)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode = estimate_key(chroma.mean(axis=1))

    rms = librosa.feature.rms(y=y)[0]
    rms_safe = np.maximum(rms, 1e-8)
    loudness_db = float(20.0 * np.log10(rms_safe.mean()))
    dyn_range_db = float(20.0 * np.log10(np.percentile(rms_safe, 95) / np.percentile(rms_safe, 5)))
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y=y)[0].mean())

    return {
        "bpm": round(tempo, 2),
        "key": key,
        "mode": mode,
        "energy": round(float(rms.mean()), 6),
        "loudness_db": round(loudness_db, 2),
        "dyn_range_db": round(dyn_range_db, 2),
        "spectral_centroid": round(centroid, 2),
        "zero_crossing_rate": round(zcr, 6),
        # Essentia / Spotify territory — not derivable from Librosa:
        "danceability": None,
        "valence": None,
        "arousal": None,
        "instrumentalness": None,
        "acousticness": None,
    }
