"""Structure segmentation.

CLAUDE.md's pipeline uses Essentia for this; Essentia has no Windows build, so
sections come from Librosa instead: agglomerative clustering of the chroma
sequence yields section boundaries. Boundaries only — semantic labels
(verse / chorus) are a harder MIR problem and not needed for the embedding
approach, so every section is recorded with ``kind = other``.
"""

import librosa
import numpy as np


def segment_track(waveform: np.ndarray, sr: int, n_sections: int = 8) -> list[tuple[float, float]]:
    """Return contiguous ``(start_s, end_s)`` spans for a track's sections.

    The spans tile the whole track: the first starts at 0, the last ends at the
    track duration. Sub-half-second slivers are merged away.
    """
    y = np.asarray(waveform, dtype=np.float32)
    duration = len(y) / sr
    if duration < 2.0:
        return [(0.0, round(duration, 3))]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    k = max(1, min(n_sections, chroma.shape[1] - 1))
    bounds = librosa.segment.agglomerative(chroma, k)
    starts = sorted({0.0, *librosa.frames_to_time(bounds, sr=sr).tolist()})
    edges = [t for t in starts if 0.0 <= t < duration] + [duration]

    spans: list[tuple[float, float]] = []
    for start, end in zip(edges, edges[1:], strict=False):
        if end - start <= 0.5 and spans:
            spans[-1] = (spans[-1][0], round(end, 3))  # merge sliver leftward
        else:
            spans.append((round(start, 3), round(end, 3)))
    return spans or [(0.0, round(duration, 3))]
