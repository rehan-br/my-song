"""Per-stem MERT embeddings — reruns the audio embedder over each Demucs stem.

Lets the taste model reason about a track's vocal vs instrumental character
separately. On-demand only — reached via ``music analyze --deep``.
"""

from pathlib import Path
from typing import Any

import numpy as np

from extraction.audio import load_audio


def embed_stems(
    stem_paths: dict[str, Path], embedder: Any, sample_rate: int
) -> dict[str, np.ndarray]:
    """MERT-embed each stem WAV; returns ``{stem_name: embedding}``."""
    return {
        name: embedder.embed_song(load_audio(path, sample_rate))
        for name, path in stem_paths.items()
    }
