"""Audio decoding + normalization.

Decoding goes through the ffmpeg executable rather than torchaudio: ffmpeg is
already a hard dependency (yt-dlp post-processing), it decodes every format the
cache may hold (m4a/opus/mp3/wav), and it avoids torchaudio's flaky codec
backends on Windows. ffmpeg also does the resample, so callers get audio at
exactly the sample rate a model expects (invariant 8: MERT 24kHz, CLAP 48kHz).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np


class AudioDecodeError(RuntimeError):
    """Raised when ffmpeg fails to decode an audio file."""


def load_audio(path: str | Path, target_sr: int, mono: bool = True) -> np.ndarray:
    """Decode ``path`` to float32 PCM at ``target_sr`` via ffmpeg.

    Returns a 1-D array (mono) or ``(n, 2)`` array (stereo) of float32 samples
    in roughly ``[-1, 1]``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("ffmpeg not found on PATH — required for audio decoding.")

    channels = 1 if mono else 2
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        str(channels),
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace")[:500]
        raise AudioDecodeError(f"ffmpeg failed to decode {path}: {detail}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AudioDecodeError(f"decoded no samples from {path}")
    return audio if mono else audio.reshape(-1, 2)


def chunk(waveform: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Split a 1-D waveform into fixed-length chunks, zero-padding the last one.

    A waveform shorter than one chunk yields a single padded chunk.
    """
    if waveform.ndim != 1:
        raise ValueError("chunk() expects a mono (1-D) waveform")
    chunks: list[np.ndarray] = []
    for start in range(0, max(len(waveform), 1), chunk_samples):
        piece = waveform[start : start + chunk_samples]
        if len(piece) < chunk_samples:
            piece = np.pad(piece, (0, chunk_samples - len(piece)))
        chunks.append(piece)
    return chunks
