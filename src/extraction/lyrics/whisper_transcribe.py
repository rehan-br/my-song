"""Whisper transcription — on-demand lyric text for the ``--deep`` path.

Heavy and only meaningful for tracks with intelligible vocals, so it is never
in the default pipeline. Model: ``whisper-medium`` (CLAUDE.md pin).
"""

from pathlib import Path
from typing import Any

from core.logging import get_logger

log = get_logger("whisper")

_models: dict[str, Any] = {}


def transcribe(audio_path: Path, model_size: str = "medium") -> str:
    """Transcribe a track's audio to text (empty string if nothing recognised)."""
    import whisper

    if model_size not in _models:
        log.info("whisper.loading", model=model_size)
        _models[model_size] = whisper.load_model(model_size)
    result = _models[model_size].transcribe(str(audio_path))
    return str(result.get("text", "")).strip()
