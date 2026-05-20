"""Demucs stem separation — on-demand (the ``--deep`` path).

Splits a track into 4 stems (vocals / drums / bass / other) with ``htdemucs_ft``.
Heavy (~3 GB VRAM, several seconds per track), so it is never in the default
extraction pipeline — only ``music analyze <track> --deep`` reaches it.
"""

from pathlib import Path

from core.logging import get_logger

log = get_logger("demucs")

STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other")


def separate_stems(
    audio_path: Path, out_dir: Path, model_name: str = "htdemucs_ft"
) -> dict[str, Path]:
    """Separate a track into stem WAV files; return ``{stem_name: path}``."""
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio
    from demucs.pretrained import get_model

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("demucs.separating", track=str(audio_path), model=model_name)
    model = get_model(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wav = AudioFile(str(audio_path)).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels
    )
    ref = wav.mean(dim=0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model.to(device), wav[None].to(device), device=device)[0]
    sources = sources.cpu() * ref.std() + ref.mean()

    paths: dict[str, Path] = {}
    for name, source in zip(model.sources, sources, strict=True):
        path = out_dir / f"{name}.wav"
        save_audio(source, str(path), samplerate=model.samplerate)
        paths[name] = path
    log.info("demucs.done", stems=sorted(paths))
    return paths
