"""On-demand deep analysis — Demucs stems + Whisper lyrics for one track.

Reached only via ``music analyze <track> --deep``. Heavy: Demucs separation,
per-stem MERT, Whisper transcription and an e5 lyric embedding. Outputs:
``stems`` rows + stem WAVs, a per-stem MERT shard, the lyric text file, and a
row in ``lyrics_e5.parquet``.
"""

from datetime import UTC, datetime

from omegaconf import DictConfig
from sqlmodel import Session, select

from core import paths
from core.config import config_hash
from core.logging import get_logger
from storage import vectors
from storage.schema import Stem, StemKind, Track

log = get_logger("analyze")


def analyze_track(cfg: DictConfig, session: Session, track: Track) -> dict[str, int]:
    """Run Demucs stem separation + Whisper lyric transcription for one track."""
    from extraction.embeddings.mert import MertEmbedder
    from extraction.lyrics.text_embed import embed_text
    from extraction.lyrics.whisper_transcribe import transcribe
    from extraction.stems.demucs_split import STEM_NAMES, separate_stems
    from extraction.stems.per_stem import embed_stems

    if not track.audio_path:
        raise RuntimeError(f"track {track.id} has no downloaded audio")

    chash = config_hash(cfg)
    audio_root = paths.resolve(cfg.paths.audio)
    audio_file = audio_root / track.audio_path
    sr_mert = int(cfg.extraction.target_sr_mert)

    # --- Demucs stems + per-stem MERT --------------------------------------
    stem_paths = separate_stems(
        audio_file, audio_root / "stems" / track.id, str(cfg.models.demucs.model)
    )
    for old in session.exec(select(Stem).where(Stem.track_id == track.id)).all():
        session.delete(old)
    for name, path in stem_paths.items():
        session.add(
            Stem(
                track_id=track.id,
                kind=StemKind(name),
                path=str(path.relative_to(audio_root)),
                analyzed=True,
            )
        )

    embedder = MertEmbedder(
        str(cfg.models.mert.repo),
        sample_rate=sr_mert,
        device=str(cfg.extraction.device),
        fp16=bool(cfg.extraction.fp16),
        chunk_seconds=int(cfg.extraction.mert.chunk_seconds),
        batch_size=int(cfg.extraction.mert.batch_size),
        seed=int(cfg.extraction.seed),
    )
    stem_embeddings = embed_stems(stem_paths, embedder, sr_mert)
    vectors.write_section_embeddings(
        vectors.stem_embedding_path(cfg, track.id),
        [stem_embeddings[name] for name in STEM_NAMES if name in stem_embeddings],
        str(cfg.models.mert.repo),
        chash,
    )

    # --- Whisper lyrics + e5 embedding -------------------------------------
    whisper_size = str(cfg.models.whisper.repo).rsplit("-", 1)[-1]
    text = transcribe(audio_file, whisper_size)
    lyrics_dir = paths.resolve(cfg.paths.features) / "lyrics"
    lyrics_dir.mkdir(parents=True, exist_ok=True)
    (lyrics_dir / f"{track.id}.txt").write_text(text, encoding="utf-8")

    if text:
        lyrics_path = vectors.song_embedding_path(cfg, "lyrics_e5")
        store = vectors.read_embeddings(lyrics_path)
        store[track.id] = vectors.EmbeddingRow(
            track_id=track.id,
            embedding=embed_text(text, str(cfg.models.lyrics_e5.repo)),
            model=str(cfg.models.lyrics_e5.repo),
            config_hash=chash,
        )
        vectors.write_embeddings(lyrics_path, store.values())

    track.extracted_at = datetime.now(UTC)
    session.add(track)
    log.info("analyze.done", track_id=track.id, stems=len(stem_paths), lyrics=len(text))
    return {"stems": len(stem_paths), "lyric_chars": len(text)}
