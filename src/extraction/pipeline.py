"""End-to-end feature extraction for one or more tracks.

Phase 1 runs, per track: MERT full-song embedding, CLAP full-song embedding,
and Librosa interpretable features. Structure segmentation + per-section MERT
slot in next. Every output stores ``config_hash`` (invariant 2).

Which tracks get (re-)extracted is decided by the caller (the ``extract``
command: status=downloaded by default, all tracks with ``--force``). This
module fully extracts whatever it is handed.
"""

from datetime import UTC, datetime

from omegaconf import DictConfig
from sqlmodel import Session

from core import paths
from core.config import config_hash
from core.logging import get_logger
from storage import vectors
from storage.schema import FeaturesInterpretable, Track, TrackStatus

log = get_logger("pipeline")


def _upsert_features(
    session: Session, track_id: str, chash: str, features: dict[str, object]
) -> None:
    """Insert or update a track's interpretable-features row."""
    row = session.get(FeaturesInterpretable, track_id)
    if row is None:
        row = FeaturesInterpretable(track_id=track_id, config_hash=chash)
    row.config_hash = chash
    row.extracted_at = datetime.now(UTC)
    for name, value in features.items():
        setattr(row, name, value)
    session.add(row)


def run_extraction(
    cfg: DictConfig, session: Session, tracks: list[Track], force: bool = False
) -> dict[str, int]:
    """Extract MERT + CLAP embeddings and Librosa features for ``tracks``.

    Returns a counts summary: ``ok`` extracted, ``failed`` errored. ``force`` is
    accepted for signature symmetry; track selection is the caller's job.
    """
    from extraction.audio import load_audio
    from extraction.embeddings.clap import ClapEmbedder
    from extraction.embeddings.mert import MertEmbedder
    from extraction.interpretable.librosa_extract import extract_interpretable

    chash = config_hash(cfg)
    audio_root = paths.resolve(cfg.paths.audio)
    sr_mert = int(cfg.extraction.target_sr_mert)
    sr_clap = int(cfg.extraction.target_sr_clap)
    mert_repo = str(cfg.models.mert.repo)
    clap_repo = str(cfg.models.clap.repo)

    mert_path = vectors.song_embedding_path(cfg, "mert_song")
    clap_path = vectors.song_embedding_path(cfg, "clap_song")
    mert_store = vectors.read_embeddings(mert_path)
    clap_store = vectors.read_embeddings(clap_path)

    mert_embedder: MertEmbedder | None = None
    clap_embedder: ClapEmbedder | None = None
    counts = {"ok": 0, "failed": 0}
    changed = False

    for track in tracks:
        if not track.audio_path:
            counts["failed"] += 1
            log.warning("extract.no_audio", track_id=track.id)
            continue
        audio_file = audio_root / track.audio_path
        try:
            if mert_embedder is None:
                mert_embedder = MertEmbedder(
                    mert_repo,
                    sample_rate=sr_mert,
                    device=str(cfg.extraction.device),
                    fp16=bool(cfg.extraction.fp16),
                    chunk_seconds=int(cfg.extraction.mert.chunk_seconds),
                    batch_size=int(cfg.extraction.mert.batch_size),
                    seed=int(cfg.extraction.seed),
                )
                clap_embedder = ClapEmbedder(
                    clap_repo,
                    sample_rate=sr_clap,
                    device=str(cfg.extraction.device),
                    fp16=bool(cfg.extraction.fp16),
                    chunk_seconds=int(cfg.extraction.clap.chunk_seconds),
                    batch_size=int(cfg.extraction.clap.batch_size),
                    seed=int(cfg.extraction.seed),
                )
            assert clap_embedder is not None

            # MERT + interpretable share the 24kHz decode.
            wav_mert = load_audio(audio_file, sr_mert)
            mert_store[track.id] = vectors.EmbeddingRow(
                track_id=track.id,
                embedding=mert_embedder.embed_song(wav_mert),
                model=mert_repo,
                config_hash=chash,
            )
            _upsert_features(session, track.id, chash, extract_interpretable(wav_mert, sr_mert))

            # CLAP needs 48kHz.
            wav_clap = load_audio(audio_file, sr_clap)
            clap_store[track.id] = vectors.EmbeddingRow(
                track_id=track.id,
                embedding=clap_embedder.embed_song(wav_clap),
                model=clap_repo,
                config_hash=chash,
            )

            track.status = TrackStatus.extracted
            track.extracted_at = datetime.now(UTC)
            session.add(track)
            changed = True
            counts["ok"] += 1
            log.info("extract.ok", track_id=track.id)
        except Exception as exc:
            # Leave status as `downloaded` so the track can be retried later.
            counts["failed"] += 1
            log.warning("extract.failed", track_id=track.id, error=str(exc))

    if changed:
        vectors.write_embeddings(mert_path, mert_store.values())
        vectors.write_embeddings(clap_path, clap_store.values())
        log.info("extract.written", mert=len(mert_store), clap=len(clap_store))
    return counts
