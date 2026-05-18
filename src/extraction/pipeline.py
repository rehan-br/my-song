"""End-to-end feature extraction.

Per track: MERT full-song embedding, and — unless ``fast`` — a CLAP embedding
plus Librosa interpretable features. ``fast`` mode runs **MERT only**: that is
all the M1 recommender ranks on, it roughly halves extraction time, and CLAP /
Librosa can be backfilled later with a normal run.

Audio decoding runs on a thread pool that reads *ahead* of the GPU (see
``_prefetch``), so MERT/CLAP inference is never blocked waiting on ffmpeg —
the single-machine form of the multi-user "GPU-saturated extraction" stage.
config_hash is stored on every output (invariant 2).
"""

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from omegaconf import DictConfig
from sqlmodel import Session

from core import paths
from core.config import config_hash
from core.logging import get_logger
from storage import vectors
from storage.schema import FeaturesInterpretable, Track, TrackStatus

log = get_logger("pipeline")


def _prefetch[T, R](
    pool: ThreadPoolExecutor, items: list[T], job: Callable[[T], R], ahead: int
) -> Iterator[tuple[T, R]]:
    """Yield ``(item, job(item))`` in order, keeping ``ahead`` jobs in flight.

    Lets slow per-item work (audio decoding) run on worker threads ahead of the
    consumer (GPU inference) without buffering every result in memory at once.
    """
    pending: deque[tuple[T, Future[R]]] = deque()
    pump = iter(items)
    for _ in range(max(1, ahead)):
        nxt = next(pump, None)
        if nxt is None:
            break
        pending.append((nxt, pool.submit(job, nxt)))
    while pending:
        item, future = pending.popleft()
        follow = next(pump, None)
        if follow is not None:
            pending.append((follow, pool.submit(job, follow)))
        yield item, future.result()


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
    cfg: DictConfig,
    session: Session,
    tracks: list[Track],
    force: bool = False,  # noqa: ARG001 — track selection is the caller's job
    fast: bool = False,
) -> dict[str, int]:
    """Extract features for ``tracks``.

    ``fast`` runs MERT only (enough for the recommender). Returns counts:
    ``ok`` extracted, ``failed`` errored.
    """
    from extraction.audio import load_audio
    from extraction.embeddings.mert import MertEmbedder
    from extraction.interpretable.librosa_extract import extract_interpretable

    chash = config_hash(cfg)
    audio_root = paths.resolve(cfg.paths.audio)
    sr_mert = int(cfg.extraction.target_sr_mert)
    sr_clap = int(cfg.extraction.target_sr_clap)
    mert_repo = str(cfg.models.mert.repo)
    clap_repo = str(cfg.models.clap.repo)

    counts = {"ok": 0, "failed": 0}
    todo = [t for t in tracks if t.audio_path]
    for track in tracks:
        if not track.audio_path:
            counts["failed"] += 1
            log.warning("extract.no_audio", track_id=track.id)
    if not todo:
        return counts

    mert_path = vectors.song_embedding_path(cfg, "mert_song")
    clap_path = vectors.song_embedding_path(cfg, "clap_song")
    mert_store = vectors.read_embeddings(mert_path)
    clap_store = vectors.read_embeddings(clap_path)

    mert = MertEmbedder(
        mert_repo,
        sample_rate=sr_mert,
        device=str(cfg.extraction.device),
        fp16=bool(cfg.extraction.fp16),
        chunk_seconds=int(cfg.extraction.mert.chunk_seconds),
        batch_size=int(cfg.extraction.mert.batch_size),
        seed=int(cfg.extraction.seed),
    )
    clap = None
    if not fast:
        from extraction.embeddings.clap import ClapEmbedder

        clap = ClapEmbedder(
            clap_repo,
            sample_rate=sr_clap,
            device=str(cfg.extraction.device),
            fp16=bool(cfg.extraction.fp16),
            chunk_seconds=int(cfg.extraction.clap.chunk_seconds),
            batch_size=int(cfg.extraction.clap.batch_size),
            seed=int(cfg.extraction.seed),
        )

    def decode(track: Track) -> tuple[Any, Any, str | None]:
        """Decode a track's audio (runs on the prefetch thread pool)."""
        try:
            audio_file = audio_root / str(track.audio_path)
            wav_mert = load_audio(audio_file, sr_mert)
            wav_clap = None if fast else load_audio(audio_file, sr_clap)
            return wav_mert, wav_clap, None
        except Exception as exc:
            return None, None, str(exc)

    clap_changed = False
    with ThreadPoolExecutor(max_workers=int(cfg.extraction.decode_workers)) as pool:
        stream = _prefetch(pool, todo, decode, int(cfg.extraction.prefetch))
        for track, (wav_mert, wav_clap, decode_error) in stream:
            if decode_error is not None:
                counts["failed"] += 1
                log.warning("extract.decode_failed", track_id=track.id, error=decode_error)
                continue
            try:
                mert_store[track.id] = vectors.EmbeddingRow(
                    track_id=track.id,
                    embedding=mert.embed_song(wav_mert),
                    model=mert_repo,
                    config_hash=chash,
                )
                if not fast:
                    assert clap is not None
                    _upsert_features(
                        session, track.id, chash, extract_interpretable(wav_mert, sr_mert)
                    )
                    clap_store[track.id] = vectors.EmbeddingRow(
                        track_id=track.id,
                        embedding=clap.embed_song(wav_clap),
                        model=clap_repo,
                        config_hash=chash,
                    )
                    clap_changed = True
                track.status = TrackStatus.extracted
                track.extracted_at = datetime.now(UTC)
                session.add(track)
                counts["ok"] += 1
                log.info("extract.ok", track_id=track.id, fast=fast)
            except Exception as exc:
                counts["failed"] += 1
                log.warning("extract.failed", track_id=track.id, error=str(exc))

    if counts["ok"]:
        vectors.write_embeddings(mert_path, mert_store.values())
        if clap_changed:
            vectors.write_embeddings(clap_path, clap_store.values())
        log.info("extract.written", mert=len(mert_store), clap=len(clap_store), fast=fast)
    return counts
