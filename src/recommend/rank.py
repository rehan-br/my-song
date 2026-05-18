"""Recommendation ranking — wires the taste model to the candidate pool.

Phase 2 / M1: the centroid is fitted on the user's *liked* tracks (their
library — any provenance except ``crawl``), and candidates are ranked by cosine
fit. If the crawler has added candidates (``crawl`` provenance) that are
extracted, those are what gets ranked — genuine discovery. With no crawled
candidates yet, the library itself is ranked (the M1 sanity baseline).
"""

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig
from sqlmodel import Session, select

from core.config import config_hash
from core.logging import get_logger
from storage import vectors
from storage.schema import TasteModelRun, Track, TrackSource
from taste_model.centroid import CentroidModel

log = get_logger("rank")


@dataclass(slots=True)
class Recommendation:
    """One ranked recommendation."""

    rank: int
    score: float
    track_id: str
    artist: str
    title: str


def _sources_by_track(session: Session) -> dict[str, set[str]]:
    """Map track_id -> set of its provenance source types."""
    out: dict[str, set[str]] = {}
    for ts in session.exec(select(TrackSource)).all():
        out.setdefault(ts.track_id, set()).add(str(ts.source_type))
    return out


def recommend(cfg: DictConfig, session: Session, top_k: int = 20) -> list[Recommendation]:
    """Rank candidates by M1 centroid fit and record a taste-model run.

    The centroid is the user's liked tracks (non-``crawl`` provenance) weighted
    by ``taste_weight``. Candidates are extracted ``crawl`` tracks if any exist,
    else the whole library.
    """
    store = vectors.read_embeddings(vectors.song_embedding_path(cfg, "mert_song"))
    if not store:
        raise RuntimeError("no MERT embeddings found — run `music extract` first")

    track_ids = sorted(store)
    matrix = np.stack([store[t].embedding for t in track_ids])
    index = {tid: i for i, tid in enumerate(track_ids)}
    tracks = {t.id: t for t in session.exec(select(Track).where(Track.id.in_(track_ids))).all()}
    sources = _sources_by_track(session)

    crawl_only = {t for t in track_ids if sources.get(t) == {"crawl"}}
    liked_ids = [t for t in track_ids if t not in crawl_only]
    if not liked_ids:
        raise RuntimeError("no liked tracks to build a taste centroid from")

    liked_rows = [index[t] for t in liked_ids]
    liked_weights = np.array([tracks[t].taste_weight if t in tracks else 1.0 for t in liked_ids])
    space_mean = matrix.mean(axis=0)
    model = CentroidModel().fit(matrix[liked_rows], liked_weights, space_mean)

    # Rank crawled candidates if we have any; otherwise rank the library.
    discovery = bool(crawl_only)
    candidate_ids = sorted(crawl_only) if discovery else track_ids
    cand_rows = [index[t] for t in candidate_ids]
    scores = model.score(matrix[cand_rows])
    order = np.argsort(-scores)

    recs: list[Recommendation] = []
    for position, rank_idx in enumerate(order[:top_k], start=1):
        tid = candidate_ids[rank_idx]
        track = tracks.get(tid)
        recs.append(
            Recommendation(
                rank=position,
                score=float(scores[rank_idx]),
                track_id=tid,
                artist=track.artist if track else "?",
                title=track.title if track else "?",
            )
        )

    session.add(
        TasteModelRun(
            version="m1-centroid",
            config_hash=config_hash(cfg),
            metrics_json={
                "mode": "discovery" if discovery else "library",
                "n_liked": len(liked_ids),
                "n_candidates": len(candidate_ids),
                "top_score": float(scores[order[0]]) if len(order) else 0.0,
            },
        )
    )
    log.info(
        "recommend.done",
        mode="discovery" if discovery else "library",
        liked=len(liked_ids),
        candidates=len(candidate_ids),
    )
    return recs
