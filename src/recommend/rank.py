"""Recommendation ranking — wires the taste model to the candidate pool.

The centroid/contrastive model is fitted (or loaded) over the user's *liked*
tracks (their library — any provenance except ``crawl``); candidates are then
ranked by fit. If the crawler has added extracted ``crawl`` candidates, those
are what gets ranked — genuine discovery — otherwise the library itself is
ranked (the sanity baseline).

Model selection: ``auto`` uses the trained M2 if a checkpoint exists, else M1.
"""

from dataclasses import dataclass
from typing import Any

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


def split_pool(session: Session, track_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split track ids into ``(liked, crawl_candidates)`` by provenance.

    A track is a *candidate* if its only provenance is ``crawl`` (discovered,
    never in the user's library); everything else is *liked* and feeds the
    taste model. Shared by the recommender and the eval harness.
    """
    sources: dict[str, set[str]] = {}
    for track_source in session.exec(select(TrackSource)).all():
        sources.setdefault(track_source.track_id, set()).add(str(track_source.source_type))
    liked, candidates = [], []
    for track_id in track_ids:
        if sources.get(track_id) == {"crawl"}:
            candidates.append(track_id)
        else:
            liked.append(track_id)
    return liked, candidates


def _build_scorer(
    cfg: DictConfig,
    model: str,
    matrix: np.ndarray,
    index: dict[str, int],
    liked_ids: list[str],
    tracks: dict[str, Track],
) -> tuple[str, Any]:
    """Return ``(model_name, fitted_scorer)`` for the requested model.

    ``auto`` picks the trained M2 if its checkpoint exists, else M1.
    """
    from taste_model.trainer import checkpoint_path

    checkpoint = checkpoint_path(cfg)
    want_m2 = model in ("contrastive", "m2") or (model == "auto" and checkpoint.exists())
    if model in ("contrastive", "m2") and not checkpoint.exists():
        raise RuntimeError("no trained M2 — run `music train` first")

    if want_m2:
        from taste_model.contrastive import ContrastiveModel

        return "m2-contrastive", ContrastiveModel.load(checkpoint)

    liked_rows = [index[t] for t in liked_ids]
    weights = np.array([tracks[t].taste_weight if t in tracks else 1.0 for t in liked_ids])
    centroid = CentroidModel().fit(matrix[liked_rows], weights, matrix.mean(axis=0))
    return "m1-centroid", centroid


def recommend(
    cfg: DictConfig,
    session: Session,
    top_k: int = 20,
    model: str = "auto",
    composite: bool = False,
) -> list[Recommendation]:
    """Rank candidates by taste-model fit and record a taste-model run.

    With ``composite``, the MERT taste score is blended with a CLAP fit score.
    """
    store = vectors.read_embeddings(vectors.song_embedding_path(cfg, "mert_song"))
    if not store:
        raise RuntimeError("no MERT embeddings found — run `music extract` first")

    track_ids = sorted(store)
    matrix = np.stack([store[t].embedding for t in track_ids])
    index = {tid: i for i, tid in enumerate(track_ids)}
    tracks = {t.id: t for t in session.exec(select(Track).where(Track.id.in_(track_ids))).all()}
    liked_ids, crawl_ids = split_pool(session, track_ids)
    if not liked_ids:
        raise RuntimeError("no liked tracks to build a taste model from")

    model_name, scorer = _build_scorer(cfg, model, matrix, index, liked_ids, tracks)

    # Rank crawled candidates if we have any; otherwise rank the library.
    discovery = bool(crawl_ids)
    candidate_ids = sorted(crawl_ids) if discovery else track_ids
    scores = scorer.score(matrix[[index[t] for t in candidate_ids]])
    if composite:
        from recommend.composite import blend, clap_fit_scores

        weights = cfg.taste.composite
        clap_scores, has_clap = clap_fit_scores(cfg, liked_ids, candidate_ids)
        scores = blend(
            scores,
            clap_scores,
            has_clap,
            float(weights.mert_weight),
            float(weights.clap_weight),
        )
        model_name = f"{model_name}+clap"
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
            version=model_name,
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
        model=model_name,
        mode="discovery" if discovery else "library",
        liked=len(liked_ids),
        candidates=len(candidate_ids),
    )
    return recs
