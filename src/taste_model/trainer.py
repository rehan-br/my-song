"""Taste-model training — the ``music train`` orchestration.

M1 (centroid) needs no training. M2 (contrastive) trains a per-dimension weight
on positives (liked) vs negatives (crawled + skipped) tracks. M3 (manifold)
trains a VAE on the liked-track embedding distribution.
"""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sqlmodel import Session, select

from core import paths
from core.config import config_hash
from core.logging import get_logger
from recommend.rank import split_pool
from storage import vectors
from storage.schema import DEFAULT_USER_ID, EssenceSibling, Rating, TasteModelRun, Track
from taste_model.contrastive import ContrastiveModel
from taste_model.engagement import refresh_engagement_weights

log = get_logger("trainer")


def checkpoint_path(cfg: DictConfig, user_id: str = DEFAULT_USER_ID) -> Path:
    """Where the trained M2 (contrastive) weight checkpoint lives, per user."""
    return paths.resolve(cfg.paths.data) / "models" / user_id / "m2.npz"


def m3_checkpoint_path(cfg: DictConfig, user_id: str = DEFAULT_USER_ID) -> Path:
    """Where the trained M3 (manifold VAE) checkpoint lives, per user."""
    return paths.resolve(cfg.paths.data) / "models" / user_id / "m3.pt"


def train_contrastive(cfg: DictConfig, session: Session) -> dict[str, float]:
    """Train M2 on the current library + crawled pool; save the checkpoint.

    Refreshes engagement weights from any new listening events first, so a
    `music train` after an audition session actually folds in that feedback —
    matching the UI's promise.
    """
    refresh_engagement_weights(session, cfg)

    store = vectors.read_embeddings(vectors.song_embedding_path(cfg, "mert_song"))
    if not store:
        raise RuntimeError("no MERT embeddings — run `music extract` first")

    track_ids = sorted(store)
    liked_ids, crawl_ids = split_pool(session, track_ids)
    skipped = {
        rating.track_id for rating in session.exec(select(Rating).where(Rating.skip >= 4)).all()
    }

    positive_ids = [t for t in liked_ids if t not in skipped]
    negative_ids = sorted(set(crawl_ids) | (skipped & set(track_ids)))
    if not positive_ids:
        raise RuntimeError("no liked tracks to train on")
    if not negative_ids:
        raise RuntimeError("no negatives — crawl candidates (`music crawl`) or rate tracks first")

    space_mean = np.stack([store[t].embedding for t in track_ids]).mean(axis=0)

    # Engagement-derived taste weights tilt the centroid toward tracks the user
    # actually completes/replays — set by `sync-history`, uniform 1.0 until then.
    weight_by_id = {
        track.id: track.taste_weight
        for track in session.exec(select(Track).where(Track.id.in_(positive_ids))).all()  # type: ignore[attr-defined]
    }
    positive_weights = np.array([weight_by_id.get(t, 1.0) for t in positive_ids])

    m2 = cfg.taste.m2
    model = ContrastiveModel().fit(
        np.stack([store[t].embedding for t in positive_ids]),
        np.stack([store[t].embedding for t in negative_ids]),
        space_mean,
        positive_weights=positive_weights,
        epochs=int(m2.epochs),
        lr=float(m2.lr),
        tau=float(m2.temperature),
    )

    checkpoint = checkpoint_path(cfg)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)

    assert model.scale is not None
    metrics = {
        "n_positives": float(len(positive_ids)),
        "n_negatives": float(len(negative_ids)),
        "mean_positive_weight": float(positive_weights.mean()),
        "final_loss": model.final_loss,
        "scale_mean": float(model.scale.mean()),
        "scale_std": float(model.scale.std()),
    }
    session.add(
        TasteModelRun(
            version="m2-contrastive",
            config_hash=config_hash(cfg),
            checkpoint_path=str(checkpoint),
            metrics_json=metrics,
        )
    )
    log.info("train.m2.done", **metrics)
    return metrics


def train_manifold(cfg: DictConfig, session: Session) -> dict[str, float]:
    """Train M3 — a VAE over the liked-track embedding distribution."""
    from taste_model.manifold import ManifoldModel

    store = vectors.read_embeddings(vectors.song_embedding_path(cfg, "mert_song"))
    if not store:
        raise RuntimeError("no MERT embeddings — run `music extract` first")

    track_ids = sorted(store)
    liked_ids, _crawl = split_pool(session, track_ids)
    if not liked_ids:
        raise RuntimeError("no liked tracks to train on")

    liked = np.stack([store[t].embedding for t in liked_ids])
    space_mean = np.stack([store[t].embedding for t in track_ids]).mean(axis=0)

    # essence_siblings -> index pairs within the liked array (extra supervision).
    index = {tid: i for i, tid in enumerate(liked_ids)}
    sibling_pairs = [
        (index[s.track_a], index[s.track_b])
        for s in session.exec(select(EssenceSibling)).all()
        if s.track_a in index and s.track_b in index
    ]

    m3 = cfg.taste.m3
    model = ManifoldModel().fit(
        liked,
        space_mean,
        latent_dim=int(m3.latent_dim),
        hidden=int(m3.hidden),
        epochs=int(m3.epochs),
        lr=float(m3.lr),
        beta=float(m3.beta),
        sibling_pairs=sibling_pairs,
    )

    checkpoint = m3_checkpoint_path(cfg)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)

    metrics = {
        "n_liked": float(len(liked_ids)),
        "n_sibling_pairs": float(len(sibling_pairs)),
        "final_loss": model.final_loss,
    }
    session.add(
        TasteModelRun(
            version="m3-manifold",
            config_hash=config_hash(cfg),
            checkpoint_path=str(checkpoint),
            metrics_json=metrics,
        )
    )
    log.info("train.m3.done", **metrics)
    return metrics
