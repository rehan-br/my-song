"""Taste-model training — the ``music train`` orchestration.

M1 (centroid) needs no training. M2 (contrastive) trains a per-dimension weight
on positives (liked tracks) vs negatives (crawled candidates + tracks the user
skipped, ``skip >= 4``).
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
from storage.schema import Rating, TasteModelRun
from taste_model.contrastive import ContrastiveModel

log = get_logger("trainer")


def checkpoint_path(cfg: DictConfig) -> Path:
    """Where the trained M2 weight checkpoint lives."""
    return paths.resolve(cfg.paths.data) / "models" / "m2.npz"


def train_contrastive(cfg: DictConfig, session: Session) -> dict[str, float]:
    """Train M2 on the current library + crawled pool; save the checkpoint."""
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
    m2 = cfg.taste.m2
    model = ContrastiveModel().fit(
        np.stack([store[t].embedding for t in positive_ids]),
        np.stack([store[t].embedding for t in negative_ids]),
        space_mean,
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
