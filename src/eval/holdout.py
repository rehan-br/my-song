"""Hold-out evaluation of the centroid recommender (recall@k, MAP).

Liked tracks are split into train + held-out; the M1 centroid is fitted on the
train split; the held-out liked tracks are then ranked against the crawled
candidate pool. A good model ranks the genuinely-liked held-out tracks above
the candidates. Averaged over several random splits for a stable estimate.

This is the quantitative half of the CLAUDE.md eval gate — a new model must
beat the previous one on hold-out recall@20 before it is promoted.
"""

from collections.abc import Mapping

import numpy as np

from eval.metrics import average_precision, recall_at_k
from taste_model.centroid import CentroidModel


def evaluate_holdout(
    liked: Mapping[str, np.ndarray],
    candidates: Mapping[str, np.ndarray],
    *,
    holdout_frac: float = 0.2,
    k: int = 20,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Run hold-out evaluation; return recall@k and MAP averaged over splits."""
    liked_ids = sorted(liked)
    cand_ids = sorted(candidates)
    if len(liked_ids) < 2:
        raise ValueError("hold-out evaluation needs at least 2 liked tracks")

    everything = [liked[t] for t in liked_ids] + [candidates[t] for t in cand_ids]
    space_mean = np.stack(everything).mean(axis=0)
    n_hold = max(1, round(len(liked_ids) * holdout_frac))
    rng = np.random.default_rng(seed)

    recalls: list[float] = []
    aps: list[float] = []
    for _ in range(n_splits):
        order = [str(t) for t in rng.permutation(liked_ids)]
        held, train = order[:n_hold], order[n_hold:]
        if not train:
            raise ValueError("holdout_frac too high — no training tracks left")

        model = CentroidModel().fit(np.stack([liked[t] for t in train]), space_mean=space_mean)
        pool_ids = held + cand_ids
        pool = np.stack([liked[t] for t in held] + [candidates[t] for t in cand_ids])
        ranked = [pool_ids[i] for i in np.argsort(-model.score(pool))]

        recalls.append(recall_at_k(ranked, held, k))
        aps.append(average_precision(ranked, held))

    return {
        "recall_at_k": float(np.mean(recalls)),
        "map": float(np.mean(aps)),
        "k": float(k),
        "holdout_frac": holdout_frac,
        "n_splits": float(n_splits),
        "n_liked": float(len(liked_ids)),
        "n_candidates": float(len(cand_ids)),
    }
