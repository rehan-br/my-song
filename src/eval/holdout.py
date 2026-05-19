"""Hold-out evaluation of a taste model (recall@k, MAP).

Per split, *both* pools are held out:
- liked tracks → train-positives + held-out (the relevant items to recover);
- crawled candidates → train-negatives + eval-negatives.

A model is fitted on (train-positives, train-negatives), then the held-out
liked tracks are ranked against the **eval-negatives** — negatives the model
never trained on. This avoids the leak where a contrastive model is rewarded
for down-ranking the very tracks it was trained against. Averaged over several
random splits.

Model-agnostic: the caller supplies a ``fit_fn`` so the same harness evaluates
M1 (centroid) or M2 (contrastive). The quantitative half of the CLAUDE.md eval
gate — a new model must beat the previous on hold-out recall@20.
"""

from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np

from eval.metrics import average_precision, recall_at_k


class Scorer(Protocol):
    """A fitted taste model — scores candidate embeddings."""

    def score(self, candidates: np.ndarray) -> np.ndarray: ...


# (train positives, train negatives, space mean) -> fitted Scorer
FitFn = Callable[[np.ndarray, np.ndarray, np.ndarray], Scorer]


def _default_fit(positives: np.ndarray, negatives: np.ndarray, space_mean: np.ndarray) -> Scorer:
    """M1 centroid fit — ignores negatives."""
    from taste_model.centroid import CentroidModel

    return CentroidModel().fit(positives, space_mean=space_mean)


def evaluate_holdout(
    liked: Mapping[str, np.ndarray],
    candidates: Mapping[str, np.ndarray],
    *,
    fit_fn: FitFn | None = None,
    holdout_frac: float = 0.2,
    k: int = 20,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Run leak-free hold-out evaluation; return recall@k and MAP over splits."""
    fit = fit_fn or _default_fit
    liked_ids = sorted(liked)
    cand_ids = sorted(candidates)
    if len(liked_ids) < 2:
        raise ValueError("hold-out evaluation needs at least 2 liked tracks")

    everything = [liked[t] for t in liked_ids] + [candidates[t] for t in cand_ids]
    dim = np.stack(everything).shape[1]
    space_mean = np.stack(everything).mean(axis=0)
    n_hold = max(1, round(len(liked_ids) * holdout_frac))
    rng = np.random.default_rng(seed)

    def _stack(ids: list[str], pool: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.stack([pool[t] for t in ids]) if ids else np.empty((0, dim))

    recalls: list[float] = []
    aps: list[float] = []
    for _ in range(n_splits):
        liked_order = [str(t) for t in rng.permutation(liked_ids)]
        held, train_pos = liked_order[:n_hold], liked_order[n_hold:]
        if not train_pos:
            raise ValueError("holdout_frac too high — no training tracks left")

        # Split the candidate pool: half train the model, half are eval-only.
        cand_order = [str(t) for t in rng.permutation(cand_ids)] if cand_ids else []
        cut = len(cand_order) // 2
        train_neg, eval_neg = cand_order[:cut], cand_order[cut:]

        model = fit(_stack(train_pos, liked), _stack(train_neg, candidates), space_mean)

        pool_ids = held + eval_neg
        pool = np.concatenate([_stack(held, liked), _stack(eval_neg, candidates)])
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
