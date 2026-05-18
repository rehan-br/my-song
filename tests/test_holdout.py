"""Tests for hold-out evaluation."""

import numpy as np
import pytest

from eval.holdout import evaluate_holdout


def test_separable_pool_scores_high() -> None:
    # liked tracks cluster near [1,0,0]; candidates near [0,0,1] — held-out
    # liked tracks should rank well above the candidates.
    rng = np.random.default_rng(0)
    liked = {f"L{i}": np.array([1.0, 0.0, 0.0]) + 0.05 * rng.standard_normal(3) for i in range(20)}
    candidates = {
        f"C{i}": np.array([0.0, 0.0, 1.0]) + 0.05 * rng.standard_normal(3) for i in range(20)
    }
    metrics = evaluate_holdout(liked, candidates, holdout_frac=0.25, k=10, n_splits=4, seed=1)
    assert metrics["recall_at_k"] > 0.8
    assert metrics["map"] > 0.8
    assert metrics["n_liked"] == 20.0


def test_indistinguishable_pool_scores_near_chance() -> None:
    # liked and candidates drawn from the same distribution — no real signal
    rng = np.random.default_rng(0)
    liked = {f"L{i}": rng.standard_normal(8) for i in range(20)}
    candidates = {f"C{i}": rng.standard_normal(8) for i in range(60)}
    metrics = evaluate_holdout(liked, candidates, holdout_frac=0.25, k=10, n_splits=5)
    assert metrics["map"] < 0.5  # well below the separable case


def test_too_few_liked_tracks_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        evaluate_holdout({"a": np.ones(3)}, {}, holdout_frac=0.2)
