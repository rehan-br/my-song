"""Tests for retrieval metrics."""

from eval.metrics import average_precision, recall_at_k


def test_recall_at_k() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, ["a", "c"], k=3) == 1.0  # both within top 3
    assert recall_at_k(ranked, ["a", "e"], k=3) == 0.5  # only "a" within top 3
    assert recall_at_k(ranked, ["d", "e"], k=3) == 0.0


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


def test_average_precision_rewards_early_hits() -> None:
    # relevant items at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
    ap = average_precision(["a", "x", "b", "y"], ["a", "b"])
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-9


def test_average_precision_perfect_and_miss() -> None:
    assert average_precision(["a", "b", "c"], ["a", "b"]) == 1.0
    assert average_precision(["x", "y", "z"], ["a"]) == 0.0
