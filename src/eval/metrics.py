"""Retrieval metrics for evaluating the recommender."""

from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the ``relevant`` items that appear in the top ``k`` of ``ranked``."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant_set)
    return hits / len(relevant_set)


def average_precision(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """Average precision of one ranking against the relevant set.

    Precision is accumulated at each rank where a relevant item is hit, then
    averaged over the relevant set — rewarding relevant items ranked early.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    cumulative = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant_set:
            hits += 1
            cumulative += hits / position
    return cumulative / len(relevant_set)
