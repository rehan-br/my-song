"""Tests for the extraction pipeline's prefetch helper."""

from concurrent.futures import ThreadPoolExecutor

from extraction.pipeline import _prefetch


def test_prefetch_yields_all_items_in_order() -> None:
    with ThreadPoolExecutor(max_workers=3) as pool:
        out = list(_prefetch(pool, [1, 2, 3, 4, 5], lambda x: x * 10, ahead=2))
    assert out == [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]


def test_prefetch_empty_input() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(_prefetch(pool, [], lambda x: x, ahead=3)) == []


def test_prefetch_ahead_larger_than_input() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        out = list(_prefetch(pool, [7], lambda x: x + 1, ahead=10))
    assert out == [(7, 8)]
