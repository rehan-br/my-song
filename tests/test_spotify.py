"""Tests for Spotify search-match selection."""

from acquisition.spotify import _best_spotify_match


def _item(track_id: str, duration_ms: int) -> dict:
    return {"id": track_id, "uri": f"spotify:track:{track_id}", "duration_ms": duration_ms}


def test_best_match_picks_duration_closest() -> None:
    items = [_item("a", 180_000), _item("b", 200_000), _item("c", 240_000)]
    best = _best_spotify_match(items, duration_ms=205_000)
    assert best is not None and best["id"] == "b"


def test_best_match_rejects_all_off_duration() -> None:
    # every candidate is well over 10% off a 200s target → no viable match
    items = [_item("a", 100_000), _item("b", 300_000)]
    assert _best_spotify_match(items, duration_ms=200_000) is None


def test_best_match_without_target_duration_takes_first() -> None:
    # unknown target duration can't reject anything — take the top hit
    items = [_item("a", 180_000), _item("b", 200_000)]
    best = _best_spotify_match(items, duration_ms=0)
    assert best is not None and best["id"] == "a"


def test_best_match_empty_is_none() -> None:
    assert _best_spotify_match([], duration_ms=200_000) is None
