"""Tests for cross-source track resolution."""

from sqlmodel import Session

from acquisition.base import AudioCandidate, TrackRef
from acquisition.resolver import (
    duration_matches,
    find_existing,
    pick_best_candidate,
    upsert_track,
)


def _candidate(source_id: str, duration_ms: int) -> AudioCandidate:
    return AudioCandidate(
        source="youtube",
        source_id=source_id,
        title=source_id,
        url=f"https://youtu.be/{source_id}",
        duration_ms=duration_ms,
    )


def test_duration_matches_within_tolerance() -> None:
    assert duration_matches(200_000, 210_000, 0.10)
    assert not duration_matches(200_000, 240_000, 0.10)


def test_duration_matches_unknown_is_not_rejected() -> None:
    # cannot verify -> do not reject
    assert duration_matches(None, 200_000, 0.10)
    assert duration_matches(200_000, 0, 0.10)


def test_pick_best_candidate_prefers_closest_duration() -> None:
    ref = TrackRef(title="t", artist="a", duration_ms=200_000)
    best = pick_best_candidate(ref, [_candidate("far", 218_000), _candidate("near", 205_000)], 0.10)
    assert best is not None
    assert best.source_id == "near"


def test_pick_best_candidate_returns_none_when_all_off() -> None:
    ref = TrackRef(title="t", artist="a", duration_ms=200_000)
    assert pick_best_candidate(ref, [_candidate("live", 400_000)], 0.10) is None


def test_upsert_track_creates_then_dedupes_by_spotify_id(session: Session) -> None:
    ref = TrackRef(
        title="Weird Fishes",
        artist="Radiohead",
        duration_ms=318_000,
        spotify_id="sp-1",
    )
    track, created = upsert_track(session, ref)
    session.commit()
    assert created

    again, created_again = upsert_track(session, ref)
    assert not created_again
    assert again.id == track.id


def test_upsert_backfills_missing_ids(session: Session) -> None:
    track, _ = upsert_track(session, TrackRef(title="t", artist="a", spotify_id="sp-2"))
    session.commit()

    upsert_track(session, TrackRef(title="t", artist="a", spotify_id="sp-2", mbid="mb-9"))
    session.commit()

    found = find_existing(session, spotify_id="sp-2")
    assert found is not None
    assert found.id == track.id
    assert found.mbid == "mb-9"
