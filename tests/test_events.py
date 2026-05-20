"""Tests for listening-event inference and ingestion."""

from datetime import datetime, timedelta

from sqlmodel import Session, select

from acquisition.events import InferredEvent, RecentPlay, infer_events, ingest_events
from storage.schema import EventType, ListeningEvent, Track

_T0 = datetime(2026, 5, 19, 12, 0, 0)


def _play(spotify_id: str, dur_ms: int, at: datetime) -> RecentPlay:
    return RecentPlay(spotify_id=spotify_id, duration_ms=dur_ms, played_at=at)


def test_skip_inferred_from_short_gap() -> None:
    # newest-first: B at T0, A started 30s earlier; A is a 200s track → skipped.
    plays = [_play("B", 200_000, _T0), _play("A", 200_000, _T0 - timedelta(seconds=30))]
    a = next(e for e in infer_events(plays) if e.spotify_id == "A")
    assert a.event_type == EventType.skip
    assert a.position_ms == 30_000
    assert abs((a.completion or 0) - 0.15) < 1e-6


def test_complete_inferred_from_full_gap() -> None:
    plays = [_play("B", 200_000, _T0), _play("A", 200_000, _T0 - timedelta(seconds=210))]
    a = next(e for e in infer_events(plays) if e.spotify_id == "A")
    assert a.event_type == EventType.complete
    assert a.completion == 1.0


def test_newest_play_has_unknown_completion() -> None:
    # The newest item has no successor, so we cannot tell if it finished.
    plays = [_play("B", 200_000, _T0), _play("A", 200_000, _T0 - timedelta(seconds=210))]
    b = next(e for e in infer_events(plays) if e.spotify_id == "B")
    assert b.event_type == EventType.play
    assert b.completion is None


def test_zero_duration_is_a_plain_play() -> None:
    plays = [_play("B", 200_000, _T0), _play("A", 0, _T0 - timedelta(seconds=30))]
    a = next(e for e in infer_events(plays) if e.spotify_id == "A")
    assert a.event_type == EventType.play


def _inferred(spotify_id: str, at: datetime) -> InferredEvent:
    return InferredEvent(spotify_id, EventType.complete, at, 200_000, 1.0, None)


def test_ingest_maps_spotify_id_and_skips_unknown(session: Session) -> None:
    session.add(Track(id="t1", spotify_id="A", title="x", artist="y"))
    session.commit()

    written = ingest_events(session, [_inferred("A", _T0), _inferred("Z", _T0)], source="spotify")

    rows = session.exec(select(ListeningEvent)).all()
    assert written == 1  # "Z" has no matching track and is dropped
    assert rows[0].track_id == "t1"


def test_ingest_is_idempotent_on_overlapping_windows(session: Session) -> None:
    session.add(Track(id="t1", spotify_id="A", title="x", artist="y"))
    session.commit()

    first = ingest_events(session, [_inferred("A", _T0)], source="spotify")
    session.commit()
    second = ingest_events(session, [_inferred("A", _T0)], source="spotify")

    assert (first, second) == (1, 0)
    assert len(session.exec(select(ListeningEvent)).all()) == 1
