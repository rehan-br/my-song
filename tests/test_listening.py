"""Tests for listening-session support — selection, auditions, ratings."""

from sqlmodel import Session, select

from eval.listening import (
    classify_audition,
    pick_session_tracks,
    record_audition,
    record_rating,
)
from storage.schema import EventType, ListeningEvent, Rating, Track, TrackStatus


def test_pick_session_tracks_excludes_rated_and_unextracted(session: Session) -> None:
    extracted = []
    for i in range(4):
        track = Track(title=f"t{i}", artist="a", duration_ms=1000, status=TrackStatus.extracted)
        session.add(track)
        extracted.append(track)
    session.add(Track(title="q", artist="a", duration_ms=1000, status=TrackStatus.queued))
    session.commit()

    record_rating(session, extracted[0].id, vibe=4, replay=4, skip=2)
    session.commit()

    picked = pick_session_tracks(session, count=10, seed=1)
    picked_ids = {t.id for t in picked}
    assert extracted[0].id not in picked_ids  # already rated
    assert len(picked) == 3  # the 3 unrated extracted tracks; queued excluded
    assert all(t.status is TrackStatus.extracted for t in picked)


def test_pick_session_tracks_respects_count(session: Session) -> None:
    for i in range(10):
        session.add(
            Track(title=f"t{i}", artist="a", duration_ms=1000, status=TrackStatus.extracted)
        )
    session.commit()
    assert len(pick_session_tracks(session, count=5, seed=1)) == 5


def test_record_rating_persists(session: Session) -> None:
    track = Track(title="t", artist="a", duration_ms=1000, status=TrackStatus.extracted)
    session.add(track)
    session.commit()

    record_rating(session, track.id, vibe=5, replay=4, skip=1, notes="warm")
    session.commit()

    rating = session.exec(select(Rating)).first()
    assert rating is not None
    assert rating.track_id == track.id
    assert rating.vibe == 5
    assert rating.notes == "warm"


def test_pick_session_tracks_excludes_locally_auditioned(session: Session) -> None:
    tracks = []
    for i in range(3):
        track = Track(title=f"t{i}", artist="a", duration_ms=1000, status=TrackStatus.extracted)
        session.add(track)
        tracks.append(track)
    session.commit()

    record_audition(session, tracks[0].id, EventType.complete, 1.0)
    session.commit()

    picked_ids = {t.id for t in pick_session_tracks(session, count=10, seed=1)}
    assert tracks[0].id not in picked_ids  # already auditioned locally
    assert len(picked_ids) == 2


def test_classify_audition_thumb_overrides_dwell() -> None:
    # an explicit thumb wins regardless of how long the track was on screen
    assert classify_audition(0.5, 200_000, thumb="up") == (EventType.complete, 1.0)
    down_type, down_completion = classify_audition(999.0, 200_000, thumb="down")
    assert down_type == EventType.skip
    assert down_completion is not None and down_completion < 0.5


def test_classify_audition_infers_skip_and_complete_from_dwell() -> None:
    # 20s on a 200s track → skip; 190s on a 200s track → complete
    skip_type, skip_completion = classify_audition(20.0, 200_000)
    assert skip_type == EventType.skip
    assert skip_completion is not None and abs(skip_completion - 0.1) < 1e-6

    complete_type, _ = classify_audition(190.0, 200_000)
    assert complete_type == EventType.complete


def test_classify_audition_too_short_to_judge_is_a_plain_play() -> None:
    event_type, completion = classify_audition(2.0, 200_000)
    assert event_type == EventType.play
    assert completion is None


def test_record_audition_writes_a_local_event(session: Session) -> None:
    track = Track(title="t", artist="a", duration_ms=1000, status=TrackStatus.extracted)
    session.add(track)
    session.commit()

    record_audition(session, track.id, EventType.skip, 0.2)
    session.commit()

    event = session.exec(select(ListeningEvent)).first()
    assert event is not None
    assert event.source == "local"
    assert event.event_type == EventType.skip
    assert event.completion == 0.2
