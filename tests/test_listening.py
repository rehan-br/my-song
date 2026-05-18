"""Tests for blind listening-session support."""

from sqlmodel import Session, select

from eval.listening import pick_session_tracks, record_rating
from storage.schema import Rating, Track, TrackStatus


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
