"""Tests for the per-user seam (single default user today)."""

from sqlmodel import Session, select

from storage.schema import DEFAULT_USER_ID, Rating, Track, TrackStatus, User
from storage.users import current_user_id, ensure_default_user


def test_current_user_id_returns_default() -> None:
    assert current_user_id() == DEFAULT_USER_ID


def test_ensure_default_user_is_idempotent(session: Session) -> None:
    # the session fixture already calls ensure_default_user, so this is a re-call
    ensure_default_user(session)
    session.commit()
    users = session.exec(select(User)).all()
    assert len(users) == 1
    assert users[0].id == DEFAULT_USER_ID


def test_user_scoped_rows_default_to_the_default_user(session: Session) -> None:
    # The whole point of the seam: existing code that doesn't know about users
    # writes rows that belong to the default user automatically.
    track = Track(title="t", artist="a", duration_ms=1000, status=TrackStatus.extracted)
    session.add(track)
    session.commit()

    session.add(Rating(track_id=track.id, vibe=4, replay=4, skip=2))
    session.commit()

    rating = session.exec(select(Rating)).one()
    assert rating.user_id == DEFAULT_USER_ID
