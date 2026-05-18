"""Blind listening-session support — track selection + rating capture.

The interactive loop (audio playback, prompts) lives in the ``rate`` CLI
command; the testable pieces — which tracks to present, persisting a rating —
live here. The rubric is vibe / replay / skip, each 1–5 (CLAUDE.md data model).
"""

import random

from sqlmodel import Session, select

from storage.schema import Rating, Track, TrackStatus


def pick_session_tracks(session: Session, count: int, seed: int | None = None) -> list[Track]:
    """Pick up to ``count`` extracted, not-yet-rated tracks, in random order.

    Random order keeps the session *blind* — the order carries no signal.
    """
    rated = set(session.exec(select(Rating.track_id)).all())
    pool = [
        track
        for track in session.exec(select(Track).where(Track.status == TrackStatus.extracted)).all()
        if track.id not in rated
    ]
    random.Random(seed).shuffle(pool)
    return pool[:count]


def record_rating(
    session: Session,
    track_id: str,
    vibe: int,
    replay: int,
    skip: int,
    notes: str | None = None,
) -> Rating:
    """Persist one rubric rating for a track."""
    rating = Rating(track_id=track_id, vibe=vibe, replay=replay, skip=skip, notes=notes)
    session.add(rating)
    return rating
