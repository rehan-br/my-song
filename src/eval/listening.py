"""Listening-session support — track selection and feedback capture.

Two feedback paths share this module:

- the CLI ``rate`` command writes explicit rubric :class:`Rating` rows;
- the Streamlit audition page writes silent :class:`ListeningEvent` rows — how
  long a track held attention, optionally corrected by a 👍/👎.

The testable pieces — which tracks to present, classifying an audition,
persisting feedback — live here; the interactive loops live in the CLI / UI.
"""

import random
from datetime import UTC, datetime

from sqlmodel import Session, select

from acquisition.events import COMPLETE_FRACTION
from storage.schema import EventType, ListeningEvent, Rating, Track, TrackStatus

# Below this many seconds, advancing past a track tells us nothing useful.
MIN_DWELL_S = 5.0


def pick_session_tracks(session: Session, count: int, seed: int | None = None) -> list[Track]:
    """Pick up to ``count`` extracted tracks with no feedback yet, random order.

    "No feedback" means neither an explicit :class:`Rating` nor a local audition
    (a ``source='local'`` :class:`ListeningEvent`). Random order avoids any
    ordering bias in the session.
    """
    rated = set(session.exec(select(Rating.track_id)).all())
    auditioned = set(
        session.exec(
            select(ListeningEvent.track_id).where(ListeningEvent.source == "local")
        ).all()
    )
    seen = rated | auditioned
    pool = [
        track
        for track in session.exec(select(Track).where(Track.status == TrackStatus.extracted)).all()
        if track.id not in seen
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
    """Persist one rubric rating for a track (the explicit CLI feedback path)."""
    rating = Rating(track_id=track_id, vibe=vibe, replay=replay, skip=skip, notes=notes)
    session.add(rating)
    return rating


def classify_audition(
    dwell_s: float, duration_ms: int, thumb: str | None = None
) -> tuple[EventType, float | None]:
    """Map an audition (dwell time + optional 👍/👎) to an event type + completion.

    An explicit ``thumb`` ('up'/'down') wins — that is the user telling us
    directly. Otherwise dwell vs. track length infers skip vs. complete; a dwell
    too short to mean anything yields a low-confidence plain ``play``.
    """
    if thumb == "up":
        return EventType.complete, 1.0
    if thumb == "down":
        return EventType.skip, 0.1
    if dwell_s < MIN_DWELL_S or duration_ms <= 0:
        return EventType.play, None
    completion = min(dwell_s * 1000.0 / duration_ms, 1.0)
    if completion >= COMPLETE_FRACTION:
        return EventType.complete, 1.0
    return EventType.skip, completion


def record_audition(
    session: Session, track_id: str, event_type: EventType, completion: float | None
) -> ListeningEvent:
    """Persist one local audition as a silent listening event."""
    event = ListeningEvent(
        track_id=track_id,
        event_type=event_type,
        occurred_at=datetime.now(UTC).replace(tzinfo=None),
        completion=completion,
        source="local",
    )
    session.add(event)
    return event
