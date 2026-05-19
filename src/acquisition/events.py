"""Listening-event ingestion — Spotify recently-played → silent signals.

The recently-played endpoint gives only a timestamp per play, never whether the
track finished. We *infer* skip vs. complete from the gap to the next play: if
the following track started before this one could have ended, this one was
skipped, and the gap is how far it got. The heuristic is imperfect — a long gap
is the end of a listening session, not a long listen — but it is the richest
signal the endpoint offers, and it is honest behavioural data, not a self-report.

The inference here is intentionally user-agnostic: no coefficient is tuned to
any one listener. It turns raw plays into events; `taste_model.engagement` turns
events into a per-user weight.
"""

from dataclasses import dataclass
from datetime import datetime

from sqlmodel import Session, select

from core.logging import get_logger
from storage.schema import EventType, ListeningEvent, Track

log = get_logger("events")

# Played within this fraction of its length ⇒ "complete" rather than "skip".
COMPLETE_FRACTION = 0.85


@dataclass(slots=True)
class RecentPlay:
    """One raw recently-played item. Lists of these are ordered newest-first."""

    spotify_id: str
    duration_ms: int
    played_at: datetime
    context: str | None = None


@dataclass(slots=True)
class InferredEvent:
    """A skip/complete/play event inferred from a :class:`RecentPlay`."""

    spotify_id: str
    event_type: EventType
    occurred_at: datetime
    position_ms: int | None
    completion: float | None
    context: str | None


def infer_events(plays: list[RecentPlay]) -> list[InferredEvent]:
    """Infer skip/complete events from recently-played items (newest-first).

    For each play, the *next* track to start is the preceding (newer) item, so
    how long this track actually played ≈ that gap. Gap ≥ most of the track ⇒
    complete; shorter ⇒ skip, with the gap as the skip position. The newest item
    has no successor, so its completion is unknown (a plain ``play``).
    """
    events: list[InferredEvent] = []
    for i, play in enumerate(plays):
        successor = plays[i - 1] if i > 0 else None
        if successor is None or play.duration_ms <= 0:
            events.append(
                InferredEvent(
                    play.spotify_id, EventType.play, play.played_at, None, None, play.context
                )
            )
            continue
        listened_ms = int((successor.played_at - play.played_at).total_seconds() * 1000)
        if listened_ms <= 0:
            events.append(
                InferredEvent(
                    play.spotify_id, EventType.play, play.played_at, None, None, play.context
                )
            )
        elif listened_ms >= play.duration_ms * COMPLETE_FRACTION:
            events.append(
                InferredEvent(
                    play.spotify_id,
                    EventType.complete,
                    play.played_at,
                    play.duration_ms,
                    1.0,
                    play.context,
                )
            )
        else:
            events.append(
                InferredEvent(
                    play.spotify_id,
                    EventType.skip,
                    play.played_at,
                    listened_ms,
                    listened_ms / play.duration_ms,
                    play.context,
                )
            )
    return events


def ingest_events(
    session: Session, inferred: list[InferredEvent], *, source: str = "spotify"
) -> int:
    """Persist inferred events, mapping ``spotify_id`` → ``track_id``.

    Events whose ``spotify_id`` is not in ``tracks`` are dropped — only library
    tracks carry a silent-signal history. Duplicates (same track/time/source,
    from an overlapping sync window) are skipped. Returns the new-row count.
    """
    spotify_ids = {e.spotify_id for e in inferred}
    if not spotify_ids:
        return 0
    id_map = {
        track.spotify_id: track.id
        for track in session.exec(
            select(Track).where(Track.spotify_id.in_(spotify_ids))  # type: ignore[attr-defined]
        ).all()
        if track.spotify_id
    }
    written = 0
    for event in inferred:
        track_id = id_map.get(event.spotify_id)
        if track_id is None:
            continue
        already = session.exec(
            select(ListeningEvent.id).where(  # type: ignore[attr-defined]
                ListeningEvent.track_id == track_id,
                ListeningEvent.occurred_at == event.occurred_at,
                ListeningEvent.source == source,
            )
        ).first()
        if already is not None:
            continue
        session.add(
            ListeningEvent(
                track_id=track_id,
                event_type=event.event_type,
                occurred_at=event.occurred_at,
                position_ms=event.position_ms,
                completion=event.completion,
                source=source,
                context=event.context,
            )
        )
        written += 1
    log.info("events.ingested", source=source, inferred=len(inferred), written=written)
    return written
