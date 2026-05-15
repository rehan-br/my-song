"""Spotify listening signals.

`sync` records Spotify top-track and recently-played data onto each track.
These are *observed signals*, not a weighting: they are inputs the taste model
and feedback loop (Phase 2+) will consume to decide how much a track matters.

Phase 0 deliberately does not collapse them into a ``taste_weight`` — that
would be a premature hand-crafted heuristic, and weighting belongs to the
actual taste model, not an ingestion-time formula.
"""

from collections.abc import Mapping
from datetime import datetime

from sqlmodel import Session, select

from storage.schema import Track


def refresh_listening_signals(
    session: Session,
    top_map: Mapping[str, tuple[str, int]],
    recent_map: Mapping[str, datetime],
) -> int:
    """Record current Spotify listening signals on every track.

    ``top_map`` maps ``spotify_id -> (top-track tier, rank)``; ``recent_map``
    maps ``spotify_id -> last played``. A track no longer present in
    ``top_map`` has its tier cleared, so the signal reflects current habits.

    Returns the number of tracks examined. Does not touch ``taste_weight``.
    """
    refreshed = 0
    for track in session.exec(select(Track)).all():
        spotify_id = track.spotify_id
        if not spotify_id:
            continue
        tier_rank = top_map.get(spotify_id)
        track.spotify_top_tier = tier_rank[0] if tier_rank else None
        track.spotify_top_rank = tier_rank[1] if tier_rank else None
        played = recent_map.get(spotify_id)
        if played is not None:
            track.last_played_at = played
        session.add(track)
        refreshed += 1
    return refreshed
