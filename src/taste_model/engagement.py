"""Listening signals → automatic taste weighting.

Two layers:

- ``refresh_listening_signals`` records raw Spotify top-track / recency data
  onto each track — observed inputs, no weighting.
- ``engagement_weight`` / ``refresh_engagement_weights`` collapse the silent
  ``listening_events`` log into ``Track.taste_weight``. This is *not* a
  hand-crafted ingestion heuristic: the weight is consumed by the taste model
  (M1's weighted centroid, M2's weighted taste centroid), so weighting still
  belongs to the model — engagement just supplies a behavioural prior.

The formula is deliberately user-agnostic: a generic Bayesian shrinkage with
one regularisation constant, never tuned to any individual listener.
"""

from collections.abc import Mapping
from datetime import datetime

from omegaconf import DictConfig
from sqlmodel import Session, select

from storage.schema import ListeningEvent, Track


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


def engagement_weight(completions: list[float], prior_strength: float) -> float:
    """Bayesian-shrunk completion rate → a ``taste_weight`` in [0, 1].

    ``completions`` are observed completion fractions from ``listening_events``
    (1.0 for a finished play, the listened fraction for a skip). With few
    observations the weight stays near the 1.0 prior — "full influence until
    behaviour says otherwise"; as consistent events accumulate it converges to
    the observed mean. Play *count* enters only as confidence (less shrinkage),
    never as a separate boost. ``prior_strength`` is a generic regulariser.
    """
    if not completions:
        return 1.0
    n = len(completions)
    observed = sum(completions) / n
    return (prior_strength + n * observed) / (prior_strength + n)


def refresh_engagement_weights(session: Session, cfg: DictConfig) -> int:
    """Set ``Track.taste_weight`` from the silent ``listening_events`` log.

    Shrinks each track's observed completion behaviour toward the 1.0 prior and
    writes the result — but only for tracks whose weight is still automatic
    (``taste_weight_auto``); hand-pinned weights are left alone. Returns the
    number of tracks whose weight actually changed.
    """
    prior = float(cfg.taste.engagement.prior_strength)
    completions: dict[str, list[float]] = {}
    for event in session.exec(select(ListeningEvent)).all():
        if event.completion is not None:
            completions.setdefault(event.track_id, []).append(event.completion)

    changed = 0
    for track in session.exec(select(Track)).all():
        if not track.taste_weight_auto:
            continue
        weight = engagement_weight(completions.get(track.id, []), prior)
        if abs(track.taste_weight - weight) > 1e-9:
            track.taste_weight = weight
            session.add(track)
            changed += 1
    return changed
