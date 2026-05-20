"""Tests for listening-signal recording and engagement weighting."""

from datetime import datetime

from omegaconf import OmegaConf
from sqlmodel import Session

from acquisition.base import Provenance, TrackRef
from acquisition.resolver import upsert_track
from storage.schema import EventType, ListeningEvent
from taste_model.engagement import (
    engagement_weight,
    refresh_engagement_weights,
    refresh_listening_signals,
)


def _engagement_cfg(prior: float = 3.0) -> OmegaConf:
    return OmegaConf.create({"taste": {"engagement": {"prior_strength": prior}}})


def _ref(spotify_id: str) -> TrackRef:
    return TrackRef(title="t", artist="a", spotify_id=spotify_id)


def test_refresh_records_top_tier_and_recency(session: Session) -> None:
    track, _ = upsert_track(session, _ref("sp-A"), Provenance("saved"))
    session.commit()

    played = datetime(2026, 5, 15, 12, 0, 0)
    n = refresh_listening_signals(session, {"sp-A": ("long_term", 3)}, {"sp-A": played})
    session.commit()

    assert n == 1
    assert track.spotify_top_tier == "long_term"
    assert track.spotify_top_rank == 3
    assert track.last_played_at == played


def test_refresh_decays_when_no_longer_a_top_track(session: Session) -> None:
    track, _ = upsert_track(session, _ref("sp-B"), Provenance("saved"))
    session.commit()

    refresh_listening_signals(session, {"sp-B": ("short_term", 0)}, {})
    session.commit()
    assert track.spotify_top_tier == "short_term"

    # a later sync: no longer a top track -> tier clears
    refresh_listening_signals(session, {}, {})
    session.commit()
    assert track.spotify_top_tier is None
    assert track.spotify_top_rank is None


def test_refresh_does_not_assign_taste_weight(session: Session) -> None:
    # weighting is the taste model's job — signal recording must not touch it
    track, _ = upsert_track(session, _ref("sp-C"), Provenance("playlist", "pl1"))
    session.commit()

    refresh_listening_signals(session, {"sp-C": ("long_term", 0)}, {})
    session.commit()
    assert track.taste_weight == 1.0  # untouched, uniform default
    assert track.taste_weight_auto is True


def test_refresh_skips_tracks_without_spotify_id(session: Session) -> None:
    track, _ = upsert_track(session, TrackRef(title="t", artist="a"))
    session.commit()

    assert refresh_listening_signals(session, {}, {}) == 0
    assert track.spotify_top_tier is None


def test_engagement_weight_no_events_is_full_influence() -> None:
    assert engagement_weight([], prior_strength=3.0) == 1.0


def test_engagement_weight_shrinks_toward_prior_with_few_events() -> None:
    # one low-completion event barely moves the weight off the 1.0 prior
    weight = engagement_weight([0.1], prior_strength=3.0)
    assert 0.7 < weight < 1.0


def test_engagement_weight_converges_to_observed_with_many_events() -> None:
    # consistent skipping, many events -> weight approaches the observed rate
    weight = engagement_weight([0.1] * 60, prior_strength=3.0)
    assert abs(weight - 0.1) < 0.05


def _skip_events(track_id: str, n: int, base_hour: int) -> list[ListeningEvent]:
    # distinct occurred_at — the (track, time, source) uniqueness constraint
    return [
        ListeningEvent(
            track_id=track_id,
            event_type=EventType.skip,
            occurred_at=datetime(2026, 5, 19, base_hour, 0, i),
            completion=0.1,
            source="spotify",
        )
        for i in range(n)
    ]


def test_refresh_engagement_weights_downweights_a_skipped_track(session: Session) -> None:
    track, _ = upsert_track(session, _ref("sp-D"), Provenance("saved"))
    session.commit()
    for event in _skip_events(track.id, 20, base_hour=12):
        session.add(event)
    session.commit()

    changed = refresh_engagement_weights(session, _engagement_cfg())
    session.commit()

    assert changed == 1
    assert track.taste_weight < 0.3  # consistent skipping pulls weight down


def test_refresh_engagement_weights_leaves_pinned_weights(session: Session) -> None:
    track, _ = upsert_track(session, _ref("sp-E"), Provenance("saved"))
    track.taste_weight_auto = False
    track.taste_weight = 0.5
    session.add(track)
    session.commit()
    for event in _skip_events(track.id, 10, base_hour=13):
        session.add(event)
    session.commit()

    refresh_engagement_weights(session, _engagement_cfg())
    session.commit()

    assert track.taste_weight == 0.5  # hand-pinned — engagement must not touch it


def test_refresh_engagement_weights_keeps_full_weight_without_events(session: Session) -> None:
    track, _ = upsert_track(session, _ref("sp-F"), Provenance("saved"))
    session.commit()

    refresh_engagement_weights(session, _engagement_cfg())
    session.commit()

    assert track.taste_weight == 1.0  # no behavioural evidence -> full influence
