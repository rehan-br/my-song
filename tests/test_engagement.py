"""Tests for Spotify listening-signal recording.

These signals are observed data; `sync` records them but does not derive a
taste weight from them — weighting is the taste model's job.
"""

from datetime import datetime

from sqlmodel import Session

from acquisition.base import Provenance, TrackRef
from acquisition.resolver import upsert_track
from taste_model.engagement import refresh_listening_signals


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
