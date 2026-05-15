"""Tests for the data schema (storage/schema.py)."""

import pytest
from sqlmodel import Session, select

from storage.schema import EssenceSibling, Track, TrackStatus


def test_track_gets_uuid_and_defaults(session: Session) -> None:
    track = Track(title="Weird Fishes", artist="Radiohead", duration_ms=318_000)
    session.add(track)
    session.commit()
    session.refresh(track)

    assert track.id  # a UUID was assigned
    assert track.status is TrackStatus.queued
    assert track.added_at is not None
    assert track.extracted_at is None


def test_multiple_tracks_may_have_null_external_ids(session: Session) -> None:
    # SQLite unique columns permit multiple NULLs — manually-added tracks
    # have no spotify_id until resolved.
    for i in range(3):
        session.add(Track(title=f"track {i}", artist="artist", duration_ms=1_000))
    session.commit()
    assert len(session.exec(select(Track)).all()) == 3


def test_essence_sibling_create_orders_the_pair() -> None:
    sibling = EssenceSibling.create("zzz", "aaa", strength=0.9)
    assert sibling.track_a == "aaa"
    assert sibling.track_b == "zzz"


def test_essence_sibling_create_rejects_self_pair() -> None:
    with pytest.raises(ValueError, match="distinct"):
        EssenceSibling.create("x", "x", strength=0.5)
