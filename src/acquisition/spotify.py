"""Spotify Web API wrapper (spotipy) — user library, playlists, listening data.

Read-only scopes only (see ``config/default.yaml``): library + playlists for
ingestion, plus top-tracks and recently-played for engagement weighting.
"""

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import spotipy
from omegaconf import DictConfig
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth

from acquisition.base import Provenance, TrackRef
from core import paths
from core.logging import get_logger

log = get_logger("spotify")


class SpotifyAuthError(RuntimeError):
    """Raised when Spotify credentials are missing."""


def _track_to_ref(track: dict[str, Any]) -> TrackRef:
    """Convert a Spotify track object into a :class:`TrackRef`."""
    return TrackRef(
        title=track["name"],
        artist=", ".join(a["name"] for a in track.get("artists", [])),
        album=(track.get("album") or {}).get("name"),
        duration_ms=track.get("duration_ms") or 0,
        spotify_id=track["id"],
    )


class SpotifyClient:
    """Authenticated Spotify client over the user's own library."""

    def __init__(self, cfg: DictConfig) -> None:
        client_id = os.environ.get("SPOTIFY_CLIENT_ID")
        client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise SpotifyAuthError(
                "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set. "
                "Copy .env.example to .env and fill them in."
            )

        cache_path = paths.resolve(cfg.paths.data) / ".spotify-token.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._auth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=str(cfg.spotify.redirect_uri),
            scope=" ".join(cfg.spotify.scopes),
            cache_handler=CacheFileHandler(cache_path=str(cache_path)),
            open_browser=True,
        )
        self._sp = spotipy.Spotify(auth_manager=self._auth)

    @property
    def sp(self) -> spotipy.Spotify:
        return self._sp

    def authenticate(self) -> dict[str, Any]:
        """Trigger the OAuth flow if needed and return the user profile."""
        profile: dict[str, Any] = self._sp.current_user()
        log.info("spotify.authenticated", user=profile.get("id"))
        return profile

    def iter_saved_tracks(self) -> Iterator[tuple[TrackRef, Provenance]]:
        """Yield every track in the user's saved ("Liked Songs") library."""
        saved = Provenance(source_type="saved")
        results = self._sp.current_user_saved_tracks(limit=50)
        while results is not None:
            for item in results.get("items", []):
                track = item.get("track")
                if track and track.get("id"):
                    yield _track_to_ref(track), saved
            results = self._sp.next(results) if results.get("next") else None

    def iter_playlist_tracks(self) -> Iterator[tuple[TrackRef, Provenance]]:
        """Yield every track across all of the user's playlists, with the
        playlist recorded as provenance."""
        playlists = self._sp.current_user_playlists(limit=50)
        while playlists is not None:
            for playlist in playlists.get("items", []):
                prov = Provenance(
                    source_type="playlist",
                    source_ref=str(playlist["id"]),
                    source_name=playlist.get("name"),
                )
                items = self._sp.playlist_items(playlist["id"], limit=100)
                while items is not None:
                    for item in items.get("items", []):
                        track = item.get("track")
                        if track and track.get("id") and track.get("type") == "track":
                            yield _track_to_ref(track), prov
                    items = self._sp.next(items) if items.get("next") else None
            playlists = self._sp.next(playlists) if playlists.get("next") else None

    def iter_library(self) -> Iterator[tuple[TrackRef, Provenance]]:
        """Yield saved tracks followed by all playlist tracks, each paired with
        its provenance."""
        yield from self.iter_saved_tracks()
        yield from self.iter_playlist_tracks()

    def iter_top_tracks(self) -> Iterator[tuple[TrackRef, str, int]]:
        """Yield the user's Spotify top tracks across all time ranges.

        Yields ``(track, term, rank)``. Iterated short -> medium -> long term
        so that, for a track in several ranges, the strongest term is seen last.
        """
        for term in ("short_term", "medium_term", "long_term"):
            offset = 0
            rank = 0
            while offset < 200:  # safety cap; the API tops out well before this
                page = self._sp.current_user_top_tracks(limit=50, offset=offset, time_range=term)
                items = page.get("items", [])
                if not items:
                    break
                for track in items:
                    if track.get("id"):
                        yield _track_to_ref(track), term, rank
                    rank += 1
                if not page.get("next"):
                    break
                offset += 50

    def recently_played(self) -> dict[str, datetime]:
        """Map ``spotify_id`` -> most recent play timestamp (naive UTC).

        The API window is shallow (last ~50 plays); this is a recency signal,
        not a play count.
        """
        out: dict[str, datetime] = {}
        page = self._sp.current_user_recently_played(limit=50)
        for item in page.get("items", []):
            track = item.get("track") or {}
            track_id = track.get("id")
            played_at = item.get("played_at")
            if not track_id or not played_at:
                continue
            ts = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
            ts = ts.astimezone(UTC).replace(tzinfo=None)
            if track_id not in out or ts > out[track_id]:
                out[track_id] = ts
        return out
