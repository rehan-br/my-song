"""Last.fm tag lookup (pylast).

Last.fm crowd-sourced tags are the genre/mood signal for the candidate crawler
(Phase 4) and for colouring sanity plots. Read-only: needs only
``LASTFM_API_KEY`` (no scrobbling, no auth flow).
"""

import os

import pylast

from core.logging import get_logger

log = get_logger("lastfm")


class LastfmError(RuntimeError):
    """Raised when the Last.fm API key is missing."""


class LastfmClient:
    """Thin pylast wrapper for fetching crowd-sourced tags."""

    def __init__(self) -> None:
        api_key = os.environ.get("LASTFM_API_KEY")
        if not api_key:
            raise LastfmError(
                "LASTFM_API_KEY not set. Create a free key at "
                "https://www.last.fm/api/account/create and add it to .env."
            )
        self._net = pylast.LastFMNetwork(
            api_key=api_key,
            api_secret=os.environ.get("LASTFM_API_SECRET", ""),
        )

    def artist_tags(self, artist: str, limit: int = 5) -> list[str]:
        """Return an artist's top tags (lowercased), most popular first."""
        try:
            tags = self._net.get_artist(artist).get_top_tags(limit=limit)
        except pylast.WSError as exc:
            log.warning("lastfm.artist_miss", artist=artist, error=str(exc))
            return []
        return [t.item.get_name().lower() for t in tags]

    def track_tags(self, artist: str, title: str, limit: int = 5) -> list[str]:
        """Return a track's top tags (lowercased), most popular first."""
        try:
            tags = self._net.get_track(artist, title).get_top_tags(limit=limit)
        except pylast.WSError as exc:
            log.warning("lastfm.track_miss", artist=artist, title=title, error=str(exc))
            return []
        return [t.item.get_name().lower() for t in tags]
