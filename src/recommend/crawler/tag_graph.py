"""Tag-graph candidate crawler — the 20% serendipity stream.

For each tag the user's library artists carry (Last.fm), this pulls that tag's
top tracks. It surfaces music that shares a *vibe / genre* with the library but
is not necessarily near it in the artist-similarity graph — the serendipity the
artist-graph walk alone would miss. Tracks by artists already in the library or
already found by the artist-graph walk are skipped.
"""

from typing import Protocol

from acquisition.base import TrackRef
from core.logging import get_logger

log = get_logger("crawler")


class TagSource(Protocol):
    """The Last.fm surface the tag crawler needs (see ``acquisition.lastfm``)."""

    def artist_tags(self, artist: str, limit: int = ...) -> list[str]: ...

    def tag_top_tracks(self, tag: str, limit: int = ...) -> list[tuple[str, str]]: ...


def crawl_tag_graph(
    client: TagSource,
    seed_artists: list[str],
    *,
    tags_per_artist: int = 3,
    tracks_per_tag: int = 20,
    known_artists: set[str] | None = None,
    reachable_artists: set[str] | None = None,
    target: int = 200,
) -> list[TrackRef]:
    """Collect candidate tracks from the tags the seed artists carry."""
    excluded = {a.lower() for a in (known_artists or set())}
    excluded |= {a.lower() for a in (reachable_artists or set())}

    tags: list[str] = []
    seen_tags: set[str] = set()
    for artist in seed_artists:
        for tag in client.artist_tags(artist, limit=tags_per_artist):
            if tag not in seen_tags:
                seen_tags.add(tag)
                tags.append(tag)

    candidates: list[TrackRef] = []
    seen_tracks: set[tuple[str, str]] = set()
    for tag in tags:
        for artist, title in client.tag_top_tracks(tag, limit=tracks_per_tag):
            if artist.lower() in excluded:
                continue
            key = (artist.lower(), title.lower())
            if key in seen_tracks:
                continue
            seen_tracks.add(key)
            candidates.append(TrackRef(title=title, artist=artist))
            if len(candidates) >= target:
                log.info("crawl.tag.target_reached", candidates=len(candidates))
                return candidates
    log.info("crawl.tag.done", tags=len(tags), candidates=len(candidates))
    return candidates
