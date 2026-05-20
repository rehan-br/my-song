"""Artist-graph candidate crawler.

BFS from the user's top artists through the Last.fm similar-artist graph
(Spotify's related-artists endpoint is 403 for our Development-mode app — the
fallback CLAUDE.md anticipated). Each newly discovered artist contributes its
top tracks as candidates for the recommendation pool.

The 20% tag-graph stream is a Phase 4 addition; this is the Phase 2 (80%)
artist-graph walk.
"""

from typing import Protocol

from acquisition.base import TrackRef
from core.logging import get_logger

log = get_logger("crawler")


class SimilarArtistSource(Protocol):
    """The Last.fm surface the crawler needs (see ``acquisition.lastfm``)."""

    def similar_artists(self, artist: str, limit: int = ...) -> list[str]: ...

    def artist_top_track_titles(self, artist: str, limit: int = ...) -> list[str]: ...


def crawl_artist_graph(
    client: SimilarArtistSource,
    seed_artists: list[str],
    *,
    depth: int = 2,
    similar_per_artist: int = 15,
    tracks_per_artist: int = 10,
    known_artists: set[str] | None = None,
    target: int = 500,
) -> list[TrackRef]:
    """Walk the similar-artist graph from ``seed_artists`` and collect tracks.

    Discovery and collection are interleaved: a newly found artist's top tracks
    are gathered immediately, and the crawl stops as soon as ``target`` distinct
    candidate tracks are reached (so a small target doesn't over-expand the
    graph). Seed artists and ``known_artists`` (the existing library, lowercased)
    are skipped for collection — only genuinely new artists yield candidates.
    """
    known = {a.lower() for a in (known_artists or set())}
    seen_artists = {a.lower() for a in seed_artists}
    seen_tracks: set[tuple[str, str]] = set()
    candidates: list[TrackRef] = []
    frontier = list(seed_artists)

    for level in range(depth):
        next_frontier: list[str] = []
        for artist in frontier:
            for similar in client.similar_artists(artist, limit=similar_per_artist):
                key = similar.lower()
                if key in seen_artists:
                    continue
                seen_artists.add(key)
                next_frontier.append(similar)
                if key in known:
                    continue
                for title in client.artist_top_track_titles(similar, limit=tracks_per_artist):
                    track_key = (key, title.lower())
                    if track_key in seen_tracks:
                        continue
                    seen_tracks.add(track_key)
                    candidates.append(TrackRef(title=title, artist=similar))
                    if len(candidates) >= target:
                        log.info("crawl.target_reached", candidates=len(candidates))
                        return candidates
        log.info("crawl.level", level=level + 1, new_artists=len(next_frontier))
        frontier = next_frontier

    return candidates
