"""Candidate sampler — mixes the artist-graph and tag-graph crawl streams.

CLAUDE.md's pool target is ~80% artist-graph (tracks near the user's taste) and
~20% tag-graph (the serendipity stream). This combines the two, de-dupes across
them, and back-fills from whichever stream has more when the other runs short.
"""

from acquisition.base import TrackRef


def mix(
    artist_graph: list[TrackRef],
    tag_graph: list[TrackRef],
    target: int,
    artist_frac: float = 0.8,
) -> list[TrackRef]:
    """Combine the two crawl streams into a ~80/20 pool of up to ``target``."""
    seen: set[tuple[str, str]] = set()
    picked: list[TrackRef] = []

    def take(stream: list[TrackRef], count: int) -> None:
        added = 0
        for ref in stream:
            if added >= count or len(picked) >= target:
                return
            key = (ref.artist.lower(), ref.title.lower())
            if key in seen:
                continue
            seen.add(key)
            picked.append(ref)
            added += 1

    take(artist_graph, round(target * artist_frac))
    take(tag_graph, target - len(picked))
    # Back-fill from whichever stream still has unused tracks.
    take(artist_graph, target - len(picked))
    take(tag_graph, target - len(picked))
    return picked
