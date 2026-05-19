"""Tests for the candidate crawler — artist-graph, tag-graph and sampler."""

from acquisition.base import TrackRef
from recommend.crawler.artist_graph import crawl_artist_graph
from recommend.crawler.sampler import mix
from recommend.crawler.tag_graph import crawl_tag_graph


class FakeLastfm:
    """A stand-in for LastfmClient — canned graph data."""

    def __init__(
        self,
        *,
        similar: dict[str, list[str]] | None = None,
        tracks: dict[str, list[str]] | None = None,
        tags: dict[str, list[str]] | None = None,
        tag_tracks: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        self._similar = similar or {}
        self._tracks = tracks or {}
        self._tags = tags or {}
        self._tag_tracks = tag_tracks or {}

    def similar_artists(self, artist: str, limit: int = 15) -> list[str]:
        return self._similar.get(artist, [])[:limit]

    def artist_top_track_titles(self, artist: str, limit: int = 10) -> list[str]:
        return self._tracks.get(artist, [])[:limit]

    def artist_tags(self, artist: str, limit: int = 5) -> list[str]:
        return self._tags.get(artist, [])[:limit]

    def tag_top_tracks(self, tag: str, limit: int = 20) -> list[tuple[str, str]]:
        return self._tag_tracks.get(tag, [])[:limit]


# --- artist graph --------------------------------------------------------
def test_collects_tracks_from_discovered_artists() -> None:
    fake = FakeLastfm(
        similar={"Seed": ["ArtistA", "ArtistB"], "ArtistA": ["ArtistC"]},
        tracks={"ArtistA": ["a1", "a2"], "ArtistB": ["b1"], "ArtistC": ["c1"]},
    )
    cands = crawl_artist_graph(fake, ["Seed"], depth=2, target=100)
    artists = {c.artist for c in cands}
    assert {"ArtistA", "ArtistB", "ArtistC"} <= artists
    assert "Seed" not in artists


def test_known_artists_are_skipped() -> None:
    fake = FakeLastfm(
        similar={"Seed": ["KnownArtist", "NewArtist"]},
        tracks={"KnownArtist": ["k1"], "NewArtist": ["n1"]},
    )
    cands = crawl_artist_graph(fake, ["Seed"], depth=1, known_artists={"knownartist"}, target=100)
    artists = {c.artist for c in cands}
    assert "NewArtist" in artists
    assert "KnownArtist" not in artists


def test_target_caps_the_candidate_count() -> None:
    fake = FakeLastfm(
        similar={"Seed": ["A", "B", "C"]},
        tracks={"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]},
    )
    assert len(crawl_artist_graph(fake, ["Seed"], depth=1, target=3)) == 3


def test_duplicate_tracks_are_deduped() -> None:
    fake = FakeLastfm(similar={"Seed": ["A"]}, tracks={"A": ["x", "x", "y"]})
    cands = crawl_artist_graph(fake, ["Seed"], depth=1, target=100)
    assert sorted(c.title for c in cands) == ["x", "y"]


# --- tag graph -----------------------------------------------------------
def test_tag_graph_collects_from_library_tags() -> None:
    fake = FakeLastfm(
        tags={"Seed": ["dreampop", "shoegaze"]},
        tag_tracks={
            "dreampop": [("Beach House", "Space Song")],
            "shoegaze": [("Slowdive", "Alison")],
        },
    )
    pairs = {(c.artist, c.title) for c in crawl_tag_graph(fake, ["Seed"], target=100)}
    assert ("Beach House", "Space Song") in pairs
    assert ("Slowdive", "Alison") in pairs


def test_tag_graph_excludes_known_and_reachable() -> None:
    fake = FakeLastfm(
        tags={"Seed": ["dreampop"]},
        tag_tracks={"dreampop": [("InLibrary", "x"), ("FromArtistGraph", "y"), ("FreshFind", "z")]},
    )
    cands = crawl_tag_graph(
        fake,
        ["Seed"],
        known_artists={"inlibrary"},
        reachable_artists={"fromartistgraph"},
        target=100,
    )
    assert {c.artist for c in cands} == {"FreshFind"}


# --- sampler -------------------------------------------------------------
def _refs(prefix: str, n: int) -> list[TrackRef]:
    return [TrackRef(title=f"{prefix}{i}", artist=f"{prefix}artist{i}") for i in range(n)]


def test_mix_is_roughly_80_20() -> None:
    picked = mix(_refs("A", 80), _refs("T", 80), target=50, artist_frac=0.8)
    assert len(picked) == 50
    assert sum(1 for r in picked if r.title.startswith("A")) == 40


def test_mix_backfills_when_a_stream_is_short() -> None:
    picked = mix(_refs("A", 80), _refs("T", 3), target=50, artist_frac=0.8)
    assert len(picked) == 50  # back-filled from the artist stream
    assert sum(1 for r in picked if r.title.startswith("T")) == 3


def test_mix_dedupes_across_streams() -> None:
    shared = TrackRef(title="dup", artist="dupartist")
    assert len(mix([shared], [shared], target=10)) == 1
