"""Tests for the artist-graph candidate crawler."""

from recommend.crawler.artist_graph import crawl_artist_graph


class FakeLastfm:
    """A stand-in for LastfmClient — canned similar-artist + top-track data."""

    def __init__(self, similar: dict[str, list[str]], tracks: dict[str, list[str]]) -> None:
        self._similar = similar
        self._tracks = tracks

    def similar_artists(self, artist: str, limit: int = 15) -> list[str]:
        return self._similar.get(artist, [])[:limit]

    def artist_top_track_titles(self, artist: str, limit: int = 10) -> list[str]:
        return self._tracks.get(artist, [])[:limit]


def test_collects_tracks_from_discovered_artists() -> None:
    fake = FakeLastfm(
        similar={"Seed": ["ArtistA", "ArtistB"], "ArtistA": ["ArtistC"]},
        tracks={"ArtistA": ["a1", "a2"], "ArtistB": ["b1"], "ArtistC": ["c1"]},
    )
    cands = crawl_artist_graph(fake, ["Seed"], depth=2, target=100)

    artists = {c.artist for c in cands}
    assert {"ArtistA", "ArtistB", "ArtistC"} <= artists
    assert "Seed" not in artists  # the seed itself is never collected
    assert ("ArtistA", "a1") in {(c.artist, c.title) for c in cands}


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
    cands = crawl_artist_graph(fake, ["Seed"], depth=1, target=3)
    assert len(cands) == 3


def test_duplicate_tracks_are_deduped() -> None:
    # two artists list the same (artist, title) is impossible, but one artist
    # listing a title twice must yield it once
    fake = FakeLastfm(similar={"Seed": ["A"]}, tracks={"A": ["x", "x", "y"]})
    cands = crawl_artist_graph(fake, ["Seed"], depth=1, target=100)
    assert sorted(c.title for c in cands) == ["x", "y"]
