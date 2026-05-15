"""Phase 0 end-to-end smoke test.

Skipped unless real Spotify credentials are exported into the environment.
Run it explicitly (it opens a browser for the OAuth flow on first use):

    uv run pytest -m e2e

The everyday smoke check is the CLI path documented in the README:
``music sync`` then ``music download``.
"""

import itertools
import os

import pytest

pytestmark = pytest.mark.e2e

_HAVE_CREDS = bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"))


@pytest.mark.skipif(not _HAVE_CREDS, reason="Spotify credentials not in environment")
def test_pull_saved_library() -> None:
    from acquisition.spotify import SpotifyClient
    from core.config import load_config

    client = SpotifyClient(load_config())
    refs = list(itertools.islice(client.iter_saved_tracks(), 10))

    assert refs, "expected at least one saved track in the library"
    assert all(ref.spotify_id for ref in refs)
    assert all(ref.title and ref.artist for ref in refs)
