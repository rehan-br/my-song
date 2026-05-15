"""The AudioSource abstraction.

Invariant 1: all audio fetching goes through :class:`AudioSource`. yt-dlp is
one implementation; anything importing ``yt_dlp`` outside
``acquisition/youtube.py`` is a bug.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrackRef:
    """The metadata identity of a track we want — before audio is fetched."""

    title: str
    artist: str
    duration_ms: int | None = None
    album: str | None = None
    spotify_id: str | None = None
    youtube_id: str | None = None
    mbid: str | None = None


@dataclass(slots=True)
class Provenance:
    """How a track entered the library.

    Passed to the resolver so it can record provenance and seed the track's
    taste weight. ``source_type`` is one of "saved" / "playlist" / "manual"
    (kept as a plain str here so this module stays free of storage imports).
    """

    source_type: str
    source_ref: str = ""  # playlist id; "" for saved/manual
    source_name: str | None = None


@dataclass(slots=True)
class AudioCandidate:
    """A concrete fetchable result returned by an AudioSource search."""

    source: str
    source_id: str
    title: str
    url: str
    duration_ms: int = 0
    artist: str | None = None


class AudioSource(ABC):
    """Abstract audio provider: search for candidates, fetch audio to disk."""

    name: str = "base"

    @abstractmethod
    def search(self, ref: TrackRef) -> list[AudioCandidate]:
        """Return ranked fetch candidates for the given track reference."""

    @abstractmethod
    def fetch(self, candidate: AudioCandidate, dest_dir: Path) -> Path:
        """Download a candidate into ``dest_dir`` and return the file path."""
