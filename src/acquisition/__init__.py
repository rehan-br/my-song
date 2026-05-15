"""Audio acquisition layer.

Invariant 1: all audio fetching goes through the :class:`AudioSource` ABC.
Nothing outside ``acquisition/youtube.py`` may import ``yt_dlp``.
"""

from acquisition.base import AudioCandidate, AudioSource, Provenance, TrackRef

__all__ = ["AudioCandidate", "AudioSource", "Provenance", "TrackRef"]
