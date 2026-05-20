"""Parallel audio download.

Downloading is network-bound, so a thread pool of yt-dlp fetches gives a
near-linear speedup. This is the single-machine form of the multi-user
"download worker pool": workers do only network/disk work (search + fetch);
the caller serialises every database write.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from acquisition.base import AudioSource, TrackRef
from acquisition.resolver import pick_best_candidate
from core.logging import get_logger

log = get_logger("downloader")


@dataclass(slots=True)
class DownloadResult:
    """Outcome of one track's resolve-and-fetch attempt."""

    track_id: str
    ok: bool
    youtube_id: str | None = None
    audio_path: str | None = None
    error: str | None = None


def _fetch_one(
    source: AudioSource,
    track_id: str,
    ref: TrackRef,
    audio_dir: Path,
    tolerance: float,
) -> DownloadResult:
    """Resolve + download one track. Never raises — failures become a result."""
    try:
        best = pick_best_candidate(ref, source.search(ref), tolerance)
        if best is None:
            return DownloadResult(
                track_id, ok=False, error="no candidate within duration tolerance"
            )
        path = source.fetch(best, audio_dir)
        return DownloadResult(
            track_id,
            ok=True,
            youtube_id=best.source_id,
            audio_path=str(path.relative_to(audio_dir)),
        )
    except Exception as exc:
        return DownloadResult(track_id, ok=False, error=str(exc))


def download_tracks(
    source: AudioSource,
    jobs: list[tuple[str, TrackRef]],
    audio_dir: Path,
    tolerance: float = 0.10,
    workers: int = 8,
) -> Iterator[DownloadResult]:
    """Resolve + download a batch of tracks in parallel.

    ``jobs`` is ``(track_id, TrackRef)`` pairs. Yields one :class:`DownloadResult`
    per job as it completes (completion order, not input order). Does no
    database work — the caller persists each result.
    """
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(_fetch_one, source, track_id, ref, audio_dir, tolerance)
            for track_id, ref in jobs
        ]
        for future in as_completed(futures):
            yield future.result()
