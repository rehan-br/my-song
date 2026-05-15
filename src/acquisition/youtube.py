"""yt-dlp implementation of :class:`AudioSource`.

Invariant 1: this is the ONLY module permitted to import ``yt_dlp``.
yt-dlp is used for personal/research feature extraction only — see the legal
posture in CLAUDE.md. Audio is cached locally and never redistributed.
"""

from pathlib import Path
from typing import Any

import yt_dlp
from omegaconf import DictConfig

from acquisition.base import AudioCandidate, AudioSource, TrackRef
from core.logging import get_logger

log = get_logger("youtube")


class YouTubeSource(AudioSource):
    """Searches and downloads audio from YouTube via yt-dlp."""

    name = "youtube"

    def __init__(self, cfg: DictConfig) -> None:
        yt = cfg.acquisition.youtube
        self._n_results = int(yt.search_results)
        self._audio_format = str(yt.audio_format)

    def search(self, ref: TrackRef) -> list[AudioCandidate]:
        query = f"{ref.artist} - {ref.title}"
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info: dict[str, Any] = (
                ydl.extract_info(f"ytsearch{self._n_results}:{query}", download=False) or {}
            )

        candidates: list[AudioCandidate] = []
        for entry in info.get("entries") or []:
            if not entry:
                continue
            duration = entry.get("duration") or 0
            video_id = str(entry["id"])
            candidates.append(
                AudioCandidate(
                    source=self.name,
                    source_id=video_id,
                    title=str(entry.get("title") or ""),
                    url=str(
                        entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
                    ),
                    duration_ms=int(float(duration) * 1000),
                    artist=entry.get("uploader"),
                )
            )
        log.info("youtube.search", query=query, hits=len(candidates))
        return candidates

    def fetch(self, candidate: AudioCandidate, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)

        cached = next(iter(dest_dir.glob(f"{candidate.source_id}.*")), None)
        if cached is not None:
            log.info("youtube.cache_hit", source_id=candidate.source_id)
            return cached

        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": self._audio_format}],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([candidate.url])

        produced = next(iter(dest_dir.glob(f"{candidate.source_id}.*")), None)
        if produced is None:
            raise RuntimeError(
                f"yt-dlp produced no file for {candidate.source_id} "
                "(is ffmpeg installed and on PATH?)"
            )
        log.info("youtube.fetched", source_id=candidate.source_id, path=str(produced))
        return produced
