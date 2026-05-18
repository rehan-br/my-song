"""Tests for parallel audio download."""

from pathlib import Path

from acquisition.base import AudioCandidate, AudioSource, TrackRef
from acquisition.downloader import download_tracks


class _FakeSource(AudioSource):
    """An AudioSource that resolves deterministically, optionally failing some."""

    name = "fake"

    def __init__(self, fail: set[str] | None = None) -> None:
        self._fail = fail or set()

    def search(self, ref: TrackRef) -> list[AudioCandidate]:
        return [
            AudioCandidate(
                source="fake",
                source_id=f"vid-{ref.title}",
                title=ref.title,
                url="u",
                duration_ms=ref.duration_ms or 0,
            )
        ]

    def fetch(self, candidate: AudioCandidate, dest_dir: Path) -> Path:
        if candidate.source_id in self._fail:
            raise RuntimeError("fetch failed")
        return dest_dir / f"{candidate.source_id}.m4a"


def _jobs(n: int) -> list[tuple[str, TrackRef]]:
    return [(f"t{i}", TrackRef(title=str(i), artist="a", duration_ms=1000)) for i in range(n)]


def test_every_job_gets_a_result(tmp_path: Path) -> None:
    results = list(download_tracks(_FakeSource(), _jobs(6), tmp_path, workers=3))
    assert {r.track_id for r in results} == {f"t{i}" for i in range(6)}
    assert all(r.ok and r.youtube_id and r.audio_path for r in results)


def test_failures_are_reported_not_raised(tmp_path: Path) -> None:
    results = list(download_tracks(_FakeSource(fail={"vid-0"}), _jobs(3), tmp_path, workers=2))
    by_id = {r.track_id: r for r in results}
    assert not by_id["t0"].ok
    assert by_id["t0"].error
    assert by_id["t1"].ok and by_id["t2"].ok


def test_empty_job_list(tmp_path: Path) -> None:
    assert list(download_tracks(_FakeSource(), [], tmp_path)) == []
