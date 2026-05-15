"""Cross-source track resolution: dedupe, duration sanity check, upsert.

The duration check guards against yt-dlp picking the wrong track (live cuts,
covers, fan edits) — see the gotcha in CLAUDE.md.
"""

from sqlmodel import Session, select

from acquisition.base import AudioCandidate, TrackRef
from storage.schema import Track, TrackStatus


def duration_matches(
    expected_ms: int | None, actual_ms: int | None, tolerance: float = 0.10
) -> bool:
    """True if ``actual_ms`` is within ``tolerance`` of ``expected_ms``.

    If either duration is unknown/zero, returns True — the gotcha is about
    *rejecting* bad matches, and an unverifiable duration is not evidence of one.
    """
    if not expected_ms or not actual_ms:
        return True
    return abs(actual_ms - expected_ms) / expected_ms <= tolerance


def pick_best_candidate(
    ref: TrackRef, candidates: list[AudioCandidate], tolerance: float = 0.10
) -> AudioCandidate | None:
    """Return the duration-closest candidate that passes the sanity check."""
    viable = [c for c in candidates if duration_matches(ref.duration_ms, c.duration_ms, tolerance)]
    if not viable:
        return None
    if ref.duration_ms:
        target = ref.duration_ms
        viable.sort(key=lambda c: abs(c.duration_ms - target))
    return viable[0]


def find_existing(
    session: Session,
    *,
    spotify_id: str | None = None,
    youtube_id: str | None = None,
    mbid: str | None = None,
) -> Track | None:
    """Look up a track by any known external ID."""
    for column, value in (
        (Track.spotify_id, spotify_id),
        (Track.youtube_id, youtube_id),
        (Track.mbid, mbid),
    ):
        if value:
            hit = session.exec(select(Track).where(column == value)).first()
            if hit is not None:
                return hit
    return None


def upsert_track(session: Session, ref: TrackRef) -> tuple[Track, bool]:
    """Insert a track, or return the existing one with missing IDs backfilled.

    Returns ``(track, created)``.
    """
    existing = find_existing(
        session,
        spotify_id=ref.spotify_id,
        youtube_id=ref.youtube_id,
        mbid=ref.mbid,
    )
    if existing is not None:
        changed = False
        for attr in ("spotify_id", "youtube_id", "mbid", "album"):
            incoming = getattr(ref, attr)
            if incoming and not getattr(existing, attr):
                setattr(existing, attr, incoming)
                changed = True
        if changed:
            session.add(existing)
        return existing, False

    track = Track(
        title=ref.title,
        artist=ref.artist,
        album=ref.album,
        duration_ms=ref.duration_ms or 0,
        spotify_id=ref.spotify_id,
        youtube_id=ref.youtube_id,
        mbid=ref.mbid,
        status=TrackStatus.queued,
    )
    session.add(track)
    session.flush()
    return track, True
