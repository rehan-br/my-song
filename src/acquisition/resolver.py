"""Cross-source track resolution: dedupe, duration sanity check, upsert.

The duration check guards against yt-dlp picking the wrong track (live cuts,
covers, fan edits) — see the gotcha in CLAUDE.md. Provenance is recorded here;
the taste weight derived from it is computed separately in
``taste_model.engagement``.
"""

from sqlmodel import Session, select

from acquisition.base import AudioCandidate, Provenance, TrackRef
from storage.schema import SourceType, Track, TrackSource, TrackStatus


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


def _record_source(session: Session, track: Track, source: Provenance) -> None:
    """Persist a track's provenance, idempotently (one row per distinct source)."""
    source_type = SourceType(source.source_type)
    already = session.exec(
        select(TrackSource).where(
            TrackSource.track_id == track.id,
            TrackSource.source_type == source_type,
            TrackSource.source_ref == source.source_ref,
        )
    ).first()
    if already is None:
        session.add(
            TrackSource(
                track_id=track.id,
                source_type=source_type,
                source_ref=source.source_ref,
                source_name=source.source_name,
            )
        )
        session.flush()


def upsert_track(
    session: Session, ref: TrackRef, source: Provenance | None = None
) -> tuple[Track, bool]:
    """Insert a track, or return the existing one with missing IDs backfilled.

    If ``source`` is given, its provenance is recorded. The taste weight is not
    touched here — ``taste_model.engagement.apply_engagement`` owns that.

    Returns ``(track, created)``.
    """
    existing = find_existing(
        session,
        spotify_id=ref.spotify_id,
        youtube_id=ref.youtube_id,
        mbid=ref.mbid,
    )
    if existing is not None:
        for attr in ("spotify_id", "youtube_id", "mbid", "album"):
            incoming = getattr(ref, attr)
            if incoming and not getattr(existing, attr):
                setattr(existing, attr, incoming)
                session.add(existing)
        track, created = existing, False
    else:
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
        created = True

    if source is not None:
        _record_source(session, track, source)
    return track, created
