"""Data schema — the single source of truth (invariant 4).

These sqlmodel classes *are* pydantic models; the SQLite tables and the Parquet
schemas both derive from them, never the reverse. Mirrors the data model in
CLAUDE.md.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, CheckConstraint, UniqueConstraint
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TrackStatus(StrEnum):
    """Lifecycle of a track through the acquisition + extraction pipeline."""

    queued = "queued"
    downloading = "downloading"
    downloaded = "downloaded"
    extracted = "extracted"
    failed = "failed"


class SectionKind(StrEnum):
    """Structural segment label from Essentia segmentation."""

    intro = "intro"
    verse = "verse"
    chorus = "chorus"
    bridge = "bridge"
    outro = "outro"
    other = "other"


class StemKind(StrEnum):
    """Demucs 4-stem split."""

    vocals = "vocals"
    drums = "drums"
    bass = "bass"
    other = "other"


class Track(SQLModel, table=True):
    """One resolved track. External IDs are nullable and individually unique."""

    __tablename__ = "tracks"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    spotify_id: str | None = Field(default=None, unique=True, index=True)
    youtube_id: str | None = Field(default=None, unique=True, index=True)
    mbid: str | None = Field(default=None, unique=True, index=True)
    title: str
    artist: str
    album: str | None = None
    duration_ms: int = 0
    audio_path: str | None = None  # relative to data/audio/
    status: TrackStatus = Field(default=TrackStatus.queued, index=True)
    added_at: datetime = Field(default_factory=_utcnow)
    extracted_at: datetime | None = None


class Section(SQLModel, table=True):
    """A structural segment of a track (intro/verse/chorus/...)."""

    __tablename__ = "sections"

    id: int | None = Field(default=None, primary_key=True)
    track_id: str = Field(foreign_key="tracks.id", index=True)
    index: int
    kind: SectionKind = Field(default=SectionKind.other)
    start_s: float
    end_s: float


class FeaturesInterpretable(SQLModel, table=True):
    """Denormalized interpretable features — one row per track.

    Phase 0 skeleton. CLAUDE.md targets ~50 columns; Essentia/Librosa fields
    are added in Phase 1 as extraction lands. Every row stores ``config_hash``
    (invariant 2).
    """

    __tablename__ = "features_interpretable"

    track_id: str = Field(foreign_key="tracks.id", primary_key=True)
    config_hash: str
    key: str | None = None
    mode: str | None = None
    bpm: float | None = None
    danceability: float | None = None
    valence: float | None = None
    arousal: float | None = None
    energy: float | None = None
    loudness_db: float | None = None
    dyn_range_db: float | None = None
    spectral_centroid: float | None = None
    zero_crossing_rate: float | None = None
    instrumentalness: float | None = None
    acousticness: float | None = None
    extracted_at: datetime = Field(default_factory=_utcnow)


class Stem(SQLModel, table=True):
    """A Demucs-separated stem file (on-demand, Phase 4)."""

    __tablename__ = "stems"

    id: int | None = Field(default=None, primary_key=True)
    track_id: str = Field(foreign_key="tracks.id", index=True)
    kind: StemKind
    path: str
    analyzed: bool = False


class Rating(SQLModel, table=True):
    """The listening-session rubric (Phase 3)."""

    __tablename__ = "ratings"

    id: int | None = Field(default=None, primary_key=True)
    track_id: str = Field(foreign_key="tracks.id", index=True)
    rated_at: datetime = Field(default_factory=_utcnow)
    vibe: int = Field(ge=1, le=5)
    replay: int = Field(ge=1, le=5)
    skip: int = Field(ge=1, le=5)
    notes: str | None = None


class EssenceSibling(SQLModel, table=True):
    """A symmetric "feels like a sibling of" relation, stored once.

    Always enforce ``track_a < track_b`` to avoid duplicate pairs — use
    :meth:`create` rather than the constructor directly.
    """

    __tablename__ = "essence_siblings"
    __table_args__ = (
        CheckConstraint("track_a < track_b", name="ck_sibling_order"),
        UniqueConstraint("track_a", "track_b", name="uq_sibling_pair"),
    )

    id: int | None = Field(default=None, primary_key=True)
    track_a: str = Field(foreign_key="tracks.id", index=True)
    track_b: str = Field(foreign_key="tracks.id", index=True)
    strength: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)
    note: str | None = None

    @classmethod
    def create(
        cls, track_a: str, track_b: str, strength: float, note: str | None = None
    ) -> "EssenceSibling":
        """Build a sibling pair with the (track_a < track_b) invariant enforced."""
        if track_a == track_b:
            raise ValueError("an essence-sibling pair needs two distinct tracks")
        low, high = sorted((track_a, track_b))
        return cls(track_a=low, track_b=high, strength=strength, note=note)


class TasteModelRun(SQLModel, table=True):
    """A training run of a preference model (M1/M2/M3)."""

    __tablename__ = "taste_model_runs"

    id: int | None = Field(default=None, primary_key=True)
    version: str
    started_at: datetime = Field(default_factory=_utcnow)
    config_hash: str
    metrics_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    checkpoint_path: str | None = None
