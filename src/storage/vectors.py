"""Parquet feature store + FAISS index management.

Vectors do not belong in SQLite (CLAUDE.md data model): embeddings live in
Parquet shards under ``data/features/embeddings/``, FAISS indexes under
``data/indexes/``. Each row carries its ``config_hash`` (invariant 2) so a
re-extraction under a changed config is detectable, never a silent overwrite.

FAISS index management lands in Phase 2; for now this module owns the Parquet
song-embedding store.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from core import paths


def embeddings_dir(cfg: DictConfig) -> Path:
    """Directory holding Parquet embedding shards."""
    return paths.resolve(cfg.paths.features) / "embeddings"


def index_dir(cfg: DictConfig) -> Path:
    """Directory holding FAISS index files."""
    return paths.resolve(cfg.paths.indexes)


def song_embedding_path(cfg: DictConfig, name: str) -> Path:
    """Path of a per-song embedding shard, e.g. ``name='mert_song'``."""
    return embeddings_dir(cfg) / f"{name}.parquet"


@dataclass(slots=True)
class EmbeddingRow:
    """One track's song-level embedding plus the provenance of how it was made."""

    track_id: str
    embedding: np.ndarray
    model: str
    config_hash: str
    extracted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


_SCHEMA = pa.schema(
    [
        ("track_id", pa.string()),
        ("model", pa.string()),
        ("config_hash", pa.string()),
        ("dim", pa.int32()),
        ("embedding", pa.list_(pa.float32())),
        ("extracted_at", pa.timestamp("us")),
    ]
)


def read_embeddings(path: Path) -> dict[str, EmbeddingRow]:
    """Load an embedding shard into ``{track_id: EmbeddingRow}`` (empty if absent)."""
    if not path.exists():
        return {}
    table = pq.read_table(path)
    rows: dict[str, EmbeddingRow] = {}
    for record in table.to_pylist():
        rows[record["track_id"]] = EmbeddingRow(
            track_id=record["track_id"],
            embedding=np.asarray(record["embedding"], dtype=np.float32),
            model=record["model"],
            config_hash=record["config_hash"],
            extracted_at=record["extracted_at"],
        )
    return rows


def write_embeddings(path: Path, rows: Iterable[EmbeddingRow]) -> int:
    """Write the full set of embedding rows to ``path`` (overwriting the shard).

    Callers that want to add to an existing shard should ``read_embeddings``
    first, merge, then pass the merged set here. Returns the row count written.
    """
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "track_id": [r.track_id for r in rows],
            "model": [r.model for r in rows],
            "config_hash": [r.config_hash for r in rows],
            "dim": [int(len(r.embedding)) for r in rows],
            "embedding": [np.asarray(r.embedding, dtype=np.float32) for r in rows],
            "extracted_at": [r.extracted_at for r in rows],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, path)
    return len(rows)
