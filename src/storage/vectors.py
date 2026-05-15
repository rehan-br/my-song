"""Parquet feature store + FAISS index management.

Vectors do not belong in SQLite (CLAUDE.md data model): embeddings live in
Parquet shards under ``data/features/``, FAISS indexes under ``data/indexes/``.

Phase 1 implements the read/write logic. For now this module exposes only
path helpers, so the rest of the codebase can reference stable locations
without importing pyarrow/faiss (kept out of the Phase 0 dependency set).
"""

from pathlib import Path

from omegaconf import DictConfig

from core import paths


def embeddings_dir(cfg: DictConfig) -> Path:
    """Directory holding Parquet embedding shards."""
    return paths.resolve(cfg.paths.features) / "embeddings"


def index_dir(cfg: DictConfig) -> Path:
    """Directory holding FAISS index files."""
    return paths.resolve(cfg.paths.indexes)
