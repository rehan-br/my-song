"""SQLite engine + session management (sqlmodel)."""

from collections.abc import Iterator
from contextlib import contextmanager

from omegaconf import DictConfig
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from core import paths
from storage import schema  # noqa: F401 — import registers table metadata

_engines: dict[str, Engine] = {}


def get_engine(cfg: DictConfig) -> Engine:
    """Return a cached SQLite engine for the configured database path."""
    db_path = paths.resolve(cfg.paths.db)
    key = str(db_path)
    if key not in _engines:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engines[key] = create_engine(f"sqlite:///{db_path}")
    return _engines[key]


def init_db(cfg: DictConfig) -> None:
    """Create any missing tables. Idempotent."""
    SQLModel.metadata.create_all(get_engine(cfg))


@contextmanager
def session_scope(cfg: DictConfig) -> Iterator[Session]:
    """Session context manager: commits on success, rolls back on error."""
    session = Session(get_engine(cfg))
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
