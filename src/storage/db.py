"""SQLite engine + session management (sqlmodel)."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from omegaconf import DictConfig
from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from core import paths
from storage import schema  # noqa: F401 — import registers table metadata

_engines: dict[str, Engine] = {}


def get_engine(cfg: DictConfig) -> Engine:
    """Return a cached SQLite engine for the configured database path.

    WAL mode + a 30s busy timeout let a reader (e.g. extraction) coexist with a
    long-running writer (e.g. the crawler/downloader) without lock errors.
    """
    db_path = paths.resolve(cfg.paths.db)
    key = str(db_path)
    if key not in _engines:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30.0})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        _engines[key] = engine
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
