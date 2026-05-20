"""SQLite engine + session management (sqlmodel)."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from omegaconf import DictConfig
from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from core import paths
from storage import schema  # noqa: F401 — import registers table metadata
from storage.schema import DEFAULT_USER_ID

_engines: dict[str, Engine] = {}

# Tables that gained a `user_id` column when the per-user seam landed. On an
# existing DB they need ALTER + a backfill; new DBs get the column from create_all.
_USER_SCOPED_TABLES = (
    "track_sources",
    "ratings",
    "listening_events",
    "essence_siblings",
    "taste_model_runs",
)


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
    """Create any missing tables and migrate any legacy state. Idempotent."""
    engine = get_engine(cfg)
    SQLModel.metadata.create_all(engine)
    _migrate_user_scope(engine)
    _migrate_legacy_checkpoints(cfg)


def _migrate_user_scope(engine: Engine) -> None:
    """Add the per-user seam to an older DB — ALTER each user-scoped table to
    carry a ``user_id`` defaulted to ``DEFAULT_USER_ID``, and seed that user.

    Idempotent: existing rows get the default user; new DBs already have the
    column from ``create_all`` and just need the default user row.
    """
    from storage.users import ensure_default_user  # local import — avoids cycle

    with Session(engine) as session:
        ensure_default_user(session)
        session.commit()

    with engine.begin() as conn:
        for table in _USER_SCOPED_TABLES:
            cols = [
                row[1]
                for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            ]
            if "user_id" not in cols:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL "
                    f"DEFAULT '{DEFAULT_USER_ID}'"
                )


def _migrate_legacy_checkpoints(cfg: DictConfig) -> None:
    """One-time: move ``data/models/m2.npz`` → ``data/models/default/m2.npz``.

    Trained-model files used to be flat; per-user checkpoints live under a user
    subdirectory. Move only if the new path is empty so a fresh train isn't
    clobbered.
    """
    models_dir = paths.resolve(cfg.paths.data) / "models"
    default_dir = models_dir / DEFAULT_USER_ID
    for filename in ("m2.npz", "m3.pt"):
        legacy = models_dir / filename
        new = default_dir / filename
        if legacy.exists() and not new.exists():
            default_dir.mkdir(parents=True, exist_ok=True)
            legacy.rename(new)


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
