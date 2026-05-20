"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from storage import schema  # noqa: F401 — registers table metadata


@pytest.fixture
def session() -> Iterator[Session]:
    """An isolated in-memory SQLite session with all tables + the default user."""
    from storage.users import ensure_default_user

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        ensure_default_user(s)
        s.commit()
        yield s
