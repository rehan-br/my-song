"""User-scoping helpers — the seam for per-user data.

Today there is exactly one user (``DEFAULT_USER_ID``); ``current_user_id``
becomes auth/config-driven when productionization adds real users. Putting the
seam in now keeps multi-user a configuration change rather than a migration —
every user-scoped table already carries a ``user_id`` FK that defaults here.
"""

from sqlmodel import Session

from storage.schema import DEFAULT_USER_ID, User


def ensure_default_user(session: Session) -> User:
    """Idempotently create the default user — every fresh DB has one."""
    user = session.get(User, DEFAULT_USER_ID)
    if user is not None:
        return user
    user = User(id=DEFAULT_USER_ID, name="default")
    session.add(user)
    return user


def current_user_id() -> str:
    """Whose data are we operating on. Single-user today; auth-driven later."""
    return DEFAULT_USER_ID
