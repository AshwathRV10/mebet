"""Engine/session management and schema creation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from ..logging_setup import get_logger
from .models import Base

log = get_logger("db.session")

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def _configure_sqlite(dbapi_con, _record) -> None:
    cur = dbapi_con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def get_engine() -> Engine:
    global _engine, _Session
    if _engine is None:
        url = get_settings().resolved_database_url()
        _engine = create_engine(url, future=True, pool_pre_ping=True)
        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _configure_sqlite)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        log.debug("database engine ready: %s", url.split("://", 1)[0])
    return _engine


def init_db() -> None:
    """Create any missing tables. Safe to call repeatedly."""
    Base.metadata.create_all(get_engine())
    log.info("schema ensured (%d tables)", len(Base.metadata.tables))


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _Session is not None
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def new_session() -> Session:
    get_engine()
    assert _Session is not None
    return _Session()


def reset_engine() -> None:
    """Drop cached engine - used by tests that switch database URLs."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None
