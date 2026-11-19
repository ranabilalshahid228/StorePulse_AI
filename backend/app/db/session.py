"""Database engines.

* The API uses an async engine (asyncpg).
* Celery workers use a sync engine (psycopg 3) - training code is CPU/GPU bound and synchronous.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def get_async_engine() -> AsyncEngine:
    s = get_settings()
    return create_async_engine(
        s.database_url_async,
        pool_size=s.db_pool_size,
        max_overflow=s.db_max_overflow,
        pool_pre_ping=True,
    )


@lru_cache
def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_async_engine(), expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, committed by the endpoint/service."""
    async with get_async_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@lru_cache
def get_sync_engine() -> Engine:
    s = get_settings()
    return create_engine(s.database_url_sync, pool_pre_ping=True, pool_size=5, max_overflow=5)


@lru_cache
def _sync_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(get_sync_engine(), expire_on_commit=False)


@contextmanager
def sync_session() -> Iterator[Session]:
    session = _sync_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
