from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_async_sessionmaker

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready() -> JSONResponse:
    checks: dict[str, Any] = {}
    try:
        async with get_async_sessionmaker()() as s:
            await s.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {type(exc).__name__}"
    broker = get_settings().broker_url
    try:
        if broker.startswith("redis"):
            import redis.asyncio as aioredis

            r = aioredis.from_url(broker)
            await r.ping()
            await r.aclose()
        else:  # e.g. the Postgres (SQLAlchemy) broker used for lean local runs
            from kombu import Connection
            from starlette.concurrency import run_in_threadpool

            def _check() -> None:
                with Connection(broker) as conn:
                    conn.ensure_connection(max_retries=1)

            await run_in_threadpool(_check)
        checks["broker"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["broker"] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503
    )
