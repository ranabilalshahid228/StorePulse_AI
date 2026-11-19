"""Server-Sent Events stream of job progress read from PostgreSQL."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.db.session import get_async_sessionmaker

TERMINAL = {"succeeded", "failed", "cancelled", "ready"}
POLL_SECONDS = 1.0
MAX_SECONDS = 60 * 60


def progress_stream(model: type[Any], obj_id: uuid.UUID, org_id: uuid.UUID) -> EventSourceResponse:
    cols = [model.status, model.progress_pct, model.stage, model.status_message, model.error]

    async def gen() -> AsyncIterator[dict[str, str]]:
        last: tuple[Any, ...] | None = None
        waited = 0.0
        while waited < MAX_SECONDS:
            async with get_async_sessionmaker()() as s:
                row = (
                    await s.execute(select(*cols).where(model.id == obj_id, model.organization_id == org_id))
                ).first()
            if row is None:
                yield {"event": "error", "data": json.dumps({"code": "not_found"})}
                return
            current = tuple(row)
            if current != last:
                status, pct, stage, msg, err = current
                yield {
                    "event": "progress",
                    "data": json.dumps(
                        {"status": status, "progress_pct": pct, "stage": stage, "message": msg, "error": err}
                    ),
                }
                last = current
                if status in TERMINAL:
                    return
            await asyncio.sleep(POLL_SECONDS)
            waited += POLL_SECONDS

    return EventSourceResponse(gen())
