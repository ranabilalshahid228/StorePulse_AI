"""Persist job progress to PostgreSQL from inside Celery tasks.

Writes are throttled (at most every 0.5 s unless the stage changes or the job finishes)
so tight training loops don't hammer the database.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update

from app.db.session import sync_session
from app.models.enums import JobStatus


class ProgressReporter:
    MIN_INTERVAL_S = 0.5

    def __init__(self, model: type[Any], obj_id: uuid.UUID, span: tuple[float, float] = (0.0, 100.0)) -> None:
        self.model = model
        self.obj_id = obj_id
        self.lo, self.hi = span
        self._last_write = 0.0
        self._last_stage: str | None = None

    def _write(self, **values: Any) -> None:
        with sync_session() as s:
            s.execute(update(self.model).where(self.model.id == self.obj_id).values(**values))
        self._last_write = time.monotonic()

    def start(self, stage: str = "starting", running_status: str = JobStatus.RUNNING.value) -> None:
        self._write(
            status=running_status, stage=stage, progress_pct=self.lo, started_at=datetime.now(UTC), error=None
        )
        self._last_stage = stage

    def update(self, fraction: float, stage: str, message: str | None = None, force: bool = False) -> None:
        """`fraction` in [0, 1] is mapped into this reporter's percentage span."""
        pct = round(self.lo + (self.hi - self.lo) * max(0.0, min(1.0, fraction)), 2)
        now = time.monotonic()
        if not force and stage == self._last_stage and now - self._last_write < self.MIN_INTERVAL_S:
            return
        values: dict[str, Any] = {"progress_pct": pct, "stage": stage}
        if message is not None:
            values["status_message"] = message
        self._write(**values)
        self._last_stage = stage

    def sub(self, lo_frac: float, hi_frac: float) -> ProgressReporter:
        """Reporter for a sub-stage covering [lo_frac, hi_frac] of this reporter's span."""
        width = self.hi - self.lo
        child = ProgressReporter(
            self.model, self.obj_id, (self.lo + width * lo_frac, self.lo + width * hi_frac)
        )
        child._last_stage = self._last_stage
        return child

    def succeed(self, status: str = JobStatus.SUCCEEDED.value, **values: Any) -> None:
        self._write(status=status, progress_pct=100.0, stage="done", finished_at=datetime.now(UTC), **values)

    def fail(self, error: str, status: str = JobStatus.FAILED.value) -> None:
        self._write(status=status, stage="failed", error=error[:4000], finished_at=datetime.now(UTC))
