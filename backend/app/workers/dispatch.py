"""Enqueue Celery tasks from the API by name (keeps heavy ML imports out of the API process)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from kombu.exceptions import OperationalError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import AppError
from app.models import AutopilotRun, Dataset, ForecastRun, ModelRun
from app.workers.celery_app import celery_app

QUEUE_INGEST = "ingestion"
QUEUE_TRAIN = "training"
QUEUE_FORECAST = "forecasting"

TASK_INGEST = "storepulse.ingest_dataset"
TASK_HEALTH = "storepulse.compute_dataset_health"
TASK_TRAIN = "storepulse.train_model"
TASK_TUNE = "storepulse.tune_model"
TASK_FORECAST = "storepulse.run_forecast"
TASK_AUTOPILOT = "storepulse.run_autopilot"


def enqueue(task_name: str, obj_id: uuid.UUID, queue: str, **kwargs: Any) -> str:
    result = celery_app.send_task(task_name, args=[str(obj_id)], kwargs=kwargs, queue=queue)
    return str(result.id)


async def enqueue_job(
    db: AsyncSession,
    job: AutopilotRun | Dataset | ForecastRun | ModelRun,
    task_name: str,
    queue: str,
    **kwargs: Any,
) -> str:
    """Publish a committed job and record broker failures so the request can be retried."""
    try:
        task_id = await run_in_threadpool(enqueue, task_name, job.id, queue, **kwargs)
    except (OperationalError, OSError) as exc:
        # A worker may already have received the message if the broker lost its acknowledgement.
        await db.execute(
            update(type(job))
            .where(type(job).id == job.id, type(job).status == "queued")
            .values(
                status="failed",
                stage="dispatch_failed",
                error="Could not confirm job delivery. Submit it again when the job queue is available.",
                finished_at=datetime.now(UTC),
            )
        )
        await db.commit()
        raise AppError(
            "Could not confirm job delivery. Try again shortly.",
            code="job_queue_unavailable",
            status_code=503,
            details={"job_id": str(job.id)},
        ) from exc

    job.celery_task_id = task_id
    await db.commit()
    return task_id
