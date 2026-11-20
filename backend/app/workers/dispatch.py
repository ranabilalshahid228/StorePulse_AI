"""Enqueue Celery tasks from the API by name (keeps heavy ML imports out of the API process)."""

from __future__ import annotations

import uuid
from typing import Any

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
