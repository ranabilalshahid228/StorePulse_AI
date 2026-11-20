from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery("storepulse", broker=settings.broker_url, backend=settings.result_backend)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # re-deliver if a worker dies mid-task
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # long-running ML tasks: don't hoard
    task_default_queue="ingestion",
    task_routes={
        "storepulse.ingest_dataset": {"queue": "ingestion"},
        "storepulse.compute_dataset_health": {"queue": "ingestion"},
        "storepulse.train_model": {"queue": "training"},
        "storepulse.tune_model": {"queue": "training"},
        "storepulse.run_forecast": {"queue": "forecasting"},
        "storepulse.run_autopilot": {"queue": "forecasting"},
    },
    task_ignore_result=settings.result_backend is None,
    task_always_eager=settings.celery_task_always_eager,
    result_expires=60 * 60 * 24,
    broker_connection_retry_on_startup=True,
)
celery_app.autodiscover_tasks(["app.workers"], related_name="tasks")
