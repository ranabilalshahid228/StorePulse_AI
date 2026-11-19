from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse

from app.api.deps import AnalystPrincipal, CurrentPrincipal, DbSession
from app.api.v1._progress import progress_stream
from app.core.config import get_settings
from app.core.exceptions import DomainValidationError, NotFoundError, PreconditionFailedError
from app.ml.families import FAMILIES
from app.models import Dataset, ModelRun, Store
from app.models.enums import DatasetStatus, JobStatus, RunType
from app.schemas.common import JobAccepted, Page, PageParams
from app.schemas.runs import (
    Leaderboard,
    LeaderboardEntry,
    ModelFamilyInfo,
    ModelRunDetail,
    ModelRunRead,
    TrainRequest,
    TuneRequest,
)
from app.services.storage import get_storage
from app.services.tenancy import get_owned, paginate, scoped
from app.workers import dispatch

router = APIRouter(tags=["model-lab"])
LOWER_IS_BETTER = {"wape", "rmse", "mae", "rmspe", "smape", "mape", "rmsse"}


@router.get("/model-families", response_model=list[ModelFamilyInfo])
async def list_families(_: CurrentPrincipal) -> list[ModelFamilyInfo]:
    return [
        ModelFamilyInfo(
            family=fam,
            description=cls.description,
            default_hyperparameters=cls.default_params,
            search_space=cls.search_space,
            supports_gpu=cls.supports_gpu,
            available=cls.is_available(),
        )
        for fam, cls in FAMILIES.items()
    ]


async def _validate(body: TrainRequest, principal: AnalystPrincipal, db: DbSession) -> tuple[Dataset, int]:
    ds = await get_owned(db, Dataset, body.dataset_id, principal.organization_id)
    if ds.role != "primary":
        raise PreconditionFailedError("Auxiliary datasets cannot be trained on; train on the primary dataset")
    if ds.status != DatasetStatus.READY:
        raise PreconditionFailedError("Dataset is not ready for training", details={"status": ds.status})
    if body.store_id:
        await get_owned(db, Store, body.store_id, principal.organization_id)
    if not FAMILIES[body.model_family].is_available():
        raise PreconditionFailedError(f"Model family '{body.model_family}' is not installed on the workers")
    unknown = set(body.hyperparameters) - set(FAMILIES[body.model_family].default_params)
    if unknown:
        raise DomainValidationError("Unknown hyperparameters", details={"unknown": sorted(unknown)})
    validation_days = body.validation_days or body.horizon_days
    return ds, validation_days


def _new_run(
    body: TrainRequest, principal: AnalystPrincipal, run_type: RunType, validation_days: int
) -> ModelRun:
    return ModelRun(
        organization_id=principal.organization_id,
        dataset_id=body.dataset_id,
        store_id=body.store_id,
        created_by_id=principal.user_id,
        name=body.name,
        model_family=body.model_family.value,
        run_type=run_type.value,
        horizon_days=body.horizon_days,
        validation_days=validation_days,
        hyperparameters=body.hyperparameters,
        training_config=body.training_config(),
        status=JobStatus.QUEUED.value,
        progress_pct=0.0,
        stage="queued",
        trials=[],
        epoch_logs=[],
        metrics={},
        feature_columns=[],
        plot_paths=[],
    )


@router.post("/model-runs/train", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_training(body: TrainRequest, principal: AnalystPrincipal, db: DbSession) -> JobAccepted:
    _, validation_days = await _validate(body, principal, db)
    run = _new_run(body, principal, RunType.TRAIN, validation_days)
    db.add(run)
    await db.commit()
    run.celery_task_id = dispatch.enqueue(dispatch.TASK_TRAIN, run.id, dispatch.QUEUE_TRAIN)
    await db.commit()
    return JobAccepted(id=run.id, status=run.status, celery_task_id=run.celery_task_id)


@router.post("/model-runs/tune", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_tuning(body: TuneRequest, principal: AnalystPrincipal, db: DbSession) -> JobAccepted:
    _, validation_days = await _validate(body, principal, db)
    space = body.search_space or FAMILIES[body.model_family].search_space
    if not space:
        raise DomainValidationError(f"Model family '{body.model_family}' has no tunable search space")
    run = _new_run(body, principal, RunType.TUNE, validation_days)
    run.search_space = space
    run.n_trials = body.n_trials or get_settings().optuna_default_trials
    db.add(run)
    await db.commit()
    run.celery_task_id = dispatch.enqueue(
        dispatch.TASK_TUNE,
        run.id,
        dispatch.QUEUE_TRAIN,
        metric=body.metric,
        timeout_s=body.timeout_minutes * 60 if body.timeout_minutes else None,
    )
    await db.commit()
    return JobAccepted(id=run.id, status=run.status, celery_task_id=run.celery_task_id)


@router.get("/model-runs", response_model=Page[ModelRunRead])
async def list_runs(
    principal: CurrentPrincipal,
    db: DbSession,
    page: PageParams = Depends(),
    dataset_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status"),
) -> Page[ModelRunRead]:
    stmt = scoped(ModelRun, principal.organization_id)
    if dataset_id:
        stmt = stmt.where(ModelRun.dataset_id == dataset_id)
    if status_filter:
        stmt = stmt.where(ModelRun.status == status_filter)
    items, total = await paginate(db, stmt.order_by(ModelRun.created_at.desc()), page.limit, page.offset)
    return Page(
        items=[ModelRunRead.model_validate(i) for i in items],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/model-runs/leaderboard", response_model=Leaderboard)
async def leaderboard(
    principal: CurrentPrincipal,
    db: DbSession,
    dataset_id: uuid.UUID,
    metric: str = Query("wape", pattern="^(wape|rmse|mae|rmspe|smape|mape|rmsse|r2)$"),
    limit: int = Query(20, ge=1, le=100),
) -> Leaderboard:
    await get_owned(db, Dataset, dataset_id, principal.organization_id)
    stmt = scoped(ModelRun, principal.organization_id).where(
        ModelRun.dataset_id == dataset_id, ModelRun.status == JobStatus.SUCCEEDED.value
    )
    runs = [r for r in (await db.execute(stmt)).scalars() if r.metrics.get(metric) is not None]
    runs.sort(key=lambda r: r.metrics[metric], reverse=metric not in LOWER_IS_BETTER)
    entries = [
        LeaderboardEntry(
            rank=i + 1,
            model_run_id=r.id,
            name=r.name,
            model_family=r.model_family,
            run_type=r.run_type,
            metric_value=float(r.metrics[metric]),
            metrics={k: v for k, v in r.metrics.items() if k in LOWER_IS_BETTER | {"bias_pct", "n", "r2"}},
            inference_ms=r.inference_ms,
            finished_at=r.finished_at,
        )
        for i, r in enumerate(runs[:limit])
    ]
    return Leaderboard(dataset_id=dataset_id, metric=metric, entries=entries)


@router.get("/model-runs/{run_id}", response_model=ModelRunDetail)
async def get_run(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> ModelRun:
    return await get_owned(db, ModelRun, run_id, principal.organization_id)


@router.get("/model-runs/{run_id}/events")
async def run_events(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession):  # noqa: ANN201
    await get_owned(db, ModelRun, run_id, principal.organization_id)
    return progress_stream(ModelRun, run_id, principal.organization_id)


@router.get("/model-runs/{run_id}/plots/{name}")
async def get_plot(
    run_id: uuid.UUID, name: str, principal: CurrentPrincipal, db: DbSession
) -> StreamingResponse:
    run = await get_owned(db, ModelRun, run_id, principal.organization_id)
    key = next((p for p in run.plot_paths if p.endswith(f"/{name}.png")), None)
    if key is None:
        raise NotFoundError(f"Plot '{name}' not found for this run")
    return StreamingResponse(get_storage().iter_bytes(key), media_type="image/png")


@router.post("/model-runs/{run_id}/cancel", response_model=JobAccepted)
async def cancel_run(run_id: uuid.UUID, principal: AnalystPrincipal, db: DbSession) -> JobAccepted:
    run = await get_owned(db, ModelRun, run_id, principal.organization_id)
    if run.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise PreconditionFailedError("Only queued or running jobs can be cancelled")
    if run.celery_task_id:
        from app.workers.celery_app import celery_app

        celery_app.control.revoke(run.celery_task_id, terminate=True)
    run.status = JobStatus.CANCELLED.value
    run.stage = "cancelled"
    await db.commit()
    return JobAccepted(id=run.id, status=run.status, celery_task_id=run.celery_task_id)
