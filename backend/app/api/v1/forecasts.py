from __future__ import annotations

import uuid

import pandas as pd
import pyarrow.dataset as pads
from fastapi import APIRouter, Depends, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import AnalystPrincipal, CurrentPrincipal, DbSession
from app.api.v1._progress import progress_stream
from app.core.exceptions import DomainValidationError, PreconditionFailedError
from app.models import Dataset, ForecastRun, ModelRun, Store
from app.models.enums import ForecastKind, JobStatus
from app.schemas.common import JobAccepted, Page, PageParams
from app.schemas.runs import (
    EventForecastRequest,
    ForecastPoint,
    ForecastRequest,
    ForecastRunRead,
    ForecastSeries,
)
from app.services.storage import get_storage
from app.services.tenancy import get_owned, paginate, scoped
from app.workers import dispatch

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


async def _submit(
    body: ForecastRequest, kind: ForecastKind, principal: AnalystPrincipal, db: DbSession, scenario: dict
) -> JobAccepted:
    run = await get_owned(db, ModelRun, body.model_run_id, principal.organization_id)
    if run.status != JobStatus.SUCCEEDED or not run.artifact_path:
        raise PreconditionFailedError(
            "Model run has not finished successfully", details={"status": run.status}
        )
    if body.horizon_days > run.horizon_days:
        raise DomainValidationError(
            f"horizon_days ({body.horizon_days}) exceeds the model's trained horizon ({run.horizon_days})"
        )
    if run.store_id and body.store_id and body.store_id != run.store_id:
        raise DomainValidationError(
            "This model was trained on a single store and can only forecast that store"
        )
    store_id = body.store_id or run.store_id
    if store_id:
        await get_owned(db, Store, store_id, principal.organization_id)
    allowed = set(
        (await get_owned(db, Dataset, run.dataset_id, principal.organization_id)).schema_mapping.get(
            "known_future_columns", []
        )
    )
    bad = set(body.covariate_defaults) - allowed
    for ev in scenario.get("events", []):
        bad |= set(ev["overrides"]) - allowed
    if bad:
        raise DomainValidationError(
            "Overrides reference unknown covariates",
            details={"unknown": sorted(bad), "allowed": sorted(allowed)},
        )
    fr = ForecastRun(
        organization_id=principal.organization_id,
        model_run_id=run.id,
        dataset_id=run.dataset_id,
        store_id=store_id,
        created_by_id=principal.user_id,
        kind=kind.value,
        horizon_days=body.horizon_days,
        scenario={"covariate_defaults": body.covariate_defaults, **scenario},
        status=JobStatus.QUEUED.value,
        progress_pct=0.0,
        stage="queued",
        summary={},
    )
    db.add(fr)
    await db.commit()
    fr.celery_task_id = dispatch.enqueue(dispatch.TASK_FORECAST, fr.id, dispatch.QUEUE_FORECAST)
    await db.commit()
    return JobAccepted(id=fr.id, status=fr.status, celery_task_id=fr.celery_task_id)


@router.post("", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_forecast(body: ForecastRequest, principal: AnalystPrincipal, db: DbSession) -> JobAccepted:
    """Multi-horizon batch forecast with a trained model."""
    return await _submit(body, ForecastKind.MULTI_HORIZON, principal, db, {})


@router.post("/event", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_event_forecast(
    body: EventForecastRequest, principal: AnalystPrincipal, db: DbSession
) -> JobAccepted:
    """Scenario forecast: override known-future covariates (promos, closures, holidays) on date ranges
    and optionally compare to the no-event baseline."""
    scenario = {
        "events": [e.model_dump(mode="json") for e in body.events],
        "compare_to_baseline": body.compare_to_baseline,
    }
    return await _submit(body, ForecastKind.EVENT, principal, db, scenario)


@router.get("", response_model=Page[ForecastRunRead])
async def list_forecasts(
    principal: CurrentPrincipal,
    db: DbSession,
    page: PageParams = Depends(),
    model_run_id: uuid.UUID | None = None,
    store_id: uuid.UUID | None = None,
) -> Page[ForecastRunRead]:
    stmt = scoped(ForecastRun, principal.organization_id)
    if model_run_id:
        stmt = stmt.where(ForecastRun.model_run_id == model_run_id)
    if store_id:
        stmt = stmt.where(ForecastRun.store_id == store_id)
    items, total = await paginate(db, stmt.order_by(ForecastRun.created_at.desc()), page.limit, page.offset)
    return Page(
        items=[ForecastRunRead.model_validate(i) for i in items],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{forecast_id}", response_model=ForecastRunRead)
async def get_forecast(forecast_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> ForecastRun:
    return await get_owned(db, ForecastRun, forecast_id, principal.organization_id)


@router.get("/{forecast_id}/events")
async def forecast_events(forecast_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession):  # noqa: ANN201
    await get_owned(db, ForecastRun, forecast_id, principal.organization_id)
    return progress_stream(ForecastRun, forecast_id, principal.organization_id)


def read_forecast_frame(fr: ForecastRun, store_code: str | None = None) -> pd.DataFrame:
    if fr.status != JobStatus.SUCCEEDED or not fr.output_path:
        raise PreconditionFailedError("Forecast has not finished", details={"status": fr.status})
    path = get_storage().local_path(fr.output_path)
    filt = pads.field("store_code") == store_code if store_code else None
    return pads.dataset(path, format="parquet").to_table(filter=filt).to_pandas()


@router.get("/{forecast_id}/series", response_model=ForecastSeries)
async def forecast_series(
    forecast_id: uuid.UUID,
    principal: CurrentPrincipal,
    db: DbSession,
    store_code: str | None = None,
    aggregate: bool = True,
) -> ForecastSeries:
    """Daily forecast points. With `aggregate=true` (default) series are summed per day.

    Aggregated bounds are sums of per-series bounds: quantiles are not additive, so the summed band
    is wider than the true band of the total (conservative)."""
    fr = await get_owned(db, ForecastRun, forecast_id, principal.organization_id)
    df = await run_in_threadpool(read_forecast_frame, fr, store_code)
    if "yhat_p50" not in df:  # forecasts made before quantile support
        df["yhat_p50"] = float("nan")
    if aggregate:
        num = ["yhat", "yhat_lower", "yhat_upper", "yhat_p50", "baseline_yhat"]
        df = df.groupby("ds", as_index=False)[num].sum(min_count=1)
        df["store_code"] = store_code or "__aggregate__"
        df["item_id"] = None
    df = df.sort_values("ds").astype(object).where(df.notna(), None)
    points = [
        ForecastPoint(**r)
        for r in df[
            ["ds", "store_code", "item_id", "yhat", "yhat_lower", "yhat_upper", "yhat_p50", "baseline_yhat"]
        ].to_dict(orient="records")
    ]
    return ForecastSeries(forecast_run_id=fr.id, points=points)
