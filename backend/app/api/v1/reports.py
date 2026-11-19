from __future__ import annotations

import uuid

from fastapi import APIRouter
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.api.deps import CurrentPrincipal, DbSession
from app.api.v1.finance import build_context
from app.api.v1.forecasts import read_forecast_frame
from app.models import ForecastRun, ModelRun, Organization
from app.services import reports
from app.services.tenancy import get_owned

router = APIRouter(prefix="/reports", tags=["reports"])


def _download(content: bytes, filename: str, media_type: str) -> Response:
    return Response(
        content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/forecasts/{forecast_id}/forecast.csv")
async def forecast_csv(
    forecast_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession, store_code: str | None = None
) -> Response:
    """Raw daily forecast rows (all series, or one store)."""
    fr = await get_owned(db, ForecastRun, forecast_id, principal.organization_id)
    df = await run_in_threadpool(read_forecast_frame, fr, store_code)
    return _download(reports.to_csv_bytes(df), f"forecast_{forecast_id}.csv", "text/csv")


@router.get("/forecasts/{forecast_id}/store-plan.csv")
async def store_plan_csv(
    forecast_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession, store_id: uuid.UUID | None = None
) -> Response:
    """Daily P&L joined with the workforce plan for one store."""
    ctx = await build_context(forecast_id, store_id, principal, db)
    merged = ctx.pnl_days.merge(ctx.workforce.drop(columns=["is_open", "labor_cost"]), on="ds")
    return _download(
        reports.to_csv_bytes(merged), f"store_plan_{ctx.store.code}_{forecast_id}.csv", "text/csv"
    )


@router.get("/forecasts/{forecast_id}/store-plan.pdf")
async def store_plan_pdf(
    forecast_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession, store_id: uuid.UUID | None = None
) -> Response:
    ctx = await build_context(forecast_id, store_id, principal, db)
    run = await get_owned(db, ModelRun, ctx.forecast.model_run_id, principal.organization_id)
    org = await db.get(Organization, principal.organization_id)
    daily = ctx.pnl_days.merge(ctx.workforce[["ds", "staff_per_shift", "scheduled_staff_hours"]], on="ds")
    daily = daily[
        [
            "ds",
            "forecast_target",
            "is_promo",
            "net_revenue",
            "gross_profit",
            "labor_cost",
            "operating_profit",
            "staff_per_shift",
            "scheduled_staff_hours",
        ]
    ]
    pdf = await run_in_threadpool(
        reports.forecast_pdf,
        title=f"Store plan - {ctx.store.name}",
        org_name=org.name if org else "",
        store={
            "code": ctx.store.code,
            "name": ctx.store.name,
            "currency": ctx.store.currency,
            "target_type": ctx.store.target_type,
        },
        model={"model_family": run.model_family, "run_type": run.run_type, "metrics": run.metrics},
        forecast={"summary": ctx.forecast.summary, "kind": ctx.forecast.kind},
        daily=daily,
        totals=ctx.totals,
        assumptions=ctx.pnl_assumptions + ctx.workforce_assumptions,
    )
    return _download(pdf, f"store_plan_{ctx.store.code}_{forecast_id}.pdf", "application/pdf")
