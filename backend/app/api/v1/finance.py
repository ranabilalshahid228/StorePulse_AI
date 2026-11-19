"""P&L and workforce endpoints. All numbers derive from a stored forecast + the store's config."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd
import pyarrow.dataset as pads
from fastapi import APIRouter
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.api.deps import CurrentPrincipal, DbSession
from app.api.v1.forecasts import read_forecast_frame
from app.core.exceptions import DomainValidationError, NotFoundError
from app.models import Dataset, FinancialStaffingConfig, ForecastRun, Store
from app.models.enums import TargetType
from app.schemas.finance import PnLDay, PnLReport, PnLTotals, WorkforceDay, WorkforcePlan
from app.services import finance
from app.services.storage import get_storage
from app.services.tenancy import get_owned

router = APIRouter(prefix="/forecasts/{forecast_id}", tags=["pnl-workforce"])
ATV_LOOKBACK_DAYS = 90


@dataclass
class FinanceContext:
    forecast: ForecastRun
    store: Store
    days: pd.DataFrame
    pnl_days: pd.DataFrame
    totals: dict
    workforce: pd.DataFrame
    pnl_assumptions: list[str]
    workforce_assumptions: list[str]


def _historical_atv(dataset: Dataset, store_code: str) -> float | None:
    """Average transaction value = sum(sales) / sum(customers) over the last 90 observed days."""
    if not dataset.parquet_path or not (dataset.schema_mapping or {}).get("customers_column"):
        return None
    path = get_storage().local_path(dataset.parquet_path)
    t = (
        pads.dataset(path, format="parquet")
        .to_table(columns=["ds", "y", "customers"], filter=pads.field("store_code") == store_code)
        .to_pandas()
    )
    t = t[(t["customers"] > 0) & t["y"].notna()]
    if t.empty:
        return None
    t["ds"] = pd.to_datetime(t["ds"])
    t = t[t["ds"] > t["ds"].max() - pd.Timedelta(days=ATV_LOOKBACK_DAYS)]
    return float(t["y"].sum() / t["customers"].sum()) if t["customers"].sum() > 0 else None


async def build_context(
    forecast_id: uuid.UUID, store_id: uuid.UUID | None, principal: CurrentPrincipal, db: DbSession
) -> FinanceContext:
    fr = await get_owned(db, ForecastRun, forecast_id, principal.organization_id)
    sid = store_id or fr.store_id
    if sid is None:
        raise DomainValidationError("store_id is required for multi-store forecasts")
    store = await get_owned(db, Store, sid, principal.organization_id)
    cfg = (
        await db.execute(select(FinancialStaffingConfig).where(FinancialStaffingConfig.store_id == store.id))
    ).scalar_one_or_none()
    if cfg is None:
        raise NotFoundError("Set the store's financial & staffing config before running P&L / workforce")
    dataset = await get_owned(db, Dataset, fr.dataset_id, principal.organization_id)

    df = await run_in_threadpool(read_forecast_frame, fr, store.code)
    if df.empty:
        raise NotFoundError(f"Forecast has no rows for store '{store.code}'")
    # Item-level forecasts roll up to store-day for finance.
    days = (
        df.groupby("ds", as_index=False)
        .agg(yhat=("yhat", "sum"), is_promo=("is_promo", "max"), is_open=("is_open", "max"))
        .sort_values("ds", ignore_index=True)
    )
    atv = None
    if cfg.avg_transaction_value is None and store.target_type == TargetType.SALES:
        atv = await run_in_threadpool(_historical_atv, dataset, store.code)
    inputs = finance.FinanceInputs.from_config(cfg, store.target_type, avg_transaction_value=atv)
    wf, wf_notes = finance.workforce_plan(days, inputs)
    pnl_days, totals, pnl_notes = finance.pnl(days, wf, inputs)
    if atv is not None:
        wf_notes.insert(
            0,
            f"avg_transaction_value {atv:.2f} derived from the last {ATV_LOOKBACK_DAYS} days "
            f"of sales/customers history.",
        )
    return FinanceContext(fr, store, days, pnl_days, totals, wf, pnl_notes, wf_notes)


@router.get("/pnl", response_model=PnLReport)
async def forecast_pnl(
    forecast_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession, store_id: uuid.UUID | None = None
) -> PnLReport:
    ctx = await build_context(forecast_id, store_id, principal, db)
    return PnLReport(
        forecast_run_id=ctx.forecast.id,
        store_id=ctx.store.id,
        store_code=ctx.store.code,
        currency=ctx.store.currency,
        target_type=ctx.store.target_type,
        days=[PnLDay(**r) for r in ctx.pnl_days.to_dict(orient="records")],
        totals=PnLTotals(**ctx.totals),
        assumptions=ctx.pnl_assumptions,
    )


@router.get("/workforce", response_model=WorkforcePlan)
async def forecast_workforce(
    forecast_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession, store_id: uuid.UUID | None = None
) -> WorkforcePlan:
    ctx = await build_context(forecast_id, store_id, principal, db)
    wf = ctx.workforce
    return WorkforcePlan(
        forecast_run_id=ctx.forecast.id,
        store_id=ctx.store.id,
        store_code=ctx.store.code,
        currency=ctx.store.currency,
        days=[WorkforceDay(**r) for r in wf.to_dict(orient="records")],
        total_scheduled_hours=round(float(wf["scheduled_staff_hours"].sum()), 2),
        total_labor_cost=round(float(wf["labor_cost"].sum()), 2),
        peak_staff_per_shift=int(wf["staff_per_shift"].max()),
        assumptions=ctx.workforce_assumptions,
    )
