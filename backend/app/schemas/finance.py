from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel


class PnLDay(BaseModel):
    ds: date
    forecast_target: float
    is_promo: bool
    is_open: bool
    gross_revenue: float
    promo_discount: float
    net_revenue: float
    cogs: float
    gross_profit: float
    labor_cost: float
    fixed_overhead: float
    operating_profit: float


class PnLTotals(BaseModel):
    gross_revenue: float
    promo_discount: float
    net_revenue: float
    cogs: float
    gross_profit: float
    realized_gross_margin_pct: float | None
    planned_gross_margin_pct: float
    labor_cost: float
    fixed_overhead: float
    operating_profit: float
    operating_margin_pct: float | None


class PnLReport(BaseModel):
    forecast_run_id: uuid.UUID
    store_id: uuid.UUID
    store_code: str
    currency: str
    target_type: str
    days: list[PnLDay]
    totals: PnLTotals
    assumptions: list[str]


class WorkforceDay(BaseModel):
    ds: date
    is_open: bool
    expected_customers: float
    required_staff_hours: float
    shifts: int
    staff_per_shift: int
    scheduled_staff_hours: float
    labor_cost: float
    utilization_pct: float | None


class WorkforcePlan(BaseModel):
    forecast_run_id: uuid.UUID
    store_id: uuid.UUID
    store_code: str
    currency: str
    days: list[WorkforceDay]
    total_scheduled_hours: float
    total_labor_cost: float
    peak_staff_per_shift: int
    assumptions: list[str]
