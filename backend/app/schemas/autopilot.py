"""Store Autopilot: settings, what-if scenario levers, requests and run payloads."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ORMModel

HorizonKind = Literal["next_day", "week", "month", "custom"]


class ShiftDefinition(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    # Share of the day's footfall served in this shift; shares are normalized to sum to 1.
    traffic_share: float = Field(gt=0, le=1)
    # Shift length; None splits operating_hours_per_day evenly across shifts.
    hours: float | None = Field(default=None, gt=0, le=24)


def _default_shifts() -> list[ShiftDefinition]:
    return [
        ShiftDefinition(name="Morning", traffic_share=0.30),
        ShiftDefinition(name="Midday Peak", traffic_share=0.45),
        ShiftDefinition(name="Evening", traffic_share=0.25),
    ]


class AutopilotSettings(BaseModel):
    """Operational rules for the Autopilot, stored on the store's financial config.

    The defaults are planning conventions (a three-shift retail day), not data: every value used is
    echoed in the run's assumptions and can be overridden per store."""

    shifts: list[ShiftDefinition] = Field(default_factory=_default_shifts, min_length=1, max_length=6)
    peak_hour_share: float = Field(
        default=0.14, gt=0, le=1, description="Share of daily footfall arriving in the busiest hour"
    )
    max_staff_per_shift: int | None = Field(
        default=None, ge=1, description="Store capacity: most staff that can work one shift"
    )
    overstaff_utilization: float = Field(
        default=0.55, gt=0, lt=1, description="Scheduled hours below this utilization are flagged"
    )
    surge_block_hours: float = Field(default=2.0, gt=0, le=12, description="Length of a peak-surge cover")
    cold_start_min_days: int = Field(
        default=90, ge=7, le=730, description="Below this many open days of history a store is cold-start"
    )
    cohort_size: int = Field(default=10, ge=2, le=200)
    basket_lookback_days: int = Field(default=182, ge=28, le=730)
    anomaly_threshold: float = Field(
        default=0.35, gt=0, le=5, description="Flag forecasts this far (fraction) from the recent norm"
    )

    @model_validator(mode="after")
    def _unique_shift_names(self) -> AutopilotSettings:
        names = [s.name for s in self.shifts]
        if len(set(names)) != len(names):
            raise ValueError("shift names must be unique")
        return self


class ScenarioLever(BaseModel):
    """One what-if change on a date range. Flags change the model's inputs (it re-forecasts);
    `demand_pct` / `price_change_pct` adjust the forecast afterwards."""

    name: str = Field(min_length=1, max_length=80)
    kind: Literal["promo", "holiday", "closure", "local_event", "competitor", "price", "custom"] = "custom"
    start_date: date
    end_date: date
    promo: bool | None = None
    school_holiday: bool | None = None
    state_holiday: str | None = Field(default=None, max_length=8, description="Rossmann: 0, a, b or c")
    open: bool | None = None
    demand_pct: float | None = Field(
        default=None, ge=-1, le=5, description="Demand uplift (+) or shock (-), e.g. -0.15 = -15 %"
    )
    price_change_pct: float | None = Field(default=None, gt=-1, le=3)
    covariates: dict[str, float | str] = Field(
        default_factory=dict, description="Raw overrides for any known-future / categorical column"
    )

    @model_validator(mode="after")
    def _check(self) -> ScenarioLever:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self

    @property
    def changes_model_inputs(self) -> bool:
        return bool(
            self.covariates
            or any(v is not None for v in (self.promo, self.school_holiday, self.state_holiday, self.open))
        )


class AutopilotScenario(BaseModel):
    levers: list[ScenarioLever] = Field(default_factory=list, max_length=50)
    promo2: bool | None = Field(default=None, description="Toggle the store's Promo2 participation")
    promo2_interval: str | None = Field(
        default=None, max_length=64, description="Promo2 months when enabling it, e.g. Jan,Apr,Jul,Oct"
    )
    covariate_defaults: dict[str, float] = Field(
        default_factory=dict,
        description="Business-as-usual calendar for known-future covariates in baseline and scenario, "
        "e.g. {'Promo': 0}; unset ones repeat the last observed same-weekday value",
    )
    price_elasticity: float | None = Field(
        default=None, le=0, ge=-10, description="Volume response to price changes; None = volume unchanged"
    )
    cogs_pct: float | None = Field(default=None, ge=0, le=1, description="Margin adjustment: COGS override")
    promo_discount_pct: float | None = Field(default=None, ge=0, lt=1)

    @property
    def is_empty(self) -> bool:
        return (
            not self.levers
            and self.promo2 is None
            and self.cogs_pct is None
            and (self.promo_discount_pct is None)
        )

    @property
    def changes_model_inputs(self) -> bool:
        return self.promo2 is not None or any(lv.changes_model_inputs for lv in self.levers)


class AutopilotRequest(BaseModel):
    store_id: uuid.UUID
    model_run_id: uuid.UUID
    horizon: HorizonKind = "week"
    start_date: date | None = Field(default=None, description="custom horizon only")
    end_date: date | None = Field(default=None, description="custom horizon only")
    scenario: AutopilotScenario = Field(default_factory=AutopilotScenario)
    name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _custom_dates(self) -> AutopilotRequest:
        if self.horizon == "custom":
            if self.start_date is None or self.end_date is None:
                raise ValueError("custom horizon needs start_date and end_date")
            if self.end_date < self.start_date:
                raise ValueError("end_date must be on or after start_date")
        elif self.start_date or self.end_date:
            raise ValueError("start_date / end_date are only used with horizon='custom'")
        return self


class AutopilotRunRead(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    store_id: uuid.UUID
    model_run_id: uuid.UUID
    dataset_id: uuid.UUID
    name: str | None
    horizon: str
    start_date: date | None
    end_date: date | None
    scenario: dict[str, Any]
    summary: dict[str, Any]
    status: str
    progress_pct: float
    stage: str | None
    status_message: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class AutopilotRunDetail(AutopilotRunRead):
    result: dict[str, Any]
