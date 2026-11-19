from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.enums import TargetType
from app.schemas.common import ORMModel

Fraction = Field(ge=0, le=1, description="Fraction in [0, 1], e.g. 0.35 for 35 %")


class StoreBase(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    store_type: str | None = Field(default=None, max_length=32)
    assortment: str | None = Field(default=None, max_length=32)
    competition_distance: float | None = Field(default=None, ge=0)
    competition_open_since_month: int | None = Field(default=None, ge=1, le=12)
    competition_open_since_year: int | None = Field(default=None, ge=1900, le=2100)
    promo2: bool = False
    promo2_since_week: int | None = Field(default=None, ge=1, le=53)
    promo2_since_year: int | None = Field(default=None, ge=1900, le=2100)
    promo_interval: str | None = Field(default=None, max_length=64)
    target_type: TargetType = TargetType.SALES
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    timezone: str = "Europe/Berlin"
    attributes: dict[str, Any] = Field(default_factory=dict)


class StoreCreate(StoreBase):
    pass


class StoreUpdate(BaseModel):
    name: str | None = None
    store_type: str | None = None
    assortment: str | None = None
    competition_distance: float | None = Field(default=None, ge=0)
    competition_open_since_month: int | None = Field(default=None, ge=1, le=12)
    competition_open_since_year: int | None = None
    promo2: bool | None = None
    promo2_since_week: int | None = Field(default=None, ge=1, le=53)
    promo2_since_year: int | None = None
    promo_interval: str | None = None
    target_type: TargetType | None = None
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    timezone: str | None = None
    attributes: dict[str, Any] | None = None


class StoreRead(StoreBase, ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    created_at: datetime
    has_financial_config: bool = False


class FinancialStaffingConfigIn(BaseModel):
    gross_margin_pct: Decimal = Fraction
    cogs_pct: Decimal = Fraction
    fixed_daily_overhead: Decimal = Field(ge=0)
    promo_discount_pct: Decimal = Field(default=Decimal("0"), ge=0, lt=1)
    hourly_wage: Decimal = Field(ge=0)
    customers_per_staff_hour: Decimal = Field(gt=0)
    min_staff_per_shift: int = Field(ge=0)
    shift_hours: Decimal = Field(gt=0, le=24)
    operating_hours_per_day: Decimal = Field(default=Decimal("12"), gt=0, le=24)
    avg_transaction_value: Decimal | None = Field(default=None, gt=0)
    avg_units_per_transaction: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _margin_consistency(self) -> FinancialStaffingConfigIn:
        # COGS is charged on undiscounted volume and gross margin is the planned margin
        # before discounts; together they cannot exceed 100 % of revenue.
        if self.gross_margin_pct + self.cogs_pct > Decimal("1.0001"):
            raise ValueError("gross_margin_pct + cogs_pct must not exceed 1")
        return self


class FinancialStaffingConfigRead(FinancialStaffingConfigIn, ORMModel):
    id: uuid.UUID
    store_id: uuid.UUID
    updated_at: datetime
