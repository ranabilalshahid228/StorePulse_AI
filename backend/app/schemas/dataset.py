from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ORMModel


class SchemaMapping(BaseModel):
    """How raw columns map onto the canonical long format: (ds, store_code, item_id?, y, exog...).

    `layout="wide"` supports M5-style files where each day is a column (`d_1`, `d_2`, ...);
    `wide_start_date` is the calendar date of day index 1.
    """

    layout: Literal["long", "wide"] = "long"
    date_column: str | None = None
    target_column: str | None = None
    store_column: str | None = None
    item_column: str | None = None
    date_format: str | None = Field(default=None, description="strftime format; inferred if omitted")
    frequency: Literal["D", "W"] = "D"
    # Numeric/boolean covariates known in advance (promo, open, holiday flags, price ...).
    known_future_columns: list[str] = Field(default_factory=list)
    # Categorical covariates (encoded as category codes).
    categorical_columns: list[str] = Field(default_factory=list)
    # Optional secondary target captured alongside (e.g. Rossmann "Customers").
    customers_column: str | None = None
    # Semantic roles used by P&L / workforce. Must also appear in known_future_columns.
    promo_column: str | None = None
    open_column: str | None = None
    # Wide layout
    wide_value_prefix: str | None = None
    wide_start_date: date | None = None
    id_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> SchemaMapping:
        if self.layout == "long":
            if not self.date_column or not self.target_column:
                raise ValueError("long layout requires date_column and target_column")
        else:
            if not self.wide_value_prefix or not self.wide_start_date:
                raise ValueError("wide layout requires wide_value_prefix and wide_start_date")
            if self.known_future_columns or self.customers_column:
                raise ValueError("wide layout supports only id/store/item columns; join covariates later")
        for role in (self.promo_column, self.open_column):
            if role and role not in self.known_future_columns:
                raise ValueError(f"'{role}' must also be listed in known_future_columns")
        overlap = set(self.known_future_columns) & set(self.categorical_columns)
        if overlap:
            raise ValueError(f"columns cannot be both numeric and categorical: {sorted(overlap)}")
        return self

    def referenced_columns(self) -> list[str]:
        cols = [
            self.date_column,
            self.target_column,
            self.store_column,
            self.item_column,
            self.customers_column,
            *self.known_future_columns,
            *self.categorical_columns,
            *self.id_columns,
        ]
        return [c for c in cols if c]


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str
    null_fraction: float
    sample_values: list[Any]


class DatasetRead(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    store_id: uuid.UUID | None
    name: str
    description: str | None
    source_filename: str
    file_format: str
    size_bytes: int
    status: str
    progress_pct: float
    stage: str | None
    status_message: str | None
    error: str | None
    row_count: int | None
    series_count: int | None
    date_min: date | None
    date_max: date | None
    frequency: str | None
    schema_mapping: dict[str, Any] | None
    suggested_mapping: dict[str, Any]
    column_profile: dict[str, Any]
    created_at: datetime


class DatasetHealth(BaseModel):
    dataset_id: uuid.UUID
    generated_at: datetime | None
    report: dict[str, Any] | None


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
