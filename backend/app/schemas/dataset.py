from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ORMModel

CANONICAL_KEYS = ("ds", "store_code", "item_id")


class JoinSpec(BaseModel):
    """Left-join columns from an *auxiliary* dataset onto the canonical frame.

    Joins run in order, at ingestion and again at forecast time for future dates, so covariates
    known in advance (calendar events, planned prices) get their real future values.
    `left_on` may reference canonical keys (ds, store_code, item_id), categorical id columns, or
    columns produced by earlier joins. A right key joined to `ds` is parsed as a date.
    """

    dataset_id: uuid.UUID
    left_on: list[str] = Field(min_length=1)
    right_on: list[str] = Field(min_length=1)
    columns: list[str] = Field(min_length=1, description="Right-side columns to add")

    @model_validator(mode="after")
    def _keys(self) -> JoinSpec:
        if len(self.left_on) != len(self.right_on):
            raise ValueError("left_on and right_on must have the same length")
        if set(self.columns) & set(CANONICAL_KEYS):
            raise ValueError("joined columns cannot overwrite canonical keys")
        return self


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
    price_column: str | None = Field(
        default=None, description="Unit selling price; lets P&L value a units forecast as units x price"
    )
    # Wide layout
    wide_value_prefix: str | None = None
    wide_start_date: date | None = None
    id_columns: list[str] = Field(default_factory=list)
    # Multi-file datasets
    joins: list[JoinSpec] = Field(default_factory=list)
    helper_columns: list[str] = Field(
        default_factory=list, description="Join outputs used only as keys for later joins; dropped after"
    )
    # Ingest only these store codes (useful for very large multi-store files such as M5).
    store_filter: list[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> SchemaMapping:
        if self.layout == "long":
            if not self.date_column or not self.target_column:
                raise ValueError("long layout requires date_column and target_column")
        else:
            if not self.wide_value_prefix or not self.wide_start_date:
                raise ValueError("wide layout requires wide_value_prefix and wide_start_date")
            if self.customers_column:
                raise ValueError("wide layout does not support customers_column")
            joined = self.joined_columns()
            raw_numeric = set(self.known_future_columns) - joined
            if raw_numeric:
                raise ValueError(f"wide layout covariates must come from joins: {sorted(raw_numeric)}")
            raw_cats = set(self.categorical_columns) - joined - set(self.id_columns)
            if raw_cats:
                raise ValueError(
                    f"wide-layout categorical columns must be id_columns or joined: {sorted(raw_cats)}"
                )
        for role in (self.promo_column, self.open_column, self.price_column):
            if role and role not in self.known_future_columns:
                raise ValueError(f"'{role}' must also be listed in known_future_columns")
        overlap = set(self.known_future_columns) & set(self.categorical_columns)
        if overlap:
            raise ValueError(f"columns cannot be both numeric and categorical: {sorted(overlap)}")
        # Every joined column must have a role; join keys must be resolvable in order.
        available = set(CANONICAL_KEYS) | set(self.categorical_columns) - self.joined_columns()
        roles = set(self.known_future_columns) | set(self.categorical_columns) | set(self.helper_columns)
        for i, j in enumerate(self.joins):
            missing = set(j.left_on) - available
            if missing:
                raise ValueError(f"join #{i + 1}: left_on columns not available yet: {sorted(missing)}")
            unassigned = set(j.columns) - roles
            if unassigned:
                raise ValueError(
                    f"join #{i + 1}: list {sorted(unassigned)} in known_future_columns, "
                    "categorical_columns or helper_columns"
                )
            available |= set(j.columns)
        return self

    def joined_columns(self) -> set[str]:
        return {c for j in self.joins for c in j.columns}

    def referenced_columns(self) -> list[str]:
        """Columns that must exist in the primary (raw) file."""
        joined = self.joined_columns()
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
        return list(dict.fromkeys(c for c in cols if c and c not in joined))


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
    role: str
    created_at: datetime


class DatasetHealth(BaseModel):
    dataset_id: uuid.UUID
    generated_at: datetime | None
    report: dict[str, Any] | None


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
