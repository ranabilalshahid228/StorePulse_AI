from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.enums import ModelFamily
from app.schemas.common import ORMModel


class TrainRequest(BaseModel):
    dataset_id: uuid.UUID
    model_family: ModelFamily
    store_id: uuid.UUID | None = None
    name: str | None = Field(default=None, max_length=200)
    horizon_days: int = Field(default=42, ge=1, le=365)
    validation_days: int | None = Field(default=None, ge=7, le=365)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    history_days: int | None = Field(
        default=None, ge=56, le=3650, description="Train only on the most recent N days of targets"
    )
    use_store_features: bool = Field(
        default=True, description="Add store metadata features (type, assortment, competition, promo2)"
    )

    def training_config(self) -> dict[str, Any]:
        return {"history_days": self.history_days, "use_store_features": self.use_store_features}


class TuneRequest(TrainRequest):
    n_trials: int | None = Field(default=None, ge=1, le=500)
    timeout_minutes: int | None = Field(default=None, ge=1, le=24 * 60)
    metric: Literal["wape", "rmse", "mae", "rmspe", "rmsse"] = "wape"
    search_space: dict[str, Any] | None = Field(
        default=None, description="Override per-family default search space"
    )


class ModelRunRead(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    dataset_id: uuid.UUID
    store_id: uuid.UUID | None
    name: str | None
    model_family: str
    run_type: str
    horizon_days: int
    validation_days: int
    hyperparameters: dict[str, Any]
    search_space: dict[str, Any] | None
    n_trials: int | None
    metrics: dict[str, Any]
    feature_columns: list[str]
    training_config: dict[str, Any]
    artifact_path: str | None
    plot_paths: list[str]
    inference_ms: float | None
    train_rows: int | None
    device: str | None
    status: str
    progress_pct: float
    stage: str | None
    status_message: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class ModelRunDetail(ModelRunRead):
    epoch_logs: list[dict[str, Any]]
    trials: list[dict[str, Any]]


class LeaderboardEntry(BaseModel):
    rank: int
    model_run_id: uuid.UUID
    name: str | None
    model_family: str
    run_type: str
    metric_value: float
    metrics: dict[str, Any]
    inference_ms: float | None
    finished_at: datetime | None


class Leaderboard(BaseModel):
    dataset_id: uuid.UUID
    metric: str
    entries: list[LeaderboardEntry]


class ModelFamilyInfo(BaseModel):
    family: ModelFamily
    description: str
    default_hyperparameters: dict[str, Any]
    search_space: dict[str, Any]
    supports_gpu: bool
    available: bool


# ----------------------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------------------


class ScenarioEvent(BaseModel):
    """Overrides for known-future covariates over a date range (inclusive)."""

    name: str = Field(max_length=120)
    start_date: date
    end_date: date
    store_codes: list[str] | None = Field(default=None, description="None = all stores in the forecast")
    overrides: dict[str, float] = Field(
        description="Covariate -> value, e.g. {'Promo': 1} or {'Open': 0}. Keys must be model features."
    )

    @model_validator(mode="after")
    def _dates(self) -> ScenarioEvent:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class ForecastRequest(BaseModel):
    model_run_id: uuid.UUID
    horizon_days: int = Field(default=42, ge=1, le=365)
    store_id: uuid.UUID | None = None
    # Default per-covariate future values; otherwise last observed same-weekday value is used.
    covariate_defaults: dict[str, float] = Field(default_factory=dict)


class EventForecastRequest(ForecastRequest):
    events: list[ScenarioEvent] = Field(min_length=1)
    compare_to_baseline: bool = True


class ForecastRunRead(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    model_run_id: uuid.UUID
    dataset_id: uuid.UUID
    store_id: uuid.UUID | None
    kind: str
    horizon_days: int
    scenario: dict[str, Any]
    output_path: str | None
    summary: dict[str, Any]
    status: str
    progress_pct: float
    stage: str | None
    status_message: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class ForecastPoint(BaseModel):
    ds: date
    store_code: str
    item_id: str | None = None
    yhat: float
    yhat_lower: float | None = None
    yhat_upper: float | None = None
    yhat_p50: float | None = None
    baseline_yhat: float | None = None


class ForecastSeries(BaseModel):
    forecast_run_id: uuid.UUID
    points: list[ForecastPoint]
