from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, JobStateMixin, TimestampMixin, UUIDPrimaryKeyMixin


class ModelRun(UUIDPrimaryKeyMixin, TimestampMixin, JobStateMixin, Base):
    """One training or Optuna tuning job for a model family on a dataset."""

    __tablename__ = "model_runs"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Optional filter: train on a single store's series only.
    store_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stores.id", ondelete="SET NULL"), index=True
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str | None] = mapped_column(String(200))
    model_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    run_type: Mapped[str] = mapped_column(String(16), nullable=False, default="train")
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_days: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    search_space: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    n_trials: Mapped[int | None] = mapped_column(Integer)
    trials: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    epoch_logs: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    feature_columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    # Non-hyperparameter run settings: history_days, use_store_features, ...
    training_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    artifact_path: Mapped[str | None] = mapped_column(String(500))
    plot_paths: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    inference_ms: Mapped[float | None] = mapped_column(Float)  # mean latency per 1k rows
    train_rows: Mapped[int | None] = mapped_column(Integer)
    device: Mapped[str | None] = mapped_column(String(16))

    __table_args__ = (
        CheckConstraint("run_type IN ('train','tune')", name="run_type_valid"),
        CheckConstraint("horizon_days > 0 AND horizon_days <= 365", name="horizon_range"),
        CheckConstraint("progress_pct >= 0 AND progress_pct <= 100", name="progress_range"),
    )


class ForecastRun(UUIDPrimaryKeyMixin, TimestampMixin, JobStateMixin, Base):
    """Batch inference with a trained model; output is a Parquet file of daily predictions."""

    __tablename__ = "forecast_runs"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    store_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stores.id", ondelete="SET NULL"), index=True
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="multi_horizon")
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    # Event scenario: overrides for known-future covariates (promo days, closures, holidays).
    scenario: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    output_path: Mapped[str | None] = mapped_column(String(500))
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        CheckConstraint("kind IN ('multi_horizon','event')", name="kind_valid"),
        CheckConstraint("horizon_days > 0 AND horizon_days <= 365", name="horizon_range"),
        CheckConstraint("progress_pct >= 0 AND progress_pct <= 100", name="progress_range"),
    )
