"""initial schema: tenancy, stores, financial configs, datasets, model & forecast runs

Revision ID: 0001
Revises:
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = pg.UUID(as_uuid=True)
JSONB = pg.JSONB()


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def _job_state() -> list[sa.Column]:
    return [
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("progress_pct", sa.Float(), nullable=False),
        sa.Column("stage", sa.String(64)),
        sa.Column("status_message", sa.Text()),
        sa.Column("celery_task_id", sa.String(64)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
    ]


def _org_fk() -> sa.Column:
    return sa.Column(
        "organization_id",
        UUID,
        sa.ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", UUID, nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(80), nullable=False),
        sa.Column("plan", sa.String(32), nullable=False),
        sa.Column("default_currency", sa.String(3), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_organizations"),
        sa.UniqueConstraint("slug", name="uq_organizations_slug"),
    )

    op.create_table(
        "users",
        sa.Column("id", UUID, nullable=False),
        _org_fk(),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(200), nullable=False),
        sa.Column("full_name", sa.String(200)),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.CheckConstraint("role IN ('owner','admin','analyst','viewer')", name=op.f("ck_users_role_valid")),
    )
    op.create_index("ix_users_organization_id", "users", ["organization_id"])

    op.create_table(
        "stores",
        sa.Column("id", UUID, nullable=False),
        _org_fk(),
        sa.Column("code", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("store_type", sa.String(32)),
        sa.Column("assortment", sa.String(32)),
        sa.Column("competition_distance", sa.Float()),
        sa.Column("competition_open_since_month", sa.Integer()),
        sa.Column("competition_open_since_year", sa.Integer()),
        sa.Column("promo2", sa.Boolean(), nullable=False),
        sa.Column("promo2_since_week", sa.Integer()),
        sa.Column("promo2_since_year", sa.Integer()),
        sa.Column("promo_interval", sa.String(64)),
        sa.Column("target_type", sa.String(16), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("attributes", JSONB, nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_stores"),
        sa.UniqueConstraint("organization_id", "code", name="uq_stores_org_code"),
        sa.CheckConstraint(
            "target_type IN ('sales','units','customers')", name=op.f("ck_stores_target_type_valid")
        ),
        sa.CheckConstraint(
            "competition_distance IS NULL OR competition_distance >= 0",
            name=op.f("ck_stores_comp_dist_nonneg"),
        ),
    )
    op.create_index("ix_stores_organization_id", "stores", ["organization_id"])

    op.create_table(
        "financial_staffing_configs",
        sa.Column("id", UUID, nullable=False),
        _org_fk(),
        sa.Column("store_id", UUID, sa.ForeignKey("stores.id", ondelete="CASCADE"), nullable=False),
        sa.Column("gross_margin_pct", sa.Numeric(6, 4), nullable=False),
        sa.Column("cogs_pct", sa.Numeric(6, 4), nullable=False),
        sa.Column("fixed_daily_overhead", sa.Numeric(14, 2), nullable=False),
        sa.Column("promo_discount_pct", sa.Numeric(6, 4), nullable=False),
        sa.Column("hourly_wage", sa.Numeric(10, 2), nullable=False),
        sa.Column("customers_per_staff_hour", sa.Numeric(10, 2), nullable=False),
        sa.Column("min_staff_per_shift", sa.Integer(), nullable=False),
        sa.Column("shift_hours", sa.Numeric(4, 2), nullable=False),
        sa.Column("operating_hours_per_day", sa.Numeric(4, 2), nullable=False),
        sa.Column("avg_transaction_value", sa.Numeric(12, 2)),
        sa.Column("avg_units_per_transaction", sa.Numeric(10, 3)),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_financial_staffing_configs"),
        sa.UniqueConstraint("store_id", name="uq_financial_staffing_configs_store_id"),
        sa.CheckConstraint(
            "gross_margin_pct >= 0 AND gross_margin_pct <= 1",
            name=op.f("ck_financial_staffing_configs_gm_range"),
        ),
        sa.CheckConstraint(
            "cogs_pct >= 0 AND cogs_pct <= 1", name=op.f("ck_financial_staffing_configs_cogs_range")
        ),
        sa.CheckConstraint(
            "promo_discount_pct >= 0 AND promo_discount_pct < 1",
            name=op.f("ck_financial_staffing_configs_promo_range"),
        ),
        sa.CheckConstraint("hourly_wage >= 0", name=op.f("ck_financial_staffing_configs_wage_nonneg")),
        sa.CheckConstraint(
            "customers_per_staff_hour > 0", name=op.f("ck_financial_staffing_configs_cpsh_pos")
        ),
        sa.CheckConstraint(
            "min_staff_per_shift >= 0", name=op.f("ck_financial_staffing_configs_min_staff_nonneg")
        ),
        sa.CheckConstraint(
            "shift_hours > 0 AND shift_hours <= 24", name=op.f("ck_financial_staffing_configs_shift_range")
        ),
        sa.CheckConstraint(
            "operating_hours_per_day > 0 AND operating_hours_per_day <= 24",
            name=op.f("ck_financial_staffing_configs_op_hours_range"),
        ),
    )
    op.create_index(
        "ix_financial_staffing_configs_organization_id", "financial_staffing_configs", ["organization_id"]
    )

    op.create_table(
        "datasets",
        sa.Column("id", UUID, nullable=False),
        _org_fk(),
        sa.Column("store_id", UUID, sa.ForeignKey("stores.id", ondelete="SET NULL")),
        sa.Column("created_by_id", UUID, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("source_filename", sa.String(300), nullable=False),
        sa.Column("file_format", sa.String(16), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("raw_path", sa.String(500), nullable=False),
        sa.Column("parquet_path", sa.String(500)),
        sa.Column("column_profile", JSONB, nullable=False),
        sa.Column("suggested_mapping", JSONB, nullable=False),
        sa.Column("schema_mapping", JSONB),
        sa.Column("health_report", JSONB),
        sa.Column("row_count", sa.BigInteger()),
        sa.Column("series_count", sa.BigInteger()),
        sa.Column("date_min", sa.Date()),
        sa.Column("date_max", sa.Date()),
        sa.Column("frequency", sa.String(8)),
        *_job_state(),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_datasets"),
    )
    op.create_index("ix_datasets_organization_id", "datasets", ["organization_id"])
    op.create_index("ix_datasets_store_id", "datasets", ["store_id"])
    op.create_index("ix_datasets_status", "datasets", ["status"])

    op.create_table(
        "model_runs",
        sa.Column("id", UUID, nullable=False),
        _org_fk(),
        sa.Column("dataset_id", UUID, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("store_id", UUID, sa.ForeignKey("stores.id", ondelete="SET NULL")),
        sa.Column("created_by_id", UUID, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(200)),
        sa.Column("model_family", sa.String(32), nullable=False),
        sa.Column("run_type", sa.String(16), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("validation_days", sa.Integer(), nullable=False),
        sa.Column("hyperparameters", JSONB, nullable=False),
        sa.Column("search_space", JSONB),
        sa.Column("n_trials", sa.Integer()),
        sa.Column("trials", JSONB, nullable=False),
        sa.Column("epoch_logs", JSONB, nullable=False),
        sa.Column("metrics", JSONB, nullable=False),
        sa.Column("feature_columns", JSONB, nullable=False),
        sa.Column("artifact_path", sa.String(500)),
        sa.Column("plot_paths", JSONB, nullable=False),
        sa.Column("inference_ms", sa.Float()),
        sa.Column("train_rows", sa.Integer()),
        sa.Column("device", sa.String(16)),
        *_job_state(),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_model_runs"),
        sa.CheckConstraint("run_type IN ('train','tune')", name=op.f("ck_model_runs_run_type_valid")),
        sa.CheckConstraint(
            "horizon_days > 0 AND horizon_days <= 365", name=op.f("ck_model_runs_horizon_range")
        ),
        sa.CheckConstraint(
            "progress_pct >= 0 AND progress_pct <= 100", name=op.f("ck_model_runs_progress_range")
        ),
    )
    for col in ("organization_id", "dataset_id", "store_id", "model_family", "status"):
        op.create_index(f"ix_model_runs_{col}", "model_runs", [col])

    op.create_table(
        "forecast_runs",
        sa.Column("id", UUID, nullable=False),
        _org_fk(),
        sa.Column("model_run_id", UUID, sa.ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", UUID, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("store_id", UUID, sa.ForeignKey("stores.id", ondelete="SET NULL")),
        sa.Column("created_by_id", UUID, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("scenario", JSONB, nullable=False),
        sa.Column("output_path", sa.String(500)),
        sa.Column("summary", JSONB, nullable=False),
        *_job_state(),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_forecast_runs"),
        sa.CheckConstraint("kind IN ('multi_horizon','event')", name=op.f("ck_forecast_runs_kind_valid")),
        sa.CheckConstraint(
            "horizon_days > 0 AND horizon_days <= 365", name=op.f("ck_forecast_runs_horizon_range")
        ),
        sa.CheckConstraint(
            "progress_pct >= 0 AND progress_pct <= 100", name=op.f("ck_forecast_runs_progress_range")
        ),
    )
    for col in ("organization_id", "model_run_id", "dataset_id", "store_id", "status"):
        op.create_index(f"ix_forecast_runs_{col}", "forecast_runs", [col])


def downgrade() -> None:
    op.drop_table("forecast_runs")
    op.drop_table("model_runs")
    op.drop_table("datasets")
    op.drop_table("financial_staffing_configs")
    op.drop_table("stores")
    op.drop_table("users")
    op.drop_table("organizations")
