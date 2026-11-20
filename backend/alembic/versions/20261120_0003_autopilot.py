"""autopilot_runs table and financial_staffing_configs.autopilot_settings

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = pg.UUID(as_uuid=True)
JSONB = pg.JSONB()


def upgrade() -> None:
    op.add_column(
        "financial_staffing_configs",
        sa.Column("autopilot_settings", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_table(
        "autopilot_runs",
        sa.Column("id", UUID, nullable=False),
        sa.Column(
            "organization_id", UUID, sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("store_id", UUID, sa.ForeignKey("stores.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_run_id", UUID, sa.ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", UUID, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by_id", UUID, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(200)),
        sa.Column("horizon", sa.String(16), nullable=False),
        sa.Column("start_date", sa.Date()),
        sa.Column("end_date", sa.Date()),
        sa.Column("scenario", JSONB, nullable=False),
        sa.Column("summary", JSONB, nullable=False),
        sa.Column("result", JSONB, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("progress_pct", sa.Float(), nullable=False),
        sa.Column("stage", sa.String(64)),
        sa.Column("status_message", sa.Text()),
        sa.Column("celery_task_id", sa.String(64)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_autopilot_runs"),
        sa.CheckConstraint(
            "horizon IN ('next_day','week','month','custom')",
            name=op.f("ck_autopilot_runs_horizon_kind_valid"),
        ),
        sa.CheckConstraint(
            "progress_pct >= 0 AND progress_pct <= 100", name=op.f("ck_autopilot_runs_progress_range")
        ),
    )
    for col in ("organization_id", "store_id", "model_run_id", "dataset_id", "status"):
        op.create_index(f"ix_autopilot_runs_{col}", "autopilot_runs", [col])


def downgrade() -> None:
    op.drop_table("autopilot_runs")
    op.drop_column("financial_staffing_configs", "autopilot_settings")
