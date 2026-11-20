"""Organizations, users, stores and per-store financial/staffing configuration."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import TargetType, UserRole

if TYPE_CHECKING:
    from app.models.dataset import Dataset


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    plan: Mapped[str] = mapped_column(String(32), nullable=False, default="trial")
    default_currency: Mapped[str] = mapped_column(String(3), nullable=False, default="EUR")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    users: Mapped[list[User]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    stores: Mapped[list[Store]] = relationship(back_populates="organization", cascade="all, delete-orphan")


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(String(200), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(32), nullable=False, default=UserRole.ANALYST.value)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    organization: Mapped[Organization] = relationship(back_populates="users")

    __table_args__ = (CheckConstraint("role IN ('owner','admin','analyst','viewer')", name="role_valid"),)


class Store(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "stores"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # External identifier used in uploaded data (e.g. Rossmann "Store" = 1, M5 "CA_1").
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    store_type: Mapped[str | None] = mapped_column(String(32))
    assortment: Mapped[str | None] = mapped_column(String(32))
    competition_distance: Mapped[float | None] = mapped_column(Float)
    competition_open_since_month: Mapped[int | None] = mapped_column(Integer)
    competition_open_since_year: Mapped[int | None] = mapped_column(Integer)
    promo2: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promo2_since_week: Mapped[int | None] = mapped_column(Integer)
    promo2_since_year: Mapped[int | None] = mapped_column(Integer)
    promo_interval: Mapped[str | None] = mapped_column(String(64))  # e.g. "Jan,Apr,Jul,Oct"
    target_type: Mapped[str] = mapped_column(String(16), nullable=False, default=TargetType.SALES.value)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="EUR")
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="Europe/Berlin")
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    organization: Mapped[Organization] = relationship(back_populates="stores")
    financial_config: Mapped[FinancialStaffingConfig | None] = relationship(
        back_populates="store", uselist=False, cascade="all, delete-orphan"
    )
    datasets: Mapped[list[Dataset]] = relationship(back_populates="store")

    __table_args__ = (
        UniqueConstraint("organization_id", "code", name="uq_stores_org_code"),
        CheckConstraint("target_type IN ('sales','units','customers')", name="target_type_valid"),
        CheckConstraint("competition_distance IS NULL OR competition_distance >= 0", name="comp_dist_nonneg"),
    )


class FinancialStaffingConfig(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Unit economics & labour model for one store. All P&L/workforce math reads from here.

    Percentages are fractions in [0, 1] (0.35 == 35 %).
    """

    __tablename__ = "financial_staffing_configs"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    store_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stores.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    # Finance
    gross_margin_pct: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    cogs_pct: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    fixed_daily_overhead: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    promo_discount_pct: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False, default=0)
    # Workforce
    hourly_wage: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    customers_per_staff_hour: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    min_staff_per_shift: Mapped[int] = mapped_column(Integer, nullable=False)
    shift_hours: Mapped[Decimal] = mapped_column(Numeric(4, 2), nullable=False)
    operating_hours_per_day: Mapped[Decimal] = mapped_column(Numeric(4, 2), nullable=False, default=12)
    # Needed to convert a sales/units forecast into footfall when target_type != customers.
    avg_transaction_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    avg_units_per_transaction: Mapped[Decimal | None] = mapped_column(Numeric(10, 3))
    # Autopilot operating rules (shift profile, capacity, thresholds); see
    # schemas.autopilot.AutopilotSettings.
    autopilot_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    store: Mapped[Store] = relationship(back_populates="financial_config")

    __table_args__ = (
        CheckConstraint("gross_margin_pct >= 0 AND gross_margin_pct <= 1", name="gm_range"),
        CheckConstraint("cogs_pct >= 0 AND cogs_pct <= 1", name="cogs_range"),
        CheckConstraint("promo_discount_pct >= 0 AND promo_discount_pct < 1", name="promo_range"),
        CheckConstraint("hourly_wage >= 0", name="wage_nonneg"),
        CheckConstraint("customers_per_staff_hour > 0", name="cpsh_pos"),
        CheckConstraint("min_staff_per_shift >= 0", name="min_staff_nonneg"),
        CheckConstraint("shift_hours > 0 AND shift_hours <= 24", name="shift_range"),
        CheckConstraint(
            "operating_hours_per_day > 0 AND operating_hours_per_day <= 24", name="op_hours_range"
        ),
    )
