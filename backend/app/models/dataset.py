from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, CheckConstraint, Date, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JobStateMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.tenancy import Store


class Dataset(UUIDPrimaryKeyMixin, TimestampMixin, JobStateMixin, Base):
    """An uploaded time-series file.

    `store_id` is set for single-store uploads; multi-store files (e.g. Rossmann train.csv)
    leave it NULL and rows are mapped to stores through the mapped `store` column.
    Stores referenced by code that don't exist yet are created during ingestion.
    """

    __tablename__ = "datasets"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    store_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stores.id", ondelete="SET NULL"), index=True
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source_filename: Mapped[str] = mapped_column(String(300), nullable=False)
    file_format: Mapped[str] = mapped_column(String(16), nullable=False)  # csv | parquet
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    raw_path: Mapped[str] = mapped_column(String(500), nullable=False)
    parquet_path: Mapped[str | None] = mapped_column(String(500))
    # primary: a time series to ingest/train on; auxiliary: a side table joined via SchemaMapping.joins
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="primary", server_default="primary")
    # `status` (from JobStateMixin) holds DatasetStatus values for datasets.
    # Column header + sample-based type inference produced at upload time.
    column_profile: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    suggested_mapping: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    # Confirmed mapping (see app.schemas.dataset.SchemaMapping).
    schema_mapping: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    health_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    row_count: Mapped[int | None] = mapped_column(BigInteger)
    series_count: Mapped[int | None] = mapped_column(BigInteger)
    date_min: Mapped[date | None] = mapped_column(Date)
    date_max: Mapped[date | None] = mapped_column(Date)
    frequency: Mapped[str | None] = mapped_column(String(8))

    store: Mapped[Store | None] = relationship(back_populates="datasets")

    __table_args__ = (CheckConstraint("role IN ('primary','auxiliary')", name="role_valid"),)
