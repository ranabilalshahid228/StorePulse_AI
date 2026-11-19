"""String enums persisted as VARCHAR (no native PG enums -> painless migrations)."""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class TargetType(StrEnum):
    SALES = "sales"  # revenue in store currency
    UNITS = "units"  # units sold
    CUSTOMERS = "customers"  # footfall / transactions


class JobStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetStatus(StrEnum):
    UPLOADED = "uploaded"
    MAPPING_REQUIRED = "mapping_required"
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ModelFamily(StrEnum):
    SEASONAL_NAIVE = "seasonal_naive"
    LIGHTGBM = "lightgbm"
    TORCH_MLP = "torch_mlp"


class RunType(StrEnum):
    TRAIN = "train"
    TUNE = "tune"


class ForecastKind(StrEnum):
    MULTI_HORIZON = "multi_horizon"
    EVENT = "event"


ROLE_RANK: dict[UserRole, int] = {
    UserRole.VIEWER: 0,
    UserRole.ANALYST: 1,
    UserRole.ADMIN: 2,
    UserRole.OWNER: 3,
}
