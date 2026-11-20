"""Import every model here so Alembic autogenerate sees the full metadata."""

from app.db.base import Base
from app.models.dataset import Dataset
from app.models.runs import AutopilotRun, ForecastRun, ModelRun
from app.models.tenancy import FinancialStaffingConfig, Organization, Store, User

__all__ = [
    "AutopilotRun",
    "Base",
    "Dataset",
    "FinancialStaffingConfig",
    "ForecastRun",
    "ModelRun",
    "Organization",
    "Store",
    "User",
]
