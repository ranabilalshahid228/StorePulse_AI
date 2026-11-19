from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-that-is-at-least-32-characters")
os.environ.setdefault("ENVIRONMENT", "test")

REPO_ROOT = Path(__file__).resolve().parents[2]
ROSSMANN_TRAIN = REPO_ROOT / "data" / "raw" / "rossman dataset" / "train.csv"


class RecordingProgress:
    """Drop-in for ProgressReporter that records updates instead of writing to PostgreSQL."""

    def __init__(self) -> None:
        self.events: list[tuple[float, str]] = []

    def update(self, fraction: float, stage: str, message: str | None = None, force: bool = False) -> None:
        self.events.append((fraction, stage))

    def sub(self, lo: float, hi: float) -> RecordingProgress:
        return self


@pytest.fixture
def progress() -> RecordingProgress:
    return RecordingProgress()


@pytest.fixture(scope="session")
def rossmann_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """First 3 Rossmann stores, written to a small CSV (skips if the raw data isn't present)."""
    if not ROSSMANN_TRAIN.exists():
        pytest.skip("Rossmann train.csv not available under data/raw")
    import pandas as pd

    df = pd.read_csv(ROSSMANN_TRAIN, dtype={"StateHoliday": str}, low_memory=False)
    sub = df[df["Store"].isin([1, 2, 3])]
    out = tmp_path_factory.mktemp("raw") / "rossmann_subset.csv"
    sub.to_csv(out, index=False)
    return out


@pytest.fixture
def org_id() -> uuid.UUID:
    return uuid.uuid4()


def rossmann_mapping() -> dict[str, Any]:
    return {
        "layout": "long",
        "date_column": "Date",
        "target_column": "Sales",
        "store_column": "Store",
        "customers_column": "Customers",
        "known_future_columns": ["Open", "Promo", "SchoolHoliday"],
        "categorical_columns": ["StateHoliday"],
        "promo_column": "Promo",
        "open_column": "Open",
    }
