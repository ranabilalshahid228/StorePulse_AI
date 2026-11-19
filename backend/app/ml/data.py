"""Loading canonical dataset Parquet into training/inference frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

from app.core.exceptions import DomainValidationError


def load_history(parquet_path: Path, store_code: str | None = None) -> pd.DataFrame:
    ds = pads.dataset(parquet_path, format="parquet")
    filt = pads.field("store_code") == store_code if store_code else None
    df = ds.to_table(filter=filt).to_pandas()
    if df.empty:
        raise DomainValidationError("No rows for the requested scope", details={"store_code": store_code})
    df["ds"] = pd.to_datetime(df["ds"])
    df["item_id"] = df["item_id"].fillna("")
    return df
