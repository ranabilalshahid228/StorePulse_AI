"""Upload profiling, schema-mapping suggestion and raw -> canonical Parquet conversion.

Canonical Parquet schema (one row per series per day):

    ds          date32       observation date
    store_code  string       store identifier ("__all__" when the file has no store column)
    item_id     string|null  item identifier (null for store-level data)
    y           float64      target
    customers   float64|null optional footfall column
    <known_future_columns>   float64
    <categorical_columns>    string
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.core.exceptions import DomainValidationError
from app.schemas.dataset import SchemaMapping

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]
ALL_STORES = "__all__"
PROFILE_ROWS = 5000

_DATE_HINTS = ("date", "ds", "day", "timestamp")
_TARGET_HINTS = ("sales", "revenue", "units", "qty", "quantity", "demand", "y", "customers", "turnover")
_STORE_HINTS = ("store", "store_id", "shop", "location", "branch")
_ITEM_HINTS = ("item", "item_id", "sku", "product", "product_id")
_PROMO_HINTS = ("promo", "promotion", "on_promo")
_OPEN_HINTS = ("open", "is_open")


def _read_sample(path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "parquet":
        pf = pq.ParquetFile(path)
        return pf.read_row_group(0).to_pandas().head(PROFILE_ROWS)
    return pd.read_csv(path, nrows=PROFILE_ROWS, low_memory=False)


def _infer_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        nunique = s.dropna().nunique()
        if nunique <= 2 and set(s.dropna().unique()).issubset({0, 1}):
            return "binary"
        return "integer" if pd.api.types.is_integer_dtype(s) else "float"
    non_null = s.dropna().astype(str)
    if len(non_null) and pd.to_datetime(non_null.head(200), errors="coerce").notna().mean() > 0.95:
        return "date"
    return "categorical" if non_null.nunique() <= max(50, len(non_null) // 20) else "text"


def _first_match(columns: list[str], hints: tuple[str, ...], allowed: set[str] | None = None) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for h in hints:
        if h in lowered and (allowed is None or lowered[h] in allowed):
            return lowered[h]
    for h in hints:
        for low, orig in lowered.items():
            if h in low and (allowed is None or orig in allowed):
                return orig
    return None


def profile_upload(path: Path, file_format: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (column_profile, suggested_mapping) from a sample of the uploaded file."""
    try:
        df = _read_sample(path, file_format)
    except Exception as exc:  # noqa: BLE001 - surface parser errors to the user
        raise DomainValidationError(f"Could not parse file: {exc}") from exc
    if df.empty:
        raise DomainValidationError("Uploaded file contains no rows")

    columns: list[dict[str, Any]] = []
    types: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        t = _infer_type(s)
        types[str(col)] = t
        sample = s.dropna().head(5).tolist()
        columns.append(
            {
                "name": str(col),
                "inferred_type": t,
                "null_fraction": round(float(s.isna().mean()), 4),
                "sample_values": [v.item() if isinstance(v, np.generic) else v for v in sample],
            }
        )
    profile = {"sample_rows": len(df), "columns": columns}

    names = [str(c) for c in df.columns]
    wide_cols = [c for c in names if re.fullmatch(r"d_\d+", c)]
    if len(wide_cols) > 30:
        suggestion: dict[str, Any] = {
            "layout": "wide",
            "wide_value_prefix": "d_",
            "wide_start_date": None,  # must be confirmed by the user (e.g. M5: 2011-01-29)
            "store_column": _first_match(names, _STORE_HINTS),
            "item_column": _first_match(names, _ITEM_HINTS),
            "id_columns": [c for c in names if c not in wide_cols][:6],
        }
        return profile, suggestion

    numeric = {c for c, t in types.items() if t in ("integer", "float")}
    binary = [c for c, t in types.items() if t in ("binary", "boolean")]
    date_col = _first_match(names, _DATE_HINTS, {c for c, t in types.items() if t == "date"})
    target_col = _first_match(names, _TARGET_HINTS, numeric)
    customers_col = _first_match(names, ("customers", "footfall", "transactions"), numeric)
    if customers_col == target_col:
        customers_col = None
    store_col = _first_match(names, _STORE_HINTS)
    item_col = _first_match(names, _ITEM_HINTS)
    used = {date_col, target_col, customers_col, store_col, item_col}
    known_future = [c for c in binary if c not in used]
    categorical = [
        c for c, t in types.items() if t == "categorical" and c not in used and c not in known_future
    ]
    suggestion = {
        "layout": "long",
        "date_column": date_col,
        "target_column": target_col,
        "store_column": store_col,
        "item_column": item_col,
        "customers_column": customers_col,
        "known_future_columns": known_future,
        "categorical_columns": categorical,
        "promo_column": _first_match(known_future, _PROMO_HINTS) if known_future else None,
        "open_column": _first_match(known_future, _OPEN_HINTS) if known_future else None,
        "frequency": "D",
    }
    return profile, suggestion


def _canonical_schema(mapping: SchemaMapping) -> pa.Schema:
    fields = [
        pa.field("ds", pa.date32()),
        pa.field("store_code", pa.string()),
        pa.field("item_id", pa.string()),
        pa.field("y", pa.float64()),
    ]
    if mapping.customers_column:
        fields.append(pa.field("customers", pa.float64()))
    fields += [pa.field(c, pa.float64()) for c in mapping.known_future_columns]
    fields += [pa.field(c, pa.string()) for c in mapping.categorical_columns]
    return pa.schema(fields)


def _transform_long(chunk: pd.DataFrame, m: SchemaMapping) -> pd.DataFrame:
    assert m.date_column and m.target_column
    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(chunk[m.date_column], format=m.date_format, errors="coerce").dt.date,
            "store_code": chunk[m.store_column].astype(str) if m.store_column else ALL_STORES,
            "item_id": chunk[m.item_column].astype(str) if m.item_column else None,
            "y": pd.to_numeric(chunk[m.target_column], errors="coerce").astype("float64"),
        }
    )
    if m.customers_column:
        out["customers"] = pd.to_numeric(chunk[m.customers_column], errors="coerce").astype("float64")
    for c in m.known_future_columns:
        col = chunk[c]
        if col.dtype == object:
            col = col.map({"True": 1, "False": 0, "true": 1, "false": 0}).fillna(col)
        out[c] = pd.to_numeric(col, errors="coerce").astype("float64")
    for c in m.categorical_columns:
        out[c] = chunk[c].astype("string").str.strip()
    return out[out["ds"].notna()]


def _transform_wide(chunk: pd.DataFrame, m: SchemaMapping) -> pd.DataFrame:
    assert m.wide_value_prefix and m.wide_start_date
    value_cols = [c for c in chunk.columns if str(c).startswith(m.wide_value_prefix)]
    id_cols = [c for c in {m.store_column, m.item_column, *m.id_columns} if c]
    long = chunk[id_cols + value_cols].melt(id_vars=id_cols, var_name="_d", value_name="y")
    offsets = long["_d"].str.slice(len(m.wide_value_prefix)).astype(int) - 1
    start = pd.Timestamp(m.wide_start_date)
    out = pd.DataFrame(
        {
            "ds": (start + pd.to_timedelta(offsets, unit="D")).dt.date,
            "store_code": long[m.store_column].astype(str) if m.store_column else ALL_STORES,
            "item_id": long[m.item_column].astype(str) if m.item_column else None,
            "y": pd.to_numeric(long["y"], errors="coerce").astype("float64"),
        }
    )
    return out


def convert_to_parquet(
    raw_path: Path,
    file_format: str,
    mapping: SchemaMapping,
    out_path: Path,
    chunk_rows: int,
    progress: ProgressFn,
) -> dict[str, Any]:
    """Stream the raw file into canonical Parquet. Returns summary stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _canonical_schema(mapping)
    total_bytes = max(raw_path.stat().st_size, 1)
    rows = 0
    dropped = 0
    ds_min = ds_max = None
    series: set[tuple[str, str | None]] = set()
    stores: set[str] = set()

    usecols = None
    if mapping.layout == "long":
        usecols = mapping.referenced_columns()

    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    try:
        if file_format == "parquet":
            pf = pq.ParquetFile(raw_path)
            n_groups = pf.num_row_groups
            iterator = (
                (pf.read_row_group(i, columns=usecols).to_pandas(), (i + 1) / n_groups)
                for i in range(n_groups)
            )
        else:
            fh = raw_path.open("rb")
            reader = pd.read_csv(fh, chunksize=chunk_rows, usecols=usecols, low_memory=False)
            iterator = ((chunk, fh.tell() / total_bytes) for chunk in reader)

        missing_checked = False
        for chunk, frac in iterator:
            if not missing_checked and usecols:
                missing = sorted(set(usecols) - set(chunk.columns))
                if missing:
                    raise DomainValidationError(
                        "Mapped columns not found in file", details={"missing": missing}
                    )
                missing_checked = True
            before = len(chunk)
            df = (
                _transform_long(chunk, mapping)
                if mapping.layout == "long"
                else _transform_wide(chunk, mapping)
            )
            dropped += before - len(df) if mapping.layout == "long" else 0
            if df.empty:
                continue
            rows += len(df)
            cmin, cmax = df["ds"].min(), df["ds"].max()
            ds_min = cmin if ds_min is None else min(ds_min, cmin)
            ds_max = cmax if ds_max is None else max(ds_max, cmax)
            keys = df[["store_code", "item_id"]].drop_duplicates()
            series.update(zip(keys["store_code"], keys["item_id"], strict=True))
            stores.update(keys["store_code"].unique().tolist())
            writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
            progress(min(frac, 0.99), "converting")
    finally:
        writer.close()
        if file_format != "parquet":
            fh.close()

    if rows == 0:
        raise DomainValidationError("No valid rows after applying schema mapping (check date format)")
    return {
        "row_count": rows,
        "dropped_rows": dropped,
        "series_count": len(series),
        "store_codes": sorted(stores),
        "date_min": ds_min,
        "date_max": ds_max,
    }


def infer_frequency(ds_min: Any, ds_max: Any, rows_per_series: float) -> str:
    if ds_min is None or ds_max is None:
        return "D"
    span = (ds_max - ds_min) + timedelta(days=1)
    return "W" if rows_per_series and span.days / rows_per_series > 5 else "D"
