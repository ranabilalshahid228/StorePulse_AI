"""Auxiliary-table joins shared by ingestion (history) and forecasting (future dates).

Auxiliary datasets are raw uploads (role="auxiliary") such as M5 `calendar.csv` / `sell_prices.csv`
or a store master file. They are loaded once, keyed, and left-joined chunk by chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.exceptions import DomainValidationError
from app.schemas.dataset import JoinSpec


@dataclass
class PreparedJoin:
    spec: JoinSpec
    table: pd.DataFrame  # indexed by the (string-normalized) join keys

    @property
    def date_keys(self) -> list[bool]:
        return [left == "ds" for left in self.spec.left_on]


def _key(series: pd.Series, is_date: bool) -> pd.Series:
    if is_date:
        return pd.to_datetime(series, errors="coerce").dt.normalize()
    # Normalize ints/floats/strings to one string form ("13" == 13 == 13.0).
    num = pd.to_numeric(series, errors="coerce")
    as_str = series.astype("string").str.strip()
    is_int = num.notna() & (num % 1 == 0)
    return as_str.where(~is_int, num.astype("Int64").astype("string"))


def load_join(spec: JoinSpec, path: Path, file_format: str) -> PreparedJoin:
    usecols = list(dict.fromkeys([*spec.right_on, *spec.columns]))
    try:
        if file_format == "parquet":
            raw = pd.read_parquet(path, columns=usecols)
        else:
            raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    except ValueError as exc:
        raise DomainValidationError(
            f"Auxiliary dataset {spec.dataset_id} is missing join columns", details={"error": str(exc)}
        ) from exc
    keys = [f"__k{i}" for i in range(len(spec.right_on))]
    for k, right, left in zip(keys, spec.right_on, spec.left_on, strict=True):
        raw[k] = _key(raw[right], left == "ds")
    table = raw.drop_duplicates(subset=keys, keep="last").set_index(keys)[spec.columns]
    return PreparedJoin(spec=spec, table=table)


def apply_joins(df: pd.DataFrame, joins: list[PreparedJoin]) -> pd.DataFrame:
    """Left-join every prepared table onto `df` in order. Joined columns overwrite any existing
    column of the same name (a missing match is a real "no value", e.g. no event that day)."""
    for j in joins:
        keys = [f"__k{i}" for i in range(len(j.spec.left_on))]
        left = pd.DataFrame(
            {
                k: _key(df[col], is_date)
                for k, col, is_date in zip(keys, j.spec.left_on, j.date_keys, strict=True)
            },
            index=df.index,
        )
        joined = left.join(j.table, on=keys)
        for c in j.spec.columns:
            df[c] = joined[c].to_numpy()
    return df
