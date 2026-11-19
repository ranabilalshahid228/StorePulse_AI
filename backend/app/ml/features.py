"""Feature engineering shared by training and inference.

Strategy: a single global model per run with a *direct* multi-horizon setup. All lag features are
offset by `lag_offset = ceil(horizon / 7) * 7` days, so every feature for a date up to
`horizon` days in the future is computable from observed history alone (no recursion, no leakage),
and lags stay weekday-aligned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SERIES_KEYS = ["store_code", "item_id"]
LAG_STEPS = (0, 7, 14, 21, 49)
ROLL_WINDOWS = (7, 28)
CALENDAR_FEATURES = [
    "dow",
    "dom",
    "month",
    "weekofyear",
    "dayofyear",
    "year",
    "is_month_start",
    "is_month_end",
    "is_weekend",
]


def lag_offset_for(horizon: int) -> int:
    return int(math.ceil(horizon / 7) * 7)


@dataclass
class FeatureSpec:
    horizon: int
    lag_offset: int
    known_future: list[str]
    categorical: list[str]
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any], horizon: int) -> FeatureSpec:
        return cls(
            horizon=horizon,
            lag_offset=lag_offset_for(horizon),
            known_future=list(mapping.get("known_future_columns") or []),
            categorical=list(mapping.get("categorical_columns") or []),
        )

    @property
    def lag_columns(self) -> list[str]:
        return [f"lag_{self.lag_offset + k}" for k in LAG_STEPS]

    @property
    def categorical_features(self) -> list[str]:
        return ["store_code", "item_id", *self.categorical]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each series to a contiguous daily calendar (missing days -> NaN rows)."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["item_id"] = df["item_id"].fillna("")
    df = df.drop_duplicates(subset=[*SERIES_KEYS, "ds"], keep="last")
    frames = []
    for (store, item), g in df.groupby(SERIES_KEYS, sort=False):
        idx = pd.date_range(g["ds"].min(), g["ds"].max(), freq="D")
        g = g.set_index("ds").reindex(idx)
        g.index.name = "ds"
        g["store_code"] = store
        g["item_id"] = item
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values([*SERIES_KEYS, "ds"], ignore_index=True)


def _calendar(df: pd.DataFrame) -> None:
    ds = df["ds"].dt
    df["dow"] = ds.day_of_week
    df["dom"] = ds.day
    df["month"] = ds.month
    df["weekofyear"] = ds.isocalendar().week.astype("int32").to_numpy()
    df["dayofyear"] = ds.day_of_year
    df["year"] = ds.year
    df["is_month_start"] = ds.is_month_start.astype("int8")
    df["is_month_end"] = ds.is_month_end.astype("int8")
    df["is_weekend"] = (ds.day_of_week >= 5).astype("int8")


def build_features(df: pd.DataFrame, spec: FeatureSpec, *, fit_categories: bool) -> pd.DataFrame:
    """`df` must be calendar-complete and sorted by series then date."""
    out = df.copy()
    _calendar(out)
    o = spec.lag_offset
    grouped = out.groupby(SERIES_KEYS, sort=False)["y"]
    for k in LAG_STEPS:
        out[f"lag_{o + k}"] = grouped.shift(o + k)
    shifted = grouped.shift(o)
    keys = [out[k] for k in SERIES_KEYS]
    roll_cols = []
    for w in ROLL_WINDOWS:
        col = f"roll_mean_{w}"
        out[col] = shifted.groupby(keys).transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        roll_cols.append(col)
    out["roll_std_28"] = shifted.groupby(keys).transform(lambda s: s.rolling(28, min_periods=7).std())
    # Same-weekday mean of the last 4 aligned weeks.
    out["dow_mean_4w"] = out[[f"lag_{o}", f"lag_{o + 7}", f"lag_{o + 14}", f"lag_{o + 21}"]].mean(axis=1)

    for c in spec.known_future:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    for c in spec.categorical_features:
        values = out[c].astype("string").fillna("")
        if fit_categories:
            spec.category_levels[c] = sorted(values.unique().tolist())
        out[c] = pd.Categorical(values, categories=spec.category_levels.get(c, []))

    feature_cols = [
        *CALENDAR_FEATURES,
        *spec.lag_columns,
        *roll_cols,
        "roll_std_28",
        "dow_mean_4w",
        *spec.known_future,
        *spec.categorical_features,
    ]
    if fit_categories:
        spec.feature_columns = feature_cols
    return out


def training_mask(features: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Rows usable for training: target observed and at least the nearest lag available."""
    return features["y"].notna() & features[f"lag_{spec.lag_offset}"].notna()


def future_frame(
    history: pd.DataFrame,
    spec: FeatureSpec,
    horizon: int,
    covariate_defaults: dict[str, float],
) -> pd.DataFrame:
    """Create rows for the next `horizon` days of every series.

    Known-future covariates default to (1) `covariate_defaults` if provided, else (2) the last
    observed value for the same weekday of that series. Categorical covariates carry forward the
    last observed value. Scenario overrides are applied by the caller afterwards.
    """
    rows = []
    for (store, item), g in history.groupby(SERIES_KEYS, sort=False):
        last = g["ds"].max()
        dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
        f = pd.DataFrame({"ds": dates, "store_code": store, "item_id": item, "y": np.nan})
        tail = g[g["ds"] > last - pd.Timedelta(days=28)]
        for c in spec.known_future:
            if c in covariate_defaults:
                f[c] = float(covariate_defaults[c])
            else:
                by_dow = tail.dropna(subset=[c]).groupby(tail["ds"].dt.day_of_week)[c].last()
                f[c] = f["ds"].dt.day_of_week.map(by_dow).astype("float64")
        for c in spec.categorical:
            non_null = g[c].dropna()
            f[c] = non_null.iloc[-1] if len(non_null) else ""
        rows.append(f)
    return pd.concat(rows, ignore_index=True)
