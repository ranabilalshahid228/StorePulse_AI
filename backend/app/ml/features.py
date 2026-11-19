"""Feature engineering shared by training and inference.

Strategy: a single global model per run with a *direct* multi-horizon setup. All lag features are
offset by `lag_offset = ceil(horizon / 7) * 7` days, so every feature for a date up to
`horizon` days in the future is computable from observed history alone (no recursion, no leakage),
and lags stay weekday-aligned.

Everything here is vectorized over all series at once (integer group ids + groupby/rolling), so a
single M5 store (3,049 items x ~1,900 days ~ 5.8M rows) builds in seconds, not minutes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
STATIC_NUMERIC = [
    "st_competition_distance",
    "st_competition_open_months",
    "st_promo2_active",
    "st_promo2_weeks",
]
STATIC_CATEGORICAL = ["st_store_type", "st_assortment"]
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
    )
}
_MONTHS["Sept"] = 9


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
    price_column: str | None = None
    # store_code -> Store attributes (store_type, assortment, competition_*, promo2_*), frozen at train time
    store_static: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, Any], horizon: int, store_static: dict[str, dict[str, Any]] | None = None
    ) -> FeatureSpec:
        return cls(
            horizon=horizon,
            lag_offset=lag_offset_for(horizon),
            known_future=list(mapping.get("known_future_columns") or []),
            categorical=list(mapping.get("categorical_columns") or []),
            price_column=mapping.get("price_column"),
            store_static=store_static or {},
        )

    @property
    def lag_columns(self) -> list[str]:
        return [f"lag_{self.lag_offset + k}" for k in LAG_STEPS]

    @property
    def static_categorical(self) -> list[str]:
        return [c for c in STATIC_CATEGORICAL if self._has_static(c)]

    @property
    def static_numeric(self) -> list[str]:
        return [c for c in STATIC_NUMERIC if self._has_static(c)]

    @property
    def categorical_features(self) -> list[str]:
        return ["store_code", "item_id", *self.categorical, *self.static_categorical]

    def _has_static(self, col: str) -> bool:
        source = {
            "st_store_type": "store_type",
            "st_assortment": "assortment",
            "st_competition_distance": "competition_distance",
            "st_competition_open_months": "competition_open_since_year",
            "st_promo2_active": "promo2",
            "st_promo2_weeks": "promo2_since_year",
        }[col]
        return any(v.get(source) not in (None, False, "") for v in self.store_static.values())

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ----------------------------------------------------------------------------------------
# Calendar completion
# ----------------------------------------------------------------------------------------


def complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each series to a contiguous daily calendar (missing days -> NaN rows), sorted."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"]).astype("datetime64[ns]")
    df["item_id"] = df["item_id"].fillna("").astype(str)
    df["store_code"] = df["store_code"].astype(str)
    df = df.drop_duplicates(subset=[*SERIES_KEYS, "ds"], keep="last")
    df = df.sort_values([*SERIES_KEYS, "ds"], ignore_index=True)

    bounds = df.groupby(SERIES_KEYS, sort=False)["ds"].agg(["min", "max", "size"])
    lengths = ((bounds["max"] - bounds["min"]).dt.days + 1).to_numpy()
    if int(lengths.sum()) == len(df):  # already contiguous (e.g. M5): fast path
        return df

    starts = np.repeat(bounds["min"].to_numpy(), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame(
        {
            "store_code": np.repeat(bounds.index.get_level_values(0).to_numpy(), lengths),
            "item_id": np.repeat(bounds.index.get_level_values(1).to_numpy(), lengths),
            "ds": starts + offsets.astype("timedelta64[D]"),
        }
    )
    return grid.merge(df, on=[*SERIES_KEYS, "ds"], how="left")


# ----------------------------------------------------------------------------------------
# Feature building
# ----------------------------------------------------------------------------------------


def _calendar(df: pd.DataFrame) -> None:
    ds = df["ds"].dt
    df["dow"] = ds.day_of_week.astype("int8")
    df["dom"] = ds.day.astype("int8")
    df["month"] = ds.month.astype("int8")
    df["weekofyear"] = ds.isocalendar().week.astype("int16").to_numpy()
    df["dayofyear"] = ds.day_of_year.astype("int16")
    df["year"] = ds.year.astype("int16")
    df["is_month_start"] = ds.is_month_start.astype("int8")
    df["is_month_end"] = ds.is_month_end.astype("int8")
    df["is_weekend"] = (ds.day_of_week >= 5).astype("int8")


def _store_static(out: pd.DataFrame, spec: FeatureSpec) -> None:
    """Rossmann-style store metadata -> static and date-dependent store features."""
    if not spec.store_static:
        return
    attrs = pd.DataFrame.from_dict(spec.store_static, orient="index")
    a = attrs.reindex(out["store_code"].astype(str).to_numpy())

    def col(name: str) -> pd.Series:
        return pd.Series(a[name].to_numpy() if name in a else np.nan, index=out.index)

    if "st_store_type" in spec.static_categorical:
        out["st_store_type"] = col("store_type")
    if "st_assortment" in spec.static_categorical:
        out["st_assortment"] = col("assortment")
    if "st_competition_distance" in spec.static_numeric:
        out["st_competition_distance"] = pd.to_numeric(col("competition_distance"), errors="coerce").astype(
            "float32"
        )
    cur_ym = out["ds"].dt.year * 12 + out["ds"].dt.month
    if "st_competition_open_months" in spec.static_numeric:
        comp_ym = pd.to_numeric(col("competition_open_since_year"), errors="coerce") * 12 + pd.to_numeric(
            col("competition_open_since_month"), errors="coerce"
        ).fillna(1)
        out["st_competition_open_months"] = (cur_ym - comp_ym).astype("float32")
    if "st_promo2_active" in spec.static_numeric:
        promo2 = col("promo2").fillna(False).astype(bool)
        year = pd.to_numeric(col("promo2_since_year"), errors="coerce")
        week = pd.to_numeric(col("promo2_since_week"), errors="coerce")
        since_str = (
            year.astype("Int64").astype("string")
            + "-W"
            + week.astype("Int64").astype("string").str.zfill(2)
            + "-1"
        )
        since = pd.to_datetime(since_str, format="%G-W%V-%u", errors="coerce")
        masks = {
            code: sum(
                1 << (_MONTHS[m.strip()] - 1)
                for m in str(v.get("promo_interval") or "").split(",")
                if m.strip() in _MONTHS
            )
            for code, v in spec.store_static.items()
        }
        mask = out["store_code"].astype(str).map(masks).fillna(0).astype("int64")
        in_month = (mask.to_numpy() >> (out["ds"].dt.month.to_numpy() - 1)) & 1
        started = (out["ds"] >= since).fillna(False).to_numpy()
        out["st_promo2_active"] = (promo2.to_numpy() & started & (in_month == 1)).astype("float32")
        if "st_promo2_weeks" in spec.static_numeric:
            weeks = ((out["ds"] - since).dt.days / 7).where(promo2.to_numpy() & started)
            out["st_promo2_weeks"] = weeks.astype("float32")


def build_features(df: pd.DataFrame, spec: FeatureSpec, *, fit_categories: bool) -> pd.DataFrame:
    """`df` must be calendar-complete and sorted by series then date (see `complete_calendar`)."""
    out = df.copy()
    _calendar(out)
    o = spec.lag_offset
    gid = out.groupby(SERIES_KEYS, sort=False).ngroup().to_numpy()
    y = out["y"].astype("float32")
    shifted_by = y.groupby(gid, sort=False)
    for k in LAG_STEPS:
        out[f"lag_{o + k}"] = shifted_by.shift(o + k)
    shifted = shifted_by.shift(o)
    roll = shifted.groupby(gid, sort=False)
    dyn = []
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = (
            roll.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        )
        dyn.append(f"roll_mean_{w}")
    out["roll_std_28"] = (
        roll.rolling(28, min_periods=7).std().reset_index(level=0, drop=True).astype("float32")
    )
    zero = (shifted == 0).astype("float32").where(shifted.notna())
    out["roll_zero_share_28"] = (
        zero.groupby(gid, sort=False)
        .rolling(28, min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    # Same-weekday mean of the last 4 aligned weeks.
    out["dow_mean_4w"] = out[[f"lag_{o}", f"lag_{o + 7}", f"lag_{o + 14}", f"lag_{o + 21}"]].mean(axis=1)
    dyn += ["roll_std_28", "roll_zero_share_28", "dow_mean_4w"]

    for c in spec.known_future:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    price_cols: list[str] = []
    if spec.price_column and spec.price_column in out:
        p = out[spec.price_column]
        by = p.groupby(gid, sort=False)
        # Relative to the trailing 28-day average price (window-local, so identical at train and inference).
        trailing = (
            by.shift(1)
            .groupby(gid, sort=False)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["price_rel_mean"] = (p / trailing).astype("float32")
        out["price_change_7"] = (p / by.shift(7) - 1).astype("float32")
        price_cols = ["price_rel_mean", "price_change_7"]

    _store_static(out, spec)

    for c in spec.categorical_features:
        values = out[c].astype("string").fillna("")
        if fit_categories:
            spec.category_levels[c] = sorted(values.unique().tolist())
        out[c] = pd.Categorical(values, categories=spec.category_levels.get(c, []))

    feature_cols = [
        *CALENDAR_FEATURES,
        *spec.lag_columns,
        *dyn,
        *spec.known_future,
        *price_cols,
        *spec.static_numeric,
        *spec.categorical_features,
    ]
    if fit_categories:
        spec.feature_columns = feature_cols
    return out


def training_mask(features: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Rows usable for training: target observed and at least the nearest lag available."""
    return features["y"].notna() & features[f"lag_{spec.lag_offset}"].notna()


# ----------------------------------------------------------------------------------------
# Future frame
# ----------------------------------------------------------------------------------------


def future_frame(
    history: pd.DataFrame,
    spec: FeatureSpec,
    horizon: int,
    covariate_defaults: dict[str, float],
    enrich: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Rows for the next `horizon` days of every series.

    Known-future covariates are resolved in priority order:
      1. `enrich` - re-applies the dataset's auxiliary joins, giving *real* future values
         (calendar events, planned prices); joined columns overwrite;
      2. `covariate_defaults` supplied with the request;
      3. the series' last observed value for the same weekday (last 28 days).
    Categorical covariates carry forward the series' last observed value (before `enrich`).
    Scenario overrides are applied by the caller afterwards.
    """
    last = history.groupby(SERIES_KEYS, sort=False)["ds"].max()
    n = len(last)
    steps = np.tile(np.arange(1, horizon + 1), n).astype("timedelta64[D]")
    f = pd.DataFrame(
        {
            "store_code": np.repeat(last.index.get_level_values(0).to_numpy(), horizon),
            "item_id": np.repeat(last.index.get_level_values(1).to_numpy(), horizon),
            "ds": np.repeat(last.to_numpy(), horizon) + steps,
            "y": np.nan,
        }
    )
    for c in spec.known_future:
        f[c] = np.nan
    # Static/id categoricals first: joins may key on them (e.g. M5 state_id -> SNAP flags).
    for c in spec.categorical:
        carry = history.dropna(subset=[c]).groupby(SERIES_KEYS)[c].last().rename(c)
        f = f.merge(carry.reset_index(), on=SERIES_KEYS, how="left")
    if enrich is not None:
        f = enrich(f)

    for c in spec.known_future:
        if c in covariate_defaults:
            f[c] = f[c].fillna(float(covariate_defaults[c]))
    pending = [c for c in spec.known_future if f[c].isna().any()]
    if pending:
        last_ds = history[SERIES_KEYS].merge(last.rename("last_ds").reset_index(), on=SERIES_KEYS)["last_ds"]
        tail = history[(history["ds"] > last_ds.to_numpy() - np.timedelta64(28, "D"))].copy()
        tail["dow"] = tail["ds"].dt.day_of_week
        f["dow"] = f["ds"].dt.day_of_week
        for c in pending:
            by_dow = tail.dropna(subset=[c]).groupby([*SERIES_KEYS, "dow"])[c].last().rename(f"__{c}")
            f = f.merge(by_dow.reset_index(), on=[*SERIES_KEYS, "dow"], how="left")
            f[c] = f[c].fillna(f.pop(f"__{c}"))
        f = f.drop(columns="dow")
    return f.reset_index(drop=True)
