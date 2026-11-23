"""Batch inference: multi-horizon and event-scenario forecasts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.data import load_history
from app.ml.families import ARIMAModel, ForecastModel
from app.ml.features import LAG_STEPS, SERIES_KEYS, build_features, complete_calendar, future_frame
from app.workers.progress import ProgressReporter

SUMMARY_BUCKETS = (7, 14, 28)


def _apply_events(
    frame: pd.DataFrame,
    events: list[dict[str, Any]],
    allowed: set[str],
    categorical: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Override covariates on date ranges. Known-future columns take numbers; categorical columns
    (e.g. Rossmann StateHoliday) take their category label."""
    frame = frame.copy()
    allowed = set(allowed) | set(categorical)
    frame["event"] = None
    for ev in events:
        unknown = set(ev["overrides"]) - allowed
        if unknown:
            raise ValueError(f"Event '{ev['name']}' overrides non-covariate columns: {sorted(unknown)}")
        mask = (frame["ds"] >= pd.Timestamp(ev["start_date"])) & (frame["ds"] <= pd.Timestamp(ev["end_date"]))
        if ev.get("store_codes"):
            mask &= frame["store_code"].isin(ev["store_codes"])
        for col, value in ev["overrides"].items():
            if col in categorical:
                if isinstance(frame[col].dtype, pd.CategoricalDtype):
                    frame[col] = frame[col].astype(object)
                frame.loc[mask, col] = str(value)
            else:
                frame.loc[mask, col] = float(value)
        frame.loc[mask, "event"] = ev["name"]
    return frame


def _predict(
    model: ForecastModel, history: pd.DataFrame, future: pd.DataFrame
) -> tuple[np.ndarray, dict[float, np.ndarray] | None]:
    if isinstance(model, ARIMAModel):
        return model.predict_from_history(future, pd.DatetimeIndex(future["ds"]), history), None

    spec = model.spec
    # Tag future rows with their position so predictions map back without a (dtype-unsafe) merge.
    hist = history.assign(_row=-1)
    fut_in = future.drop(columns=["event"], errors="ignore").assign(_row=np.arange(len(future)))
    combined = pd.concat([hist, fut_in], ignore_index=True)
    combined = combined.sort_values([*SERIES_KEYS, "ds"], ignore_index=True)
    feats = build_features(combined, spec, fit_categories=False)
    fut = feats[feats["_row"] >= 0]
    X = fut[spec.model_columns]
    rows = fut["_row"].to_numpy(dtype="int64")
    out = np.full(len(future), np.nan)
    out[rows] = np.clip(model.predict(X), 0, None)
    q = model.predict_quantiles(X)
    quantiles = None
    if q:
        quantiles = {}
        for level, values in q.items():
            arr = np.full(len(future), np.nan)
            arr[rows] = values
            quantiles[level] = arr
    return out, quantiles


def run_forecast(
    *,
    model: ForecastModel,
    metrics: dict[str, Any],
    parquet_path: Path,
    store_code: str | None,
    horizon: int,
    covariate_defaults: dict[str, float],
    events: list[dict[str, Any]],
    compare_to_baseline: bool,
    promo_column: str | None,
    open_column: str | None,
    progress: ProgressReporter,
    price_column: str | None = None,
    scope_store_code: str | None = None,
    enrich: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    include_covariates: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`include_covariates` adds the resolved known-future / categorical inputs (after scenario
    overrides) to the output, e.g. for holiday flags in the Store Autopilot."""
    spec = model.spec
    if horizon > spec.horizon:
        raise ValueError(f"Requested horizon {horizon} exceeds the model's trained horizon {spec.horizon}")

    progress.update(0.05, "loading_history", force=True)
    # Load the model's training scope (group aggregates are computed across it), filter output later.
    history = complete_calendar(load_history(parquet_path, scope_store_code))
    # Only the trailing window is needed for lag/rolling features.
    # Longest look-back: 364-day trailing price max (and 91-day intermittency / seq windows).
    keep_days = spec.lag_offset + max(max(LAG_STEPS), spec.seq_len, 371) + 35
    history = history[history["ds"] > history["ds"].max() - pd.Timedelta(days=keep_days + horizon)]

    progress.update(0.25, "building_future_frame", force=True)
    future = future_frame(history, spec, horizon, covariate_defaults, enrich=enrich)
    scenario = (
        _apply_events(future, events, set(spec.known_future), set(spec.categorical))
        if events
        else future.assign(event=None)
    )

    progress.update(0.45, "predicting", force=True)
    yhat, quantiles = _predict(model, history, scenario)
    out = scenario[["ds", "store_code", "item_id", "event"]].copy()
    out["yhat"] = yhat
    interval = metrics.get("interval")
    out["yhat_p50"] = np.nan
    if quantiles:
        out["yhat_lower"] = quantiles[min(quantiles)]
        out["yhat_upper"] = quantiles[max(quantiles)]
        out["yhat_p50"] = quantiles.get(0.5, np.nan)
    elif interval and "lower" in interval:
        out["yhat_lower"] = np.clip(yhat * (1 + interval["lower"]), 0, None)
        out["yhat_upper"] = yhat * (1 + interval["upper"])
    else:
        out["yhat_lower"] = np.nan
        out["yhat_upper"] = np.nan
    out["is_promo"] = scenario[promo_column].fillna(0).astype(float) > 0 if promo_column else False
    out["price"] = scenario[price_column].astype(float) if price_column else np.nan
    out["is_open"] = scenario[open_column].fillna(1).astype(float) > 0 if open_column else True
    if include_covariates:
        for c in [*spec.known_future, *spec.categorical]:
            if c in scenario and c not in out:
                out[c] = scenario[c].to_numpy()

    if events and compare_to_baseline:
        progress.update(0.7, "predicting_baseline", force=True)
        out["baseline_yhat"] = _predict(model, history, future.assign(event=None))[0]
    else:
        out["baseline_yhat"] = np.nan

    progress.update(0.9, "summarizing", force=True)
    if store_code is not None:
        out = out[out["store_code"] == store_code].reset_index(drop=True)
    out["item_id"] = out["item_id"].replace("", None)
    out["ds"] = out["ds"].dt.date
    return out, summarize(out, horizon)


def summarize(out: pd.DataFrame, horizon: int) -> dict[str, Any]:
    start = min(out["ds"])
    days_ahead = out["ds"].map(lambda d: (d - start).days + 1)
    buckets = {}
    for b in (*[b for b in SUMMARY_BUCKETS if b < horizon], horizon):
        sel = out[days_ahead <= b]
        buckets[f"h{b}"] = {
            "yhat": float(sel["yhat"].sum()),
            "yhat_lower": float(sel["yhat_lower"].sum()) if sel["yhat_lower"].notna().any() else None,
            "yhat_upper": float(sel["yhat_upper"].sum()) if sel["yhat_upper"].notna().any() else None,
        }
    per_store = out.groupby("store_code")["yhat"].sum().sort_values(ascending=False)
    summary: dict[str, Any] = {
        "start_date": start.isoformat(),
        "end_date": max(out["ds"]).isoformat(),
        "series_count": int(out[["store_code", "item_id"]].drop_duplicates().shape[0]),
        "store_count": int(out["store_code"].nunique()),
        "horizon_totals": buckets,
        "top_stores": [{"store_code": k, "yhat": float(v)} for k, v in per_store.head(10).items()],
    }
    if out["baseline_yhat"].notna().any():
        ev = out[out["event"].notna()]
        summary["event_impact"] = {
            name: {
                "yhat": float(g["yhat"].sum()),
                "baseline_yhat": float(g["baseline_yhat"].sum()),
                "uplift": float(g["yhat"].sum() - g["baseline_yhat"].sum()),
                "uplift_pct": float(g["yhat"].sum() / g["baseline_yhat"].sum() - 1)
                if g["baseline_yhat"].sum() > 0
                else None,
            }
            for name, g in ev.groupby("event")
        }
    return summary
