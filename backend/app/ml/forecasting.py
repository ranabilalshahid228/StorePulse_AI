"""Batch inference: multi-horizon and event-scenario forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.data import load_history
from app.ml.families import ForecastModel
from app.ml.features import SERIES_KEYS, build_features, complete_calendar, future_frame
from app.workers.progress import ProgressReporter

SUMMARY_BUCKETS = (7, 14, 28)


def _apply_events(frame: pd.DataFrame, events: list[dict[str, Any]], allowed: set[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["event"] = None
    for ev in events:
        unknown = set(ev["overrides"]) - allowed
        if unknown:
            raise ValueError(f"Event '{ev['name']}' overrides non-covariate columns: {sorted(unknown)}")
        mask = (frame["ds"] >= pd.Timestamp(ev["start_date"])) & (frame["ds"] <= pd.Timestamp(ev["end_date"]))
        if ev.get("store_codes"):
            mask &= frame["store_code"].isin(ev["store_codes"])
        for col, value in ev["overrides"].items():
            frame.loc[mask, col] = float(value)
        frame.loc[mask, "event"] = ev["name"]
    return frame


def _predict(model: ForecastModel, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    spec = model.spec
    # Tag future rows with their position so predictions map back without a (dtype-unsafe) merge.
    hist = history.assign(_row=-1)
    fut_in = future.drop(columns=["event"], errors="ignore").assign(_row=np.arange(len(future)))
    combined = pd.concat([hist, fut_in], ignore_index=True)
    combined = combined.sort_values([*SERIES_KEYS, "ds"], ignore_index=True)
    feats = build_features(combined, spec, fit_categories=False)
    fut = feats[feats["_row"] >= 0]
    preds = np.clip(model.predict(fut[spec.feature_columns]), 0, None)
    out = np.full(len(future), np.nan)
    out[fut["_row"].to_numpy(dtype="int64")] = preds
    return out


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = model.spec
    if horizon > spec.horizon:
        raise ValueError(f"Requested horizon {horizon} exceeds the model's trained horizon {spec.horizon}")

    progress.update(0.05, "loading_history", force=True)
    history = complete_calendar(load_history(parquet_path, store_code))
    # Only the trailing window is needed for lag/rolling features.
    keep_days = spec.lag_offset + max(49, 28) + 7
    history = history[history["ds"] > history["ds"].max() - pd.Timedelta(days=keep_days + horizon)]

    progress.update(0.25, "building_future_frame", force=True)
    future = future_frame(history, spec, horizon, covariate_defaults)
    scenario = _apply_events(future, events, set(spec.known_future)) if events else future.assign(event=None)

    progress.update(0.45, "predicting", force=True)
    yhat = _predict(model, history, scenario)
    out = scenario[["ds", "store_code", "item_id", "event"]].copy()
    out["yhat"] = yhat
    interval = metrics.get("interval")
    if interval:
        out["yhat_lower"] = np.clip(yhat * (1 + interval["lower"]), 0, None)
        out["yhat_upper"] = yhat * (1 + interval["upper"])
    else:
        out["yhat_lower"] = np.nan
        out["yhat_upper"] = np.nan
    out["is_promo"] = scenario[promo_column].fillna(0).astype(float) > 0 if promo_column else False
    out["is_open"] = scenario[open_column].fillna(1).astype(float) > 0 if open_column else True

    if events and compare_to_baseline:
        progress.update(0.7, "predicting_baseline", force=True)
        out["baseline_yhat"] = _predict(model, history, future.assign(event=None))
    else:
        out["baseline_yhat"] = np.nan

    progress.update(0.9, "summarizing", force=True)
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
