"""Data-health diagnostics computed from the canonical Parquet file."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SERIES_KEYS = ["store_code", "item_id"]
ROBUST_Z_THRESHOLD = 5.0
SHORT_HISTORY_DAYS = 90
STALE_DAYS = 7


def _issue(severity: str, code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"severity": severity, "code": code, "message": message, "details": details}


def compute_health_report(parquet_path: Path) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    df["ds"] = pd.to_datetime(df["ds"])
    df["item_id"] = df["item_id"].fillna("")
    n = len(df)
    issues: list[dict[str, Any]] = []

    # --- Completeness -------------------------------------------------------------------
    null_fraction = {c: round(float(v), 5) for c, v in df.isna().mean().items()}
    if null_fraction.get("y", 0) > 0:
        issues.append(
            _issue(
                "warning" if null_fraction["y"] < 0.05 else "critical",
                "missing_target",
                f"{null_fraction['y']:.2%} of target values are missing",
                fraction=null_fraction["y"],
            )
        )

    # --- Duplicates ---------------------------------------------------------------------
    dup_count = int(df.duplicated(subset=[*SERIES_KEYS, "ds"]).sum())
    if dup_count:
        issues.append(
            _issue(
                "critical", "duplicate_keys", f"{dup_count} duplicate (series, date) rows", count=dup_count
            )
        )

    # --- Value sanity -------------------------------------------------------------------
    y = df["y"]
    negative = int((y < 0).sum())
    zero_fraction = float((y == 0).mean()) if n else 0.0
    if negative:
        issues.append(
            _issue("warning", "negative_target", f"{negative} negative target values", count=negative)
        )
    if zero_fraction > 0.3:
        issues.append(
            _issue(
                "info",
                "high_zero_share",
                f"{zero_fraction:.1%} of target values are zero (closures or intermittent demand)",
                fraction=round(zero_fraction, 4),
            )
        )

    # --- Per-series coverage ------------------------------------------------------------
    g = df.groupby(SERIES_KEYS, sort=False)
    per_series = g.agg(first=("ds", "min"), last=("ds", "max"), observed=("ds", "nunique"))
    per_series["expected"] = (per_series["last"] - per_series["first"]).dt.days + 1
    per_series["missing"] = per_series["expected"] - per_series["observed"]
    global_max = df["ds"].max()
    total_missing = int(per_series["missing"].clip(lower=0).sum())
    gappy = per_series[per_series["missing"] > 0].sort_values("missing", ascending=False)
    if total_missing:
        issues.append(
            _issue(
                "warning",
                "date_gaps",
                f"{total_missing} missing series-days across {len(gappy)} series",
                top_series=[
                    {"store_code": s, "item_id": i or None, "missing_days": int(r.missing)}
                    for (s, i), r in gappy.head(10).iterrows()
                ],
            )
        )
    stale = per_series[per_series["last"] < global_max - pd.Timedelta(days=STALE_DAYS)]
    if len(stale):
        issues.append(
            _issue(
                "warning",
                "stale_series",
                f"{len(stale)} series stop more than {STALE_DAYS} days before the dataset end",
                count=int(len(stale)),
            )
        )
    short = per_series[per_series["observed"] < SHORT_HISTORY_DAYS]
    if len(short):
        issues.append(
            _issue(
                "info",
                "short_history",
                f"{len(short)} series have fewer than {SHORT_HISTORY_DAYS} observations",
                count=int(len(short)),
            )
        )

    # --- Outliers (robust z-score per series, non-zero values only) ---------------------
    nz = df[df["y"] > 0][[*SERIES_KEYS, "y"]]
    outliers = 0
    if not nz.empty:
        med = nz.groupby(SERIES_KEYS)["y"].transform("median")
        mad = (nz["y"] - med).abs().groupby([nz[k] for k in SERIES_KEYS]).transform("median")
        scale = (1.4826 * mad).replace(0, np.nan)
        rz = ((nz["y"] - med) / scale).abs()
        outliers = int((rz > ROBUST_Z_THRESHOLD).sum())
    if outliers:
        issues.append(
            _issue(
                "info",
                "outliers",
                f"{outliers} values exceed a robust z-score of {ROBUST_Z_THRESHOLD}",
                count=outliers,
            )
        )

    # --- Score --------------------------------------------------------------------------
    penalty = {"critical": 25, "warning": 8, "info": 2}
    score = max(0, 100 - sum(penalty[i["severity"]] for i in issues))

    daily_total = df.groupby("ds")["y"].sum()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "score": score,
        "row_count": n,
        "series_count": int(len(per_series)),
        "store_count": int(df["store_code"].nunique()),
        "date_min": df["ds"].min().date().isoformat() if n else None,
        "date_max": global_max.date().isoformat() if n else None,
        "null_fraction": null_fraction,
        "duplicate_rows": dup_count,
        "negative_values": negative,
        "zero_fraction": round(zero_fraction, 5),
        "missing_series_days": total_missing,
        "outlier_count": outliers,
        "target_stats": {
            k: (None if pd.isna(v) else float(v))
            for k, v in y.describe(percentiles=[0.05, 0.5, 0.95]).items()
        },
        "weekday_profile": {
            int(k): float(v) for k, v in df.groupby(df["ds"].dt.day_of_week)["y"].mean().items()
        },
        # Downsampled to <= 400 points for charting.
        "daily_total": [
            {"ds": d.date().isoformat(), "y": float(v)}
            for d, v in daily_total.iloc[:: max(1, len(daily_total) // 400)].items()
        ],
        "issues": issues,
    }
