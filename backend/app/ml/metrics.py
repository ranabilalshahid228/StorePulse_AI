from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILE_LEVELS = (0.1, 0.5, 0.9)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    series: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Point-forecast metrics.

    RMSSE (M5 definition) needs `series` (row -> series id) and `scale` (row -> that series' mean
    squared one-step naive error over its *training* history). It is computed per series and
    averaged with equal weight; series with a zero scale (constant history) are skipped.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    keys = ("mae", "rmse", "wape", "mape", "smape", "rmspe", "bias_pct", "r2", "rmsse")
    if yt.size == 0:
        return {"n": 0, **dict.fromkeys(keys)}
    err = yp - yt
    abs_sum = np.abs(yt).sum()
    nz = yt != 0
    denom = np.abs(yt) + np.abs(yp)
    sst = float(((yt - yt.mean()) ** 2).sum())
    out: dict[str, float | None] = {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.abs(err).sum() / abs_sum) if abs_sum else None,
        "mape": float(np.mean(np.abs(err[nz] / yt[nz]))) if nz.any() else None,
        "smape": float(np.mean(np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0))),
        # Rossmann competition metric: root mean squared percentage error over non-zero actuals.
        "rmspe": float(np.sqrt(np.mean((err[nz] / yt[nz]) ** 2))) if nz.any() else None,
        "bias_pct": float(err.sum() / abs_sum) if abs_sum else None,
        "r2": float(1 - (err**2).sum() / sst) if sst > 0 else None,
        "rmsse": None,
    }
    if series is not None and scale is not None:
        s = np.asarray(scale, dtype="float64")[mask]
        sid = np.asarray(series)[mask]
        df = pd.DataFrame({"sid": sid, "se": err**2, "scale": s})
        df = df[df["scale"] > 0]
        if len(df):
            per = df.groupby("sid").agg(mse=("se", "mean"), scale=("scale", "first"))
            out["rmsse"] = float(np.sqrt(per["mse"] / per["scale"]).mean())
    return out


def series_scales(train: pd.DataFrame, keys: list[str]) -> pd.Series:
    """M5 RMSSE scale per series: mean squared first difference of y over the training history,
    starting at the series' first non-zero observation."""
    df = train[[*keys, "ds", "y"]].sort_values([*keys, "ds"])
    started = (df["y"].fillna(0) != 0).groupby([df[k] for k in keys], sort=False).cummax()
    df = df[started.to_numpy()]
    diff = df.groupby(keys, sort=False)["y"].diff()
    return (diff**2).groupby([df[k] for k in keys]).mean()


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    quantiles: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    """Empirical coverage and mean width of [lower, upper]; pinball loss for each quantile given."""
    y = np.asarray(y_true, dtype="float64")
    lo, hi = np.asarray(lower, dtype="float64"), np.asarray(upper, dtype="float64")
    m = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
    out = {
        "coverage": float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()) if m.any() else float("nan"),
        "mean_width": float((hi[m] - lo[m]).mean()) if m.any() else float("nan"),
    }
    for q, pred in (quantiles or {}).items():
        out[f"pinball_p{round(q * 100):02d}"] = pinball_loss(y, pred, q)
    return out


def pinball_loss(y_true: np.ndarray, y_q: np.ndarray, q: float) -> float:
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_q, dtype="float64")
    m = ~np.isnan(y) & ~np.isnan(p)
    d = y[m] - p[m]
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def empirical_interval(
    y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.8
) -> dict[str, float | str] | None:
    """Relative-residual quantiles from validation, applied multiplicatively at inference.

    Coverage 0.8 matches the P10-P90 band of quantile models so the two are comparable.
    Zero actuals are excluded: closures are explained by covariates (e.g. Open) and would pin the
    lower bound at -100%. For intermittent series prefer quantile models (LightGBM `quantiles`).
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = (y_pred > 0) & (y_true > 0) & ~np.isnan(y_true)
    if mask.sum() < 30:
        return None
    rel = y_true[mask] / y_pred[mask] - 1.0
    alpha = (1 - coverage) / 2
    return {
        "method": "empirical_relative_residual",
        "coverage": coverage,
        "lower": float(np.quantile(rel, alpha)),
        "upper": float(np.quantile(rel, 1 - alpha)),
    }
