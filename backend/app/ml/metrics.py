from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "wape": None,
            "mape": None,
            "smape": None,
            "rmspe": None,
            "bias_pct": None,
        }
    err = yp - yt
    abs_sum = np.abs(yt).sum()
    nz = yt != 0
    denom = np.abs(yt) + np.abs(yp)
    return {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.abs(err).sum() / abs_sum) if abs_sum else None,
        "mape": float(np.mean(np.abs(err[nz] / yt[nz]))) if nz.any() else None,
        "smape": float(np.mean(np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0))),
        # Rossmann competition metric: root mean squared percentage error over non-zero actuals.
        "rmspe": float(np.sqrt(np.mean((err[nz] / yt[nz]) ** 2))) if nz.any() else None,
        "bias_pct": float(err.sum() / abs_sum) if abs_sum else None,
    }


def empirical_interval(
    y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.9
) -> dict[str, float] | None:
    """Relative-residual quantiles from validation, applied multiplicatively at inference."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    # Exclude zero actuals: closures are explained by covariates (e.g. Open) and would pin the lower bound at -100%.
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
