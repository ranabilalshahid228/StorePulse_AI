"""Training & Optuna tuning pipelines (run inside Celery workers)."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.ml import plots
from app.ml.data import load_history
from app.ml.families import FAMILIES, ForecastModel, resolve_device, suggest_params
from app.ml.features import (
    LAG_STEPS,
    SERIES_KEYS,
    FeatureSpec,
    build_features,
    complete_calendar,
    training_mask,
)
from app.ml.metrics import empirical_interval, interval_metrics, regression_metrics, series_scales
from app.models.enums import ModelFamily
from app.services.storage import ArtifactKeys, get_storage
from app.workers.progress import ProgressReporter

logger = logging.getLogger(__name__)


class PreparedData:
    def __init__(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        val_frame: pd.DataFrame,
        spec: FeatureSpec,
        val_scale: np.ndarray | None = None,
    ) -> None:
        self.X_tr, self.y_tr, self.X_val, self.y_val = X_tr, y_tr, X_val, y_val
        self.val_frame = val_frame
        self.spec = spec
        # RMSSE inputs: series id + training-period naive-error scale for every validation row.
        self.val_series = (
            val_frame["store_code"].astype(str) + "|" + val_frame["item_id"].astype(str)
        ).to_numpy()
        self.val_scale = val_scale

    def metrics(self, preds: np.ndarray) -> dict[str, Any]:
        return regression_metrics(self.y_val, preds, series=self.val_series, scale=self.val_scale)


def prepare_data(
    parquet_path: Path,
    mapping: dict[str, Any],
    horizon: int,
    validation_days: int,
    store_code: str | None,
    *,
    history_days: int | None = None,
    store_static: dict[str, dict[str, Any]] | None = None,
    seq_len: int = 0,
) -> PreparedData:
    """Load, complete, featurize and split. `history_days` keeps only the most recent N days of
    targets for training (plus the lag window needed to build their features). `seq_len` > 0 adds
    the target window consumed by sequence models."""
    raw = load_history(parquet_path, store_code)
    if history_days:
        spec_probe = FeatureSpec.from_mapping(mapping, horizon)
        keep = history_days + spec_probe.lag_offset + max(max(LAG_STEPS), seq_len, 91, 371) + 28
        raw = raw[raw["ds"] > raw["ds"].max() - pd.Timedelta(days=keep)]
    present = set(raw["store_code"].astype(str).unique())
    spec = FeatureSpec.from_mapping(
        mapping, horizon, {k: v for k, v in (store_static or {}).items() if k in present}
    )
    spec.seq_len = seq_len
    spec.seq_covariates = spec.default_seq_covariates() if seq_len else []
    full = complete_calendar(raw)
    feats = build_features(full, spec, fit_categories=True)
    cutoff = feats["ds"].max() - pd.Timedelta(days=validation_days)
    usable = feats[training_mask(feats, spec)]
    if history_days:
        usable = usable[usable["ds"] > feats["ds"].max() - pd.Timedelta(days=history_days)]
    tr = usable[usable["ds"] <= cutoff]
    if spec.open_column and spec.zero_when_closed and spec.open_column in tr:
        # Closed days are forecast as exactly 0 by rule, so the model only learns open-day demand.
        tr = tr[tr[spec.open_column].fillna(1) != 0]
    val = usable[usable["ds"] > cutoff]
    if len(tr) < 100 or len(val) < 7:
        raise ValueError(
            f"Not enough history: {len(tr)} train rows / {len(val)} validation rows "
            f"(need >= 100 / 7). Reduce horizon or validation_days."
        )
    cols = spec.model_columns
    # RMSSE scale from each series' full training-period history (not just the usable rows).
    scales = series_scales(feats[feats["ds"] <= cutoff], SERIES_KEYS)
    val_idx = pd.MultiIndex.from_frame(val[SERIES_KEYS].astype(str))
    val_scale = scales.reindex(val_idx).to_numpy(dtype="float64")
    return PreparedData(
        tr[cols],
        tr["y"].to_numpy("float64"),
        val[cols],
        val["y"].to_numpy("float64"),
        val[["ds", "store_code", "item_id", "y"]].copy(),
        spec,
        val_scale=val_scale,
    )


def inner_split(data: PreparedData, days: int) -> PreparedData:
    """Hold out the last `days` days of the *training* period for hyperparameter search, so the final
    validation window is never seen during tuning. Leak-free: target features lag by >= horizon."""
    X = data.X_tr
    dates = pd.to_datetime(
        (X["year"].astype(int) * 1000 + X["dayofyear"].astype(int)).astype(str), format="%Y%j"
    )
    val = (dates > dates.max() - pd.Timedelta(days=days)).to_numpy()
    if val.sum() < 7 or (~val).sum() < 100:
        raise ValueError("Not enough training history for an inner tuning split")
    frame = pd.DataFrame(
        {
            "ds": dates[val].to_numpy(),
            "store_code": X.loc[val, "store_code"].astype(str).to_numpy(),
            "item_id": X.loc[val, "item_id"].astype(str).to_numpy(),
            "y": data.y_tr[val],
        }
    )
    return PreparedData(X[~val], data.y_tr[~val], X[val], data.y_tr[val], frame, data.spec)


def _fit_once(
    family: ModelFamily,
    data: PreparedData,
    params: dict[str, Any],
    device: str,
    progress: ProgressReporter | None,
) -> tuple[ForecastModel, list[dict[str, Any]], np.ndarray]:
    model = FAMILIES[family](data.spec, params, device=device)
    logs: list[dict[str, Any]] = []

    def on_epoch(entry: dict[str, Any], frac: float) -> None:
        logs.append(entry)
        if progress:
            progress.update(frac, "training", message=f"epoch {entry.get('epoch')}")

    model.fit(data.X_tr, data.y_tr, data.X_val, data.y_val, on_epoch)
    preds = np.clip(model.predict(data.X_val), 0, None)
    return model, logs, preds


def _holdout_coverage(data: PreparedData, preds: np.ndarray) -> float | None:
    """Out-of-sample check of the empirical band: calibrate on the first half of the validation
    window, measure P10-P90 coverage on the second half."""
    ds = data.val_frame["ds"].to_numpy()
    days = np.unique(ds)
    if len(days) < 4:
        return None
    first = ds <= days[len(days) // 2 - 1]
    band = empirical_interval(data.y_val[first], preds[first])
    if not band:
        return None
    p, y = preds[~first], data.y_val[~first]
    lo, hi = np.clip(p * (1 + band["lower"]), 0, None), p * (1 + band["upper"])
    return interval_metrics(y, lo, hi)["coverage"]


def evaluate_model(model: ForecastModel, data: PreparedData, preds: np.ndarray) -> dict[str, Any]:
    """Validation metrics stored on the run: point metrics (incl. RMSSE, R2), the interval used at
    inference with its coverage/pinball loss, and the ensemble comparison when present."""
    metrics: dict[str, Any] = data.metrics(preds)
    quantiles = model.predict_quantiles(data.X_val)
    if quantiles:
        lo, hi = quantiles[min(quantiles)], quantiles[max(quantiles)]
        metrics["interval"] = {
            "method": "quantile_model",
            "levels": sorted(quantiles),
            **interval_metrics(data.y_val, lo, hi, quantiles),
        }
    else:
        interval = empirical_interval(data.y_val, preds)
        if interval:
            lo = np.clip(preds * (1 + interval["lower"]), 0, None)
            hi = preds * (1 + interval["upper"])
            # Coverage is in-sample here: the band was calibrated on this same window. Pinball losses
            # treat the band as P10/P90 and the point forecast as P50, comparable to quantile models.
            q = {0.1: lo, 0.5: preds, 0.9: hi}
            metrics["interval"] = {**interval, **interval_metrics(data.y_val, lo, hi, q)}
            metrics["interval"]["coverage_holdout"] = _holdout_coverage(data, preds)
    comparison = getattr(model, "comparison", None)
    if comparison:
        metrics["ensemble"] = comparison
    metrics["validation_start"] = data.val_frame["ds"].min().date().isoformat()
    metrics["validation_end"] = data.val_frame["ds"].max().date().isoformat()
    return metrics


def _benchmark_inference(model: ForecastModel, X: pd.DataFrame) -> float:
    """Median latency in ms per 1,000 rows over 3 runs."""
    sample = X.iloc[: min(len(X), 20_000)]
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.predict(sample)
        timings.append((time.perf_counter() - t0) * 1000)
    return float(np.median(timings) / max(len(sample), 1) * 1000)


def _finalize(
    *,
    org_id: uuid.UUID,
    run_id: uuid.UUID,
    family: ModelFamily,
    model: ForecastModel,
    data: PreparedData,
    preds: np.ndarray,
    logs: list[dict[str, Any]],
    progress: ProgressReporter,
) -> dict[str, Any]:
    storage = get_storage()
    metrics = evaluate_model(model, data, preds)

    progress.update(0.3, "benchmarking", force=True)
    inference_ms = _benchmark_inference(model, data.X_val)

    progress.update(0.5, "plotting", force=True)
    plot_keys: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        val = data.val_frame.assign(yhat=preds)
        rendered = {
            "validation_actual_vs_forecast": plots.actual_vs_forecast(val, tmpdir / "avf.png"),
            "learning_curve": plots.learning_curve(logs, tmpdir / "lc.png"),
        }
        importance = model.feature_importance()
        if importance:
            rendered["feature_importance"] = plots.feature_importance(importance, tmpdir / "fi.png")
            metrics["feature_importance"] = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:25])
        for name, path in rendered.items():
            if path is not None:
                key = ArtifactKeys.model_plot(org_id, run_id, name)
                storage.put_file(key, path)
                plot_keys.append(key)

        progress.update(0.8, "saving_artifact", force=True)
        art_key = ArtifactKeys.model_artifact(org_id, run_id, model.artifact_ext)
        local = tmpdir / f"model.{model.artifact_ext}"
        model.save(local)
        storage.put_file(art_key, local)

    return {
        "metrics": metrics,
        "epoch_logs": logs,
        "artifact_path": art_key,
        "plot_paths": plot_keys,
        "inference_ms": inference_ms,
        "train_rows": int(len(data.X_tr)),
        "feature_columns": data.spec.feature_columns,
        "hyperparameters": model.params,
    }


def train(
    *,
    org_id: uuid.UUID,
    run_id: uuid.UUID,
    parquet_path: Path,
    mapping: dict[str, Any],
    family: ModelFamily,
    params: dict[str, Any],
    horizon: int,
    validation_days: int,
    store_code: str | None,
    progress: ProgressReporter,
    history_days: int | None = None,
    store_static: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    progress.update(0.02, "loading_data", force=True)
    data = prepare_data(
        parquet_path,
        mapping,
        horizon,
        validation_days,
        store_code,
        history_days=history_days,
        store_static=store_static,
        seq_len=FAMILIES[family].required_seq_len(params),
    )
    device = resolve_device(get_settings().training_device, FAMILIES[family])
    progress.update(0.1, "training", message=f"device={device}", force=True)
    model, logs, preds = _fit_once(family, data, params, device, progress.sub(0.1, 0.85))
    result = _finalize(
        org_id=org_id,
        run_id=run_id,
        family=family,
        model=model,
        data=data,
        preds=preds,
        logs=logs,
        progress=progress.sub(0.85, 1.0),
    )
    result["device"] = device
    return result


def tune(
    *,
    org_id: uuid.UUID,
    run_id: uuid.UUID,
    parquet_path: Path,
    mapping: dict[str, Any],
    family: ModelFamily,
    base_params: dict[str, Any],
    search_space: dict[str, Any],
    n_trials: int,
    timeout_s: int | None,
    metric: str,
    horizon: int,
    validation_days: int,
    store_code: str | None,
    progress: ProgressReporter,
    history_days: int | None = None,
    store_static: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import optuna

    progress.update(0.02, "loading_data", force=True)
    # Size the window for the largest seq_len the search space can propose.
    seq_choices = search_space.get("seq_len", {}).get("choices", [])
    seq_len = max([FAMILIES[family].required_seq_len(base_params), *map(int, seq_choices)])
    data = prepare_data(
        parquet_path,
        mapping,
        horizon,
        validation_days,
        store_code,
        history_days=history_days,
        store_static=store_static,
        seq_len=seq_len,
    )
    device = resolve_device(get_settings().training_device, FAMILIES[family])
    trials: list[dict[str, Any]] = []
    tune_progress = progress.sub(0.08, 0.85)

    # Honest tuning: trials are scored on an inner split of the training period; the final validation
    # window is only used once, to report the tuned model.
    inner = inner_split(data, validation_days)
    best_rounds: dict[int, int] = {}
    best_epochs: dict[int, int] = {}

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest_params(trial, search_space)}
        model_t, _, preds = _fit_once(family, inner, params, device, None)
        if "best_iteration" in model_t.params:
            best_rounds[trial.number] = int(model_t.params["best_iteration"])
        elif "best_epoch" in model_t.params:
            best_epochs[trial.number] = int(model_t.params["best_epoch"])
        m = inner.metrics(preds)
        value = m.get(metric)
        if value is None:
            raise optuna.TrialPruned(f"metric {metric} undefined")
        trials.append(
            {
                "number": trial.number,
                "params": params,
                "value": float(value),
                "metrics": {k: v for k, v in m.items() if k != "n"},
            }
        )
        tune_progress.update(
            (trial.number + 1) / n_trials,
            "tuning",
            message=f"trial {trial.number + 1}/{n_trials} {metric}={value:.4f}",
            force=True,
        )
        return float(value)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"run-{run_id}",
        storage=get_settings().optuna_storage_url,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, catch=(ValueError,))
    if not study.best_trials:
        raise RuntimeError("No Optuna trial completed successfully")

    best_params = {**base_params, **study.best_params}
    if study.best_trial.number in best_rounds:
        # GBDT: fix the round count found on the inner split (scaled to the larger training set) and
        # disable early stopping, so the final validation score is fully out-of-sample.
        scale = len(data.X_tr) / max(len(inner.X_tr), 1)
        best_params |= {
            "num_boost_round": max(1, round(best_rounds[study.best_trial.number] * scale)),
            "early_stopping_rounds": 0,
        }
    elif study.best_trial.number in best_epochs:
        # Neural nets: fix the epoch count found on the inner split; no early stopping on the final window.
        best_params |= {"max_epochs": max(1, best_epochs[study.best_trial.number]), "patience": 0}
    progress.update(0.86, "training_best", force=True)
    model, logs, preds = _fit_once(family, data, best_params, device, progress.sub(0.86, 0.93))
    result = _finalize(
        org_id=org_id,
        run_id=run_id,
        family=family,
        model=model,
        data=data,
        preds=preds,
        logs=logs,
        progress=progress.sub(0.93, 1.0),
    )
    result["metrics"]["tuning"] = {
        "scheme": "inner split of the training period",
        "inner_validation_days": validation_days,
        "inner_best_value": float(study.best_value),
    }
    result["metrics"]["best_trial"] = {
        "number": study.best_trial.number,
        "value": study.best_value,
        "metric": metric,
    }
    result["trials"] = trials
    result["device"] = device
    return result
