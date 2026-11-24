"""Model families. Each implements fit / predict / save / load over the shared feature frame.

Add a family: subclass `ForecastModel`, register it in `FAMILIES`, add a default search space,
add the enum value in `app.models.enums.ModelFamily` and in the frontend `types.ts`.

Models receive the full prepared frame and select their own inputs via `input_columns`
(tabular models ignore the `seq_*` window that sequence models consume).
"""

from __future__ import annotations

import importlib.util
import io
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd

from app.ml.features import SERIES_KEYS, FeatureSpec, closed_mask
from app.ml.metrics import QUANTILE_LEVELS, regression_metrics
from app.models.enums import ModelFamily

logger = logging.getLogger(__name__)

EpochCallback = Callable[[dict[str, Any], float], None]  # (log_entry, fraction_done)
INTERMITTENT_ZERO_SHARE = 0.3


def is_intermittent(y: np.ndarray) -> bool:
    """Zero-heavy, non-negative targets (e.g. M5 item-day units) get count-appropriate losses."""
    y = np.asarray(y, dtype="float64")
    y = y[~np.isnan(y)]
    return bool(len(y)) and float(y.min()) >= 0 and float((y == 0).mean()) > INTERMITTENT_ZERO_SHARE


def _with_closed_rule(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Force exact zeros on rows the open flag marks closed (point forecasts and quantiles)."""

    def wrapper(self: ForecastModel, X: pd.DataFrame) -> Any:
        out = fn(self, X)
        mask = closed_mask(self.spec, X) if out is not None else None
        if mask is None or not mask.any():
            return out
        if isinstance(out, dict):
            return {q: np.where(mask, 0.0, v) for q, v in out.items()}
        return np.where(mask, 0.0, out)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    wrapper.__doc__ = fn.__doc__
    return wrapper


class ForecastModel(ABC):
    family: ClassVar[ModelFamily]
    description: ClassVar[str]
    artifact_ext: ClassVar[str] = "pkl"
    supports_gpu: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {}
    search_space: ClassVar[dict[str, Any]] = {}

    def __init__(self, spec: FeatureSpec, params: dict[str, Any] | None = None, device: str = "cpu") -> None:
        self.spec = spec
        self.params = {**self.default_params, **(params or {})}
        self.device = device

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("predict", "predict_quantiles"):
            fn = cls.__dict__.get(name)
            if fn is not None and not hasattr(fn, "__wrapped__"):
                setattr(cls, name, _with_closed_rule(fn))

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def required_seq_len(cls, params: dict[str, Any]) -> int:
        return 0

    @property
    def input_columns(self) -> list[str]:
        return self.spec.feature_columns

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, on_epoch: EpochCallback
    ) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray] | None:
        """P10/P50/P90 when the model learns them directly; None -> empirical interval is used."""
        return None

    def feature_importance(self) -> dict[str, float] | None:
        return None

    def save(self, path: Path) -> None:
        joblib.dump(
            {"family": self.family.value, "spec": self.spec, "params": self.params, "model": self}, path
        )

    @staticmethod
    def load(path: Path) -> ForecastModel:
        if path.suffix == ".pt":
            return TorchMLPModel.load_pt(path)
        return joblib.load(path)["model"]


# ----------------------------------------------------------------------------------------


class SeasonalNaiveModel(ForecastModel):
    family = ModelFamily.SEASONAL_NAIVE
    description = "Mean of the last N weekday-aligned observations (benchmark baseline)."
    default_params = {"weeks": 4}
    search_space = {"weeks": {"type": "int", "low": 1, "high": 4}}

    def _cols(self) -> list[str]:
        o = self.spec.lag_offset
        return [f"lag_{o + 7 * k}" for k in range(int(self.params["weeks"]))]

    def fit(self, X, y, X_val, y_val, on_epoch) -> None:  # noqa: ANN001 - nothing to learn
        m = regression_metrics(y_val, self.predict(X_val))
        on_epoch({"epoch": 1, **{f"val_{k}": v for k, v in m.items()}}, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(X[self._cols()].mean(axis=1).to_numpy(dtype="float64"), nan=0.0)


# ----------------------------------------------------------------------------------------


class ARIMAModel(ForecastModel):
    family = ModelFamily.ARIMA
    description = "CPU benchmark that fits an ARIMA model to each store/item's daily demand history."
    default_params = {"p": 1, "d": 1, "q": 1}
    search_space = {
        "p": {"type": "int", "low": 0, "high": 2},
        "d": {"type": "int", "low": 0, "high": 1},
        "q": {"type": "int", "low": 0, "high": 2},
    }

    def __init__(self, spec: FeatureSpec, params: dict[str, Any] | None = None, device: str = "cpu") -> None:
        super().__init__(spec, params, device)
        for name, value in self.params.items():
            lower, upper = (2, 31) if name == "season_length" else (0, 3)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        self.order = tuple(self.params[k] for k in ("p", "d", "q"))
        self.seasonal_order = (
            tuple(self.params[k] for k in ("P", "D", "Q", "season_length"))
            if "season_length" in self.params
            else (0, 0, 0, 0)
        )
        if not any(self.seasonal_order[:3]):
            self.seasonal_order = (0, 0, 0, 0)
        # Per series: (training target, fitted coefficients). Full statsmodels results objects keep
        # every filter matrix (~16 MB per SARIMA series), so the filter is rebuilt from these when
        # predicting - same forecasts, a few KB per series.
        self.results: dict[tuple[str, str], tuple[pd.Series, np.ndarray]] = {}
        self.weekday_profiles: dict[tuple[str, str], np.ndarray] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        on_epoch: EpochCallback,
        *,
        history: pd.DataFrame | None = None,
    ) -> None:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if history is None:
            raise ValueError("ARIMA and SARIMA must run as standalone benchmarks with daily history")
        groups = history.groupby(SERIES_KEYS, sort=False)
        n_series = groups.ngroups
        self.results.clear()
        self.weekday_profiles.clear()
        min_observations = max(14, 3 * self.seasonal_order[-1])
        for i, (key, series) in enumerate(groups, start=1):
            target = series.set_index("ds")["y"].asfreq("D")
            self.weekday_profiles[key] = self._weekday_profile(target)
            if target.count() >= min_observations:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        result = SARIMAX(target, order=self.order, seasonal_order=self.seasonal_order).fit(
                            disp=False
                        )
                    if result.mle_retvals["converged"] and np.isfinite(result.params).all():
                        self.results[key] = (target.astype("float64"), np.asarray(result.params, float))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    logger.debug("Could not fit %s for series %s", self.family, key, exc_info=True)
            if i % 25 == 0 and i < n_series:
                on_epoch({"epoch": i, "fitted_series": len(self.results)}, i / n_series)

        if not self.results:
            raise ValueError(f"{self.family} could not fit any series. Use more history or a simpler order.")
        metrics = regression_metrics(y_val, self.predict(X_val))
        on_epoch(
            {
                "epoch": n_series,
                "fitted_series": len(self.results),
                "fallback_series": n_series - len(self.results),
                **{f"val_{k}": v for k, v in metrics.items()},
            },
            1.0,
        )

    def _filtered(self, target: pd.Series, params: np.ndarray) -> Any:
        """Kalman-filter `target` with fixed coefficients (no re-estimation)."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(target, order=self.order, seasonal_order=self.seasonal_order)
            return model.filter(params)

    @staticmethod
    def _weekday_profile(target: pd.Series) -> np.ndarray:
        recent = target.tail(28)
        return recent.groupby(recent.index.dayofweek).mean().reindex(range(7)).fillna(0).to_numpy()

    def _predict_dates(
        self, X: pd.DataFrame, dates: pd.DatetimeIndex, history: pd.DataFrame | None
    ) -> np.ndarray:
        predictions = np.zeros(len(X))
        history_groups = history.groupby(SERIES_KEYS, sort=False) if history is not None else None
        for key, rows in X.groupby(SERIES_KEYS, sort=False, observed=True).indices.items():
            target = None
            profile = self.weekday_profiles.get(key)
            if history_groups is not None and key in history_groups.indices:
                target = history_groups.get_group(key).set_index("ds")["y"].asfreq("D")
                profile = self._weekday_profile(target)
            if profile is not None:
                predictions[rows] = profile[dates[rows].dayofweek]
            fitted = self.results.get(key)
            if fitted is None:
                continue
            train_target, params = fitted
            # With history, condition on the latest actuals; otherwise forecast from the training end.
            series = target if target is not None else train_target
            origin = series.index[-1]
            try:
                result = self._filtered(series, params)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                logger.debug("Could not filter %s for series %s", self.family, key, exc_info=True)
                continue
            steps = (dates[rows] - origin).days.to_numpy()
            valid = steps > 0
            if valid.any():
                values = np.asarray(result.forecast(steps=int(steps[valid].max())))[steps[valid] - 1]
                finite = np.isfinite(values)
                predictions[rows[valid][finite]] = np.clip(values[finite], 0, None)
        return np.clip(predictions, 0, None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dates = pd.to_datetime(X["year"].astype(int) * 1000 + X["dayofyear"].astype(int), format="%Y%j")
        return self._predict_dates(X, pd.DatetimeIndex(dates), history=None)

    def predict_from_history(
        self, X: pd.DataFrame, dates: pd.DatetimeIndex, history: pd.DataFrame
    ) -> np.ndarray:
        predictions = self._predict_dates(X, dates, history)
        closed = closed_mask(self.spec, X)
        return np.where(closed, 0.0, predictions) if closed is not None else predictions


class SARIMAModel(ARIMAModel):
    family = ModelFamily.SARIMA
    description = "CPU benchmark with a weekly seasonal component for each store/item's daily demand."
    default_params = {**ARIMAModel.default_params, "P": 1, "D": 0, "Q": 0, "season_length": 7}
    search_space = {
        **ARIMAModel.search_space,
        **{name: {"type": "int", "low": 0, "high": 1} for name in ("P", "D", "Q")},
    }


class LightGBMModel(ForecastModel):
    family = ModelFamily.LIGHTGBM
    description = (
        "Gradient-boosted trees (LightGBM). objective=auto picks Tweedie for zero-heavy targets; "
        "optional quantile boosters give P10/P50/P90 intervals."
    )
    default_params = {
        "objective": "auto",  # auto | regression | tweedie | poisson
        "tweedie_variance_power": 1.3,
        "quantiles": "auto",  # auto (on for intermittent targets) | true | false
        "target_transform": "none",  # none | log1p (fit on log1p(y); good for wide-scale sales)
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_boost_round": 2000,
        "early_stopping_rounds": 100,
    }
    search_space = {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "num_leaves": {"type": "int", "low": 15, "high": 255, "log": True},
        "min_data_in_leaf": {"type": "int", "low": 10, "high": 300, "log": True},
        "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0},
        "bagging_fraction": {"type": "float", "low": 0.5, "high": 1.0},
        "lambda_l2": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "objective": {"type": "categorical", "choices": ["regression", "tweedie", "poisson"]},
        "tweedie_variance_power": {"type": "float", "low": 1.05, "high": 1.9},
        "quantiles": {"type": "categorical", "choices": [False]},  # tune the point model only
        "target_transform": {"type": "categorical", "choices": ["none", "log1p"]},
    }
    _NON_LGB = (
        "target_transform",
        "num_boost_round",
        "early_stopping_rounds",
        "quantiles",
        "best_iteration",
        "resolved_objective",
    )

    def _lgb_params(self, objective: str) -> dict[str, Any]:
        p = {k: v for k, v in self.params.items() if k not in self._NON_LGB}
        p["objective"] = objective
        if objective != "tweedie":
            p.pop("tweedie_variance_power", None)
        p.update({"verbosity": -1, "seed": 42})
        return p

    def fit(self, X, y, X_val, y_val, on_epoch) -> None:  # noqa: ANN001
        import lightgbm as lgb

        X, X_val = X[self.input_columns], X_val[self.input_columns]
        intermittent = is_intermittent(y)
        self.log_target = self.params.get("target_transform", "none") == "log1p"
        objective = self.params["objective"]
        if objective == "auto":
            objective = "regression" if self.log_target else ("tweedie" if intermittent else "regression")
        if self.log_target:
            if objective in ("tweedie", "poisson"):
                raise ValueError("target_transform=log1p needs an L2/L1/huber objective, not " + objective)
            y, y_val = np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(y_val, 0, None))
        self.params["resolved_objective"] = objective
        want_q = self.params.get("quantiles", "auto")
        self.quantile_enabled = bool(intermittent) if want_q == "auto" else bool(want_q)

        rounds = int(self.params["num_boost_round"])
        es = int(self.params["early_stopping_rounds"])
        # es <= 0: train exactly num_boost_round rounds; the validation set is only logged, never used
        # to pick the model (keeps the reported validation score fully out-of-sample).
        stop = [lgb.early_stopping(es, verbose=False)] if es > 0 else []
        cats = [c for c in self.spec.categorical_features if c in X.columns]
        dtrain = lgb.Dataset(X, y, categorical_feature=cats, free_raw_data=False)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        n_models = 1 + (len(QUANTILE_LEVELS) if self.quantile_enabled else 0)

        def logger_for(stage: int, prefix: str) -> Callable[[Any], None]:
            def _log(env: Any) -> None:
                it = env.iteration + 1
                if it % 25 == 0 or it == 1:
                    entry: dict[str, Any] = {"epoch": it, "model": prefix}
                    for _, name, value, _ in env.evaluation_result_list:
                        entry[f"val_{name}"] = float(value)
                    on_epoch(entry, (stage + it / rounds) / n_models)

            return _log

        self.booster = lgb.train(
            {**self._lgb_params(objective), "metric": ["l1", "l2"]},
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[*stop, logger_for(0, "point")],
        )
        self.params["best_iteration"] = int(self.booster.best_iteration or rounds)

        self.q_boosters: dict[float, Any] = {}
        if self.quantile_enabled:
            for i, q in enumerate(QUANTILE_LEVELS, start=1):
                params = {**self._lgb_params("quantile"), "alpha": q, "metric": "quantile"}
                self.q_boosters[q] = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=rounds,
                    valid_sets=[dval],
                    valid_names=["val"],
                    callbacks=[*stop, logger_for(i, f"p{round(q * 100)}")],
                )

    def _inverse(self, z: np.ndarray) -> np.ndarray:
        return np.expm1(z) if getattr(self, "log_target", False) else z

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._inverse(
            self.booster.predict(X[self.input_columns], num_iteration=self.booster.best_iteration)
        )

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray] | None:
        if not getattr(self, "q_boosters", None):
            return None
        Xi = X[self.input_columns]
        raw = np.column_stack(
            [
                np.clip(self._inverse(b.predict(Xi, num_iteration=b.best_iteration)), 0, None)
                for b in self.q_boosters.values()
            ]
        )
        raw.sort(axis=1)  # remove quantile crossing
        return {q: raw[:, i] for i, q in enumerate(self.q_boosters)}

    def feature_importance(self) -> dict[str, float]:
        gains = self.booster.feature_importance(importance_type="gain")
        return dict(zip(self.booster.feature_name(), map(float, gains), strict=True))


# ----------------------------------------------------------------------------------------


class TorchMLPModel(ForecastModel):
    family = ModelFamily.TORCH_MLP
    description = "PyTorch MLP with categorical embeddings; trains on CUDA when available."
    artifact_ext = "pt"
    supports_gpu = True
    default_params = {
        "hidden_sizes": [256, 128],
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 4096,
        "max_epochs": 30,
        "patience": 5,  # <= 0: no early stopping, keep the weights after max_epochs
        "embedding_dim": 8,
        "loss": "auto",  # auto (poisson for intermittent targets) | mse | poisson   ("log_mse" = legacy)
        "target_transform": "log1p",  # log1p | standard   (mse/quantile space; both are z-scored)
        "quantiles": "auto",  # auto (on for intermittent targets) | true | false: P10/P50/P90 heads
        "quantile_weight": 0.5,  # pinball-loss weight of the quantile heads vs the point loss
        "feature_log": True,  # signed log1p on wide-range numeric inputs (sales-scale lags) first
    }
    search_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "weight_decay": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [1024, 2048, 4096, 8192]},
        "embedding_dim": {"type": "categorical", "choices": [4, 8, 16]},
        "loss": {"type": "categorical", "choices": ["mse", "poisson"]},
        "target_transform": {"type": "categorical", "choices": ["log1p", "standard"]},
        "quantiles": {"type": "categorical", "choices": [False]},  # tune the point model only
    }

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("torch") is not None

    # -- tensor prep -----------------------------------------------------------------------
    def _log_num(self, num: np.ndarray) -> np.ndarray:
        mask = getattr(self, "num_log", None)
        if mask is None or not np.any(mask):
            return num
        num = num.copy()
        num[:, mask] = np.sign(num[:, mask]) * np.log1p(np.abs(num[:, mask]))
        return num

    def _tabular(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cols = self.spec.feature_columns
        if not hasattr(self, "num_cols"):
            self.cat_cols = [c for c in self.spec.categorical_features if c in cols]
            self.num_cols = [c for c in cols if c not in self.cat_cols]
            num = X[self.num_cols].to_numpy(dtype="float32")
            if self.params.get("feature_log", True) and len(num):
                with np.errstate(all="ignore"):
                    p99 = np.nanquantile(np.abs(num), 0.99, axis=0)
                self.num_log = np.nan_to_num(p99, nan=0.0) > 50.0
            else:
                self.num_log = np.zeros(len(self.num_cols), dtype=bool)
            num = self._log_num(num)
            fill = np.nanmedian(num, axis=0)
            self.num_fill = np.where(np.isnan(fill), 0.0, fill).astype("float32")
            filled = np.where(np.isnan(num), self.num_fill, num)
            self.num_mean = filled.mean(axis=0)
            self.num_std = filled.std(axis=0) + 1e-6
        num = self._log_num(X[self.num_cols].to_numpy(dtype="float32"))
        # A column constant in training (e.g. lag_364 with one year of history) has std ~ 0: dividing by
        # it turns any later value into ~1e9 and saturates the net. The net learned nothing from such a
        # column, so feed its training value (z = 0). Applied at use time, so saved models are fixed too.
        constant = self.num_std < 1e-4
        filled = np.where(np.isnan(num), self.num_fill, num)
        z = (filled - self.num_mean) / np.where(constant, 1.0, self.num_std)
        num = np.where(constant, 0.0, z)
        cat = np.stack([X[c].cat.codes.to_numpy() + 1 for c in self.cat_cols], axis=1).astype("int64")
        return num.astype("float32"), cat  # code 0 == unknown/missing

    def _inputs(self, X: pd.DataFrame) -> tuple[np.ndarray, ...]:
        return self._tabular(X)

    def _encoder(self, nn: Any) -> tuple[Any, int]:
        """Optional extra encoder (sequence models); returns (module or None, output width)."""
        return None, 0

    @property
    def n_outputs(self) -> int:
        return 1 + len(getattr(self, "quantile_levels", None) or [])

    def _build(self) -> Any:
        import torch
        from torch import nn

        cards = [len(self.spec.category_levels.get(c, [])) + 1 for c in self.cat_cols]
        emb_dim = int(self.params["embedding_dim"])
        hidden = list(self.params["hidden_sizes"])
        dropout = float(self.params["dropout"])
        n_num = len(self.num_cols)
        n_out = self.n_outputs
        encoder, enc_dim = self._encoder(nn)
        bias_init = float(getattr(self, "y_bias_init", 0.0))

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embs = nn.ModuleList([nn.Embedding(c, min(emb_dim, c)) for c in cards])
                self.encoder = encoder
                in_dim = n_num + sum(min(emb_dim, c) for c in cards) + enc_dim
                layers: list[nn.Module] = []
                for h in hidden:
                    layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
                    in_dim = h
                head = nn.Linear(in_dim, n_out)
                nn.init.zeros_(head.bias)
                with torch.no_grad():
                    head.bias[0] = bias_init  # quantile heads start at the z-scored mean (0)
                layers.append(head)
                self.mlp = nn.Sequential(*layers)

            def forward(self, num: torch.Tensor, cat: torch.Tensor, *extra: torch.Tensor) -> torch.Tensor:
                parts = [num, *[e(cat[:, i]) for i, e in enumerate(self.embs)]]
                if self.encoder is not None:
                    parts.append(self.encoder(*extra))
                return self.mlp(torch.cat(parts, dim=1))  # (batch, 1 + n_quantiles)

        return Net()

    # -- target handling -------------------------------------------------------------------
    def _resolve(self, y: np.ndarray) -> None:
        loss = self.params.get("loss", "auto")
        transform = self.params.get("target_transform", "log1p")
        if loss == "log_mse":  # legacy name
            loss, transform = "mse", "log1p"
        if transform not in ("log1p", "standard"):
            raise ValueError(f"target_transform must be log1p or standard, not {transform!r}")
        intermittent = is_intermittent(y)
        self.loss_kind = ("poisson" if intermittent else "mse") if loss == "auto" else loss
        self.target_transform = transform
        q = self.params.get("quantiles", "auto")
        want = intermittent if q == "auto" else q in (True, "true")
        self.quantile_levels = list(QUANTILE_LEVELS) if want else []

    def _to_space(self, y: np.ndarray) -> np.ndarray:
        """Target units -> transform space (log1p or identity), before z-scoring."""
        y = np.clip(np.asarray(y, dtype="float64"), 0, None)
        return np.log1p(y) if self.target_transform == "log1p" else y

    def _target(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(point-loss target, z-scored target used by mse and the quantile heads)."""
        y = np.clip(np.asarray(y, dtype="float64"), 0, None)
        v = self._to_space(y)
        self.y_mu, self.y_sigma = float(v.mean()), float(v.std() + 1e-6)
        # Decoding cap: 5x the largest training target, so a wild output can't overflow expm1.
        self.v_max = float(self._to_space(np.array([max(float(y.max()), 1.0) * 5.0]))[0])
        z = (v - self.y_mu) / self.y_sigma
        if self.loss_kind == "poisson":
            # Network outputs log-rate; start at log(mean) so early epochs aren't spent on the level.
            self.y_bias_init = float(np.log(max(y.mean(), 1e-6)))
            return y, z
        self.y_bias_init = 0.0
        return z, z

    def _decode_z(self, z: np.ndarray) -> np.ndarray:
        """z-scored transform space -> target units, clipped to [0, cap]: safe at zero and at the top."""
        v = np.clip(z * self.y_sigma + self.y_mu, 0.0, getattr(self, "v_max", np.inf))
        return np.expm1(v) if self.target_transform == "log1p" else v

    def _decode_point(self, out: np.ndarray) -> np.ndarray:
        if self.loss_kind == "poisson":
            v_max = getattr(self, "v_max", None)
            cap = (
                np.inf if v_max is None else (np.expm1(v_max) if self.target_transform == "log1p" else v_max)
            )
            return np.minimum(np.exp(np.clip(out, -20, 20)), cap)
        return self._decode_z(out)

    def fit(self, X, y, X_val, y_val, on_epoch) -> None:  # noqa: ANN001
        import torch

        torch.manual_seed(42)
        dev = torch.device(self.device)
        self._resolve(y)
        train_arrays = [torch.tensor(a) for a in self._inputs(X)]
        val_arrays = self._inputs(X_val)
        main, z = self._target(y)
        yt = torch.tensor(main, dtype=torch.float32)
        zt = torch.tensor(z, dtype=torch.float32)
        qs = torch.tensor(self.quantile_levels, dtype=torch.float32, device=dev)
        q_weight = float(self.params.get("quantile_weight", 0.5))
        self.net = self._build().to(dev)
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]
        )
        loss_fn = (
            torch.nn.PoissonNLLLoss(log_input=True) if self.loss_kind == "poisson" else torch.nn.MSELoss()
        )
        # Cap the batch so small datasets still get >= ~50 optimizer steps per epoch.
        bs = max(64, min(int(self.params["batch_size"]), len(yt) // 50))
        max_epochs = int(self.params["max_epochs"])
        patience = int(self.params["patience"])
        best, best_state, best_epoch, bad = float("inf"), None, max_epochs, 0
        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            self.net.train()
            perm = torch.randperm(len(yt))
            total = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                pred = self.net(*[a[idx].to(dev) for a in train_arrays])
                loss = loss_fn(pred[:, 0], yt[idx].to(dev))
                if len(qs):
                    d = zt[idx].to(dev).unsqueeze(1) - pred[:, 1:]
                    loss = loss + q_weight * torch.maximum(qs * d, (qs - 1) * d).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                total += loss.item() * len(idx)
            m = regression_metrics(y_val, self._decode_point(self._raw(val_arrays)[:, 0]))
            entry = {
                "epoch": epoch,
                "train_loss": total / len(yt),
                "val_mae": m["mae"],
                "val_wape": m["wape"],
                "seconds": round(time.perf_counter() - t0, 3),
            }
            on_epoch(entry, epoch / max_epochs)
            if patience <= 0:
                continue  # fixed epoch budget (e.g. found by honest tuning): keep the last weights
            score = m["mae"] if m["mae"] is not None else float("inf")
            if score < best:
                best, bad, best_epoch = score, 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.params["best_epoch"] = best_epoch

    def _raw(self, arrays: tuple[np.ndarray, ...]) -> np.ndarray:
        import torch

        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(arrays[0]), 65536):
                o = self.net(*[torch.tensor(a[i : i + 65536]).to(dev) for a in arrays]).cpu().numpy()
                out.append(o.reshape(len(o), -1))
        return np.concatenate(out)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._decode_point(self._raw(self._inputs(X))[:, 0])

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray] | None:
        levels = getattr(self, "quantile_levels", None)
        if not levels:
            return None
        raw = self._raw(self._inputs(X))[:, 1:]
        # The inverse transform is monotone, so quantiles stay valid; sort removes crossing.
        qv = np.sort(self._decode_z(raw), axis=1)
        return {q: qv[:, i] for i, q in enumerate(levels)}

    # -- persistence -----------------------------------------------------------------------
    _PREP = (
        "cat_cols",
        "num_cols",
        "num_fill",
        "num_mean",
        "num_std",
        "num_log",
        "loss_kind",
        "target_transform",
        "quantile_levels",
        "y_mu",
        "y_sigma",
        "v_max",
        "y_bias_init",
        "seq_mean",
        "seq_std",
    )

    def _blob(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "spec": self.spec.to_dict(),
            "params": self.params,
            "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "prep": {k: getattr(self, k) for k in self._PREP if hasattr(self, k)},
        }

    @staticmethod
    def _from_blob(blob: dict[str, Any], device: str | None = None) -> TorchMLPModel:
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cls = FAMILIES[ModelFamily(blob["family"])]
        model = cls(FeatureSpec(**blob["spec"]), blob["params"], device=device)
        for k, v in blob["prep"].items():
            setattr(model, k, v)
        model.loss_kind = blob["prep"].get("loss_kind", "log_mse")
        if model.loss_kind == "log_mse":  # checkpoints from before target_transform existed
            model.loss_kind, model.target_transform = "mse", "log1p"
        model.target_transform = blob["prep"].get("target_transform", "log1p")
        model.quantile_levels = blob["prep"].get("quantile_levels", [])
        if model.family == ModelFamily.TORCH_GRU and "seq_covariates" not in blob["params"]:
            model.params["seq_covariates"] = False  # pre-covariate-channel GRU checkpoint
        model.net = model._build().to(device)
        model.net.load_state_dict(blob["state_dict"])
        return model  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        import torch

        torch.save(self._blob(), path)

    @classmethod
    def load_pt(cls, path: Path) -> TorchMLPModel:
        import torch

        return TorchMLPModel._from_blob(torch.load(path, map_location="cpu", weights_only=False))

    def __getstate__(self) -> dict[str, Any]:
        """Picklable (e.g. inside an ensemble): the locally-defined Net is stored as a torch blob."""
        import torch

        buf = io.BytesIO()
        torch.save(self._blob(), buf)
        return {"_torch_blob": buf.getvalue()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        import torch

        blob = torch.load(io.BytesIO(state["_torch_blob"]), map_location="cpu", weights_only=False)
        self.__dict__.update(TorchMLPModel._from_blob(blob).__dict__)


class TorchGRUModel(TorchMLPModel):
    family = ModelFamily.TORCH_GRU
    description = (
        "Sequence forecaster: a GRU reads the last `seq_len` days of the series (ending at the lag "
        "origin, scale-normalized) and is fused with the tabular embedding/MLP path."
    )
    default_params = {
        **TorchMLPModel.default_params,
        "seq_len": 56,
        "gru_hidden": 64,
        "gru_layers": 1,
        "hidden_sizes": [128, 64],
        "seq_covariates": True,  # feed open/promo history (spec.seq_covariates) as extra channels
    }
    search_space = {
        **TorchMLPModel.search_space,
        "gru_hidden": {"type": "categorical", "choices": [32, 64, 128]},
        "seq_len": {"type": "categorical", "choices": [28, 56]},
    }

    @classmethod
    def required_seq_len(cls, params: dict[str, Any]) -> int:
        return int({**cls.default_params, **params}["seq_len"])

    @property
    def input_columns(self) -> list[str]:
        return self.spec.model_columns

    @property
    def seq_channels(self) -> list[str]:
        if not self.params.get("seq_covariates", True):
            return []
        return list(getattr(self.spec, "seq_covariates", []) or [])

    def _inputs(self, X: pd.DataFrame) -> tuple[np.ndarray, ...]:
        num, cat = self._tabular(X)
        L = int(self.params["seq_len"])
        steps = list(reversed(range(L)))  # oldest -> newest
        seq = X[[f"seq_{k}" for k in steps]].to_numpy(dtype="float32")
        missing = np.isnan(seq)
        seq = np.where(missing, 0.0, np.clip(seq, 0, None))
        covs = [
            np.nan_to_num(X[[f"seqx_{c}_{k}" for k in steps]].to_numpy(dtype="float32"))
            for c in self.seq_channels
        ]
        # Per-row scale normalization; the level itself is in the tabular path (rolling means, lags).
        # With an open channel the scale averages open days only, so closures don't shrink it.
        open_col = self.spec.open_column
        if open_col in self.seq_channels:
            is_open = covs[self.seq_channels.index(open_col)] > 0
            scale = (seq * is_open).sum(axis=1, keepdims=True) / np.maximum(
                is_open.sum(axis=1, keepdims=True), 1
            ) + 1.0
        else:
            scale = seq.mean(axis=1, keepdims=True) + 1.0
        chans = [np.log1p(seq / scale), missing.astype("float32"), *covs]
        return num, cat, np.stack(chans, axis=2).astype("float32")

    def _encoder(self, nn: Any) -> tuple[Any, int]:
        hidden, layers = int(self.params["gru_hidden"]), int(self.params["gru_layers"])
        n_in = 2 + len(self.seq_channels)

        class SeqEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(input_size=n_in, hidden_size=hidden, num_layers=layers, batch_first=True)

            def forward(self, x: Any) -> Any:
                _, h = self.gru(x)
                return h[-1]

        return SeqEncoder(), hidden


# ----------------------------------------------------------------------------------------


class EnsembleModel(ForecastModel):
    family = ModelFamily.ENSEMBLE
    description = (
        "Weighted blend of member models (default LightGBM + GRU). Weights are fitted on the first half "
        "of the validation window; members and blend are compared on the untouched second half."
    )
    supports_gpu = True
    default_params = {
        "members": ["lightgbm", "torch_gru"],
        "member_params": {},
        "weights": "auto",  # auto | list of floats aligned with members
        "weight_metric": "wape",
    }

    @classmethod
    def is_available(cls) -> bool:
        return all(FAMILIES[ModelFamily(m)].is_available() for m in cls.default_params["members"])

    @classmethod
    def required_seq_len(cls, params: dict[str, Any]) -> int:
        p = {**cls.default_params, **params}
        return max(
            FAMILIES[ModelFamily(m)].required_seq_len(p["member_params"].get(m, {})) for m in p["members"]
        )

    @property
    def input_columns(self) -> list[str]:
        return self.spec.model_columns

    def fit(self, X, y, X_val, y_val, on_epoch) -> None:  # noqa: ANN001
        names: list[str] = list(self.params["members"])
        if len(names) < 2 or ModelFamily.ENSEMBLE.value in names:
            raise ValueError("ensemble needs >= 2 non-ensemble members")
        self.members: list[ForecastModel] = []
        val_preds = []
        for i, name in enumerate(names):
            fam = FAMILIES[ModelFamily(name)]
            member = fam(
                self.spec,
                self.params["member_params"].get(name, {}),
                device=self.device if fam.supports_gpu else "cpu",
            )

            def member_epoch(entry: dict[str, Any], frac: float, i: int = i, name: str = name) -> None:
                on_epoch({**entry, "member": name}, (i + frac) / len(names))

            member.fit(X, y, X_val, y_val, member_epoch)
            self.members.append(member)
            val_preds.append(np.clip(member.predict(X_val), 0, None))
        P = np.column_stack(val_preds)

        # Split the validation window by date: fit weights on A (earlier), compare on B (later).
        order = X_val["year"].to_numpy(dtype="int64") * 400 + X_val["dayofyear"].to_numpy(dtype="int64")
        in_a = order <= np.median(order)
        if self.params["weights"] == "auto":
            w = self._fit_weights(P[in_a], np.asarray(y_val)[in_a])
        else:
            w = np.asarray(self.params["weights"], dtype="float64")
            if len(w) != len(names) or w.min() < 0 or w.sum() <= 0:
                raise ValueError("weights must be non-negative and match members")
            w = w / w.sum()
        self.weights = w
        self.params["fitted_weights"] = dict(zip(names, map(float, w), strict=True))

        yb = np.asarray(y_val)[~in_a]
        metric = self.params["weight_metric"]
        comparison = {name: regression_metrics(yb, P[~in_a, i]) for i, name in enumerate(names)}
        comparison["equal_weight_blend"] = regression_metrics(yb, P[~in_a].mean(axis=1))
        comparison["fitted_blend"] = regression_metrics(yb, P[~in_a] @ w)
        self.comparison = {
            "weights": self.params["fitted_weights"],
            "weight_metric": metric,
            "weights_fitted_on": "validation first half",
            "compared_on": "validation second half",
            "holdout_rows": int((~in_a).sum()),
            "models": {
                k: {m: v[m] for m in ("wape", "rmse", "mae", "bias_pct", "r2")} for k, v in comparison.items()
            },
        }

    def _fit_weights(self, P: np.ndarray, y: np.ndarray) -> np.ndarray:
        metric = self.params["weight_metric"]

        def score(w: np.ndarray) -> float:
            v = regression_metrics(y, P @ w)[metric]
            return float("inf") if v is None else float(v)

        if P.shape[1] == 2:
            grid = [np.array([a, 1 - a]) for a in np.linspace(0, 1, 21)]
            return min(grid, key=score)
        from scipy.optimize import nnls

        w, _ = nnls(P, y)
        return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.clip(m.predict(X), 0, None) for m in self.members]) @ self.weights

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray] | None:
        """Weighted average over members that learn quantiles (the GBDT quantile boosters)."""
        parts = [(w, m.predict_quantiles(X)) for w, m in zip(self.weights, self.members, strict=True)]
        parts = [(w, q) for w, q in parts if q is not None and w > 0]
        if not parts:
            return None
        total = sum(w for w, _ in parts)
        return {lvl: sum(w * q[lvl] for w, q in parts) / total for lvl in parts[0][1]}

    def feature_importance(self) -> dict[str, float] | None:
        for m in self.members:
            fi = m.feature_importance()
            if fi:
                return fi
        return None


FAMILIES: dict[ModelFamily, type[ForecastModel]] = {
    ModelFamily.SEASONAL_NAIVE: SeasonalNaiveModel,
    ModelFamily.ARIMA: ARIMAModel,
    ModelFamily.SARIMA: SARIMAModel,
    ModelFamily.LIGHTGBM: LightGBMModel,
    ModelFamily.TORCH_MLP: TorchMLPModel,
    ModelFamily.TORCH_GRU: TorchGRUModel,
    ModelFamily.ENSEMBLE: EnsembleModel,
}


def resolve_device(requested: str, family: type[ForecastModel]) -> str:
    if not family.supports_gpu or requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if requested == "cuda":
        raise RuntimeError("TRAINING_DEVICE=cuda but CUDA is not available on this worker")
    return "cpu"


def suggest_params(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, cfg in space.items():
        t = cfg["type"]
        if t == "int":
            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"], log=cfg.get("log", False))
        elif t == "float":
            params[name] = trial.suggest_float(name, cfg["low"], cfg["high"], log=cfg.get("log", False))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
        else:
            raise ValueError(f"Unknown search-space type {t!r} for {name}")
    return params
