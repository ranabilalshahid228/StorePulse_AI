"""Model families. Each implements fit / predict / save / load over the shared feature frame.

Add a family: subclass `ForecastModel`, register it in `FAMILIES`, add a default search space,
and add the enum value in `app.models.enums.ModelFamily`.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd

from app.ml.features import FeatureSpec
from app.ml.metrics import regression_metrics
from app.models.enums import ModelFamily

logger = logging.getLogger(__name__)

EpochCallback = Callable[[dict[str, Any], float], None]  # (log_entry, fraction_done)


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

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, on_epoch: EpochCallback
    ) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

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
        on_epoch(
            {
                "epoch": 1,
                **{f"val_{k}": v for k, v in regression_metrics(y_val, self.predict(X_val)).items()},
            },
            1.0,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(X[self._cols()].mean(axis=1).to_numpy(dtype="float64"), nan=0.0)


# ----------------------------------------------------------------------------------------


class LightGBMModel(ForecastModel):
    family = ModelFamily.LIGHTGBM
    description = "Gradient-boosted trees (LightGBM) on calendar, lag, rolling and covariate features."
    default_params = {
        "objective": "regression",
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
    }

    def fit(self, X, y, X_val, y_val, on_epoch) -> None:  # noqa: ANN001
        import lightgbm as lgb

        params = dict(self.params)
        rounds = int(params.pop("num_boost_round"))
        es = int(params.pop("early_stopping_rounds"))
        params.update({"verbosity": -1, "metric": ["l1", "l2"], "seed": 42})
        cats = [c for c in self.spec.categorical_features if c in X.columns]
        dtrain = lgb.Dataset(X, y, categorical_feature=cats, free_raw_data=False)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)

        def _log(env: Any) -> None:
            it = env.iteration + 1
            if it % 25 == 0 or it == 1:
                entry = {"epoch": it}
                for _, name, value, _ in env.evaluation_result_list:
                    entry[f"val_{name}"] = float(value)
                on_epoch(entry, it / rounds)

        self.booster = lgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(es, verbose=False), _log],
        )
        self.params["best_iteration"] = int(self.booster.best_iteration or rounds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

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
        "patience": 5,
        "embedding_dim": 8,
    }
    search_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "weight_decay": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [1024, 2048, 4096, 8192]},
        "embedding_dim": {"type": "categorical", "choices": [4, 8, 16]},
    }

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("torch") is not None

    # -- tensor prep -----------------------------------------------------------------------
    def _split(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cats = [c for c in self.spec.categorical_features if c in X.columns]
        nums = [c for c in X.columns if c not in cats]
        if not hasattr(self, "num_cols"):
            self.cat_cols, self.num_cols = cats, nums
            num = X[nums].to_numpy(dtype="float32")
            self.num_fill = np.nanmedian(num, axis=0)
            self.num_fill = np.where(np.isnan(self.num_fill), 0.0, self.num_fill).astype("float32")
            filled = np.where(np.isnan(num), self.num_fill, num)
            self.num_mean = filled.mean(axis=0)
            self.num_std = filled.std(axis=0) + 1e-6
        num = X[self.num_cols].to_numpy(dtype="float32")
        num = (np.where(np.isnan(num), self.num_fill, num) - self.num_mean) / self.num_std
        cat = np.stack([X[c].cat.codes.to_numpy() + 1 for c in self.cat_cols], axis=1).astype("int64")
        return num.astype("float32"), cat  # code 0 == unknown/missing

    def _build(self) -> Any:
        import torch
        from torch import nn

        cards = [len(self.spec.category_levels.get(c, [])) + 1 for c in self.cat_cols]
        emb_dim = int(self.params["embedding_dim"])
        hidden = list(self.params["hidden_sizes"])
        dropout = float(self.params["dropout"])
        n_num = len(self.num_cols)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embs = nn.ModuleList([nn.Embedding(c, min(emb_dim, c)) for c in cards])
                in_dim = n_num + sum(min(emb_dim, c) for c in cards)
                layers: list[nn.Module] = []
                for h in hidden:
                    layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
                    in_dim = h
                layers.append(nn.Linear(in_dim, 1))
                self.mlp = nn.Sequential(*layers)

            def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
                embs = [e(cat[:, i]) for i, e in enumerate(self.embs)]
                return self.mlp(torch.cat([num, *embs], dim=1)).squeeze(1)

        return Net()

    def fit(self, X, y, X_val, y_val, on_epoch) -> None:  # noqa: ANN001
        import torch

        torch.manual_seed(42)
        dev = torch.device(self.device)
        num, cat = self._split(X)
        vnum, vcat = self._split(X_val)
        # Train in log1p space for scale robustness across stores.
        yt = torch.tensor(np.log1p(np.clip(y, 0, None)), dtype=torch.float32)
        self.net = self._build().to(dev)
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]
        )
        loss_fn = torch.nn.MSELoss()
        tn, tc = torch.tensor(num), torch.tensor(cat)
        bs = int(self.params["batch_size"])
        max_epochs = int(self.params["max_epochs"])
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            self.net.train()
            perm = torch.randperm(len(yt))
            total = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                pred = self.net(tn[idx].to(dev), tc[idx].to(dev))
                loss = loss_fn(pred, yt[idx].to(dev))
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            val_pred = self._predict_arrays(vnum, vcat)
            m = regression_metrics(y_val, val_pred)
            entry = {
                "epoch": epoch,
                "train_loss": total / len(yt),
                "val_mae": m["mae"],
                "val_wape": m["wape"],
                "seconds": round(time.perf_counter() - t0, 3),
            }
            on_epoch(entry, epoch / max_epochs)
            score = m["mae"] if m["mae"] is not None else float("inf")
            if score < best:
                best, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= int(self.params["patience"]):
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _predict_arrays(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        import torch

        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(num), 65536):
                p = self.net(
                    torch.tensor(num[i : i + 65536]).to(dev), torch.tensor(cat[i : i + 65536]).to(dev)
                )
                out.append(p.cpu().numpy())
        return np.expm1(np.concatenate(out)).clip(min=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_arrays(*self._split(X))

    def save(self, path: Path) -> None:
        import torch

        torch.save(
            {
                "family": self.family.value,
                "spec": self.spec.to_dict(),
                "params": self.params,
                "state_dict": self.net.state_dict(),
                "prep": {
                    "cat_cols": self.cat_cols,
                    "num_cols": self.num_cols,
                    "num_fill": self.num_fill,
                    "num_mean": self.num_mean,
                    "num_std": self.num_std,
                },
            },
            path,
        )

    @classmethod
    def load_pt(cls, path: Path) -> TorchMLPModel:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        blob = torch.load(path, map_location=device, weights_only=False)
        model = cls(FeatureSpec(**blob["spec"]), blob["params"], device=device)
        for k, v in blob["prep"].items():
            setattr(model, k, v)
        model.net = model._build().to(device)
        model.net.load_state_dict(blob["state_dict"])
        return model


FAMILIES: dict[ModelFamily, type[ForecastModel]] = {
    ModelFamily.SEASONAL_NAIVE: SeasonalNaiveModel,
    ModelFamily.LIGHTGBM: LightGBMModel,
    ModelFamily.TORCH_MLP: TorchMLPModel,
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
