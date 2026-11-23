"""Run the full ML pipeline locally (no Postgres/Redis) on a bundled public dataset.

Uses exactly the production code paths (ingestion, joins, diagnostics, features, model families,
forecasting); only the database/queue plumbing is bypassed.

    python -m scripts.run_pipeline rossmann                       # all 1,115 stores
    python -m scripts.run_pipeline rossmann --stores 1 2 3
    python -m scripts.run_pipeline m5 --stores CA_1               # one store, 3,049 items
    python -m scripts.run_pipeline m5 --stores CA_1 --families lightgbm --history-days 730

Data is read from ../data/raw (git-ignored). Writes a JSON summary to --out if given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("JWT_SECRET_KEY", "local-pipeline-runner-not-used-for-auth-000")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml.families import FAMILIES, ForecastModel  # noqa: E402
from app.ml.forecasting import run_forecast  # noqa: E402
from app.ml.training import (  # noqa: E402
    PreparedData,
    _fit_once,
    evaluate_model,
    inner_split,
    prepare_data,
)
from app.models.enums import ModelFamily  # noqa: E402
from app.schemas.dataset import JoinSpec, SchemaMapping  # noqa: E402
from app.services.diagnostics import compute_health_report  # noqa: E402
from app.services.ingestion import coerce_covariates, convert_to_parquet  # noqa: E402
from app.services.joins import apply_joins, load_join  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "raw"


class ConsoleProgress:
    def __init__(self) -> None:
        self._last = ""

    def update(self, fraction: float, stage: str, message: str | None = None, force: bool = False) -> None:
        line = f"    {stage:<24} {fraction * 100:5.1f}%  {message or ''}"
        if stage != self._last or force:
            print(line, flush=True)
            self._last = stage

    def sub(self, lo: float, hi: float) -> ConsoleProgress:
        return self


# ----------------------------------------------------------------------------------------
# Honest tuning: Optuna sees only an inner split of the *training* period
# ----------------------------------------------------------------------------------------

TUNE_SPACE = {
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.1, "log": True},
    "num_leaves": {"type": "int", "low": 31, "high": 511, "log": True},
    "min_data_in_leaf": {"type": "int", "low": 10, "high": 300, "log": True},
    "feature_fraction": {"type": "float", "low": 0.4, "high": 1.0},
    "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
    "lambda_l2": {"type": "float", "low": 1e-3, "high": 30.0, "log": True},
    "min_gain_to_split": {"type": "float", "low": 0.0, "high": 1.0},
}


TORCH_TUNE_SPACE = {
    "lr": {"type": "float", "low": 3e-4, "high": 3e-3, "log": True},
    "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    "weight_decay": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [1024, 2048, 4096]},
    "embedding_dim": {"type": "categorical", "choices": [8, 16]},
    "target_transform": {"type": "categorical", "choices": ["log1p", "standard"]},
}


def tune_family(
    family: ModelFamily,
    data: PreparedData,
    base: dict[str, Any],
    n_trials: int,
    days: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Optuna on an inner split of the training period; the final model gets a fixed budget (GBDT
    rounds scaled to the full training set, NN epochs) and no early stopping on the final window."""
    import optuna

    from app.ml.families import suggest_params

    inner = inner_split(data, days)
    print(
        f"    tuning {family.value} on inner split: {len(inner.X_tr):,} train / {len(inner.X_val):,} val"
        f" rows, {n_trials} trials"
    )
    is_gbdt = family == ModelFamily.LIGHTGBM
    space = TUNE_SPACE if is_gbdt else TORCH_TUNE_SPACE
    fixed = {"quantiles": False, "early_stopping_rounds": 100} if is_gbdt else {"quantiles": False}
    budget: dict[int, int] = {}

    def objective(trial: optuna.Trial) -> float:
        params = {**base, **suggest_params(trial, space), **fixed}
        t0 = time.perf_counter()
        model, _, preds = _fit_once(family, inner, params, device, None)
        budget[trial.number] = int(model.params["best_iteration" if is_gbdt else "best_epoch"])
        wape = inner.metrics(preds)["wape"]
        print(
            f"      trial {trial.number:>2}: inner ACC {100 * (1 - wape):.2f}%"
            f"  {'rounds' if is_gbdt else 'epochs'} {budget[trial.number]}  {time.perf_counter() - t0:.0f}s  "
            f"{ {k: round(v, 4) if isinstance(v, float) else v for k, v in trial.params.items()} }",
            flush=True,
        )
        return float(wape)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    n = budget[study.best_trial.number]
    if is_gbdt:
        n = int(round(n * (len(inner.X_tr) + len(inner.X_val)) / len(inner.X_tr)))
        final = {"num_boost_round": n, "early_stopping_rounds": 0}
    else:
        final = {"max_epochs": n, "patience": 0}
    print(f"    best inner ACC {100 * (1 - study.best_value):.2f}% -> final model {final}")
    return {**base, **study.best_params, **final}


def parse_kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kv in items:
        key, _, val = kv.partition("=")
        try:
            out[key] = json.loads(val)
        except json.JSONDecodeError:
            out[key] = val
    return out


# ----------------------------------------------------------------------------------------
# Dataset presets
# ----------------------------------------------------------------------------------------


STORE_CSV_RENAME = {
    "StoreType": "store_type",
    "Assortment": "assortment",
    "CompetitionDistance": "competition_distance",
    "CompetitionOpenSinceMonth": "competition_open_since_month",
    "CompetitionOpenSinceYear": "competition_open_since_year",
    "Promo2": "promo2",
    "Promo2SinceWeek": "promo2_since_week",
    "Promo2SinceYear": "promo2_since_year",
    "PromoInterval": "promo_interval",
}


def load_store_static(path: Path) -> dict[str, dict[str, Any]]:
    """Store metadata CSV (Rossmann store.csv columns or Store field names) -> {code: attributes}."""
    meta = pd.read_csv(path, dtype={"Store": str, "code": str}).rename(columns=STORE_CSV_RENAME)
    meta = meta.rename(columns={"Store": "code"}).set_index("code")
    if "promo2" in meta:
        meta["promo2"] = meta["promo2"].fillna(0).astype(bool)
    return {str(k): {f: (None if pd.isna(v) else v) for f, v in row.items()} for k, row in meta.iterrows()}


def _first_dir(data_dir: Path, *names: str) -> Path:
    for n in names:
        if (data_dir / n).is_dir():
            return data_dir / n
    raise SystemExit(f"none of {names} found under {data_dir} (pass --data-dir)")


def rossmann(stores: list[str] | None, data_dir: Path = DATA) -> dict[str, Any]:
    root = _first_dir(data_dir, "rossman dataset", "rossmann", "rossmann-store-sales")
    mapping = SchemaMapping(
        date_column="Date",
        target_column="Sales",
        store_column="Store",
        customers_column="Customers",
        known_future_columns=["Open", "Promo", "SchoolHoliday"],
        categorical_columns=["StateHoliday"],
        promo_column="Promo",
        open_column="Open",
        store_filter=stores,
    )
    static = load_store_static(root / "store.csv") if (root / "store.csv").exists() else {}
    return {"raw": root / "train.csv", "mapping": mapping, "aux": {}, "store_static": static, "horizon": 42}


def m5(stores: list[str] | None, data_dir: Path = DATA) -> dict[str, Any]:
    root = _first_dir(data_dir, "m5", "m5-forecasting-accuracy")
    cal_id, price_id = uuid.uuid4(), uuid.uuid4()
    mapping = SchemaMapping(
        layout="wide",
        wide_value_prefix="d_",
        wide_start_date="2011-01-29",
        store_column="store_id",
        item_column="item_id",
        id_columns=["dept_id", "cat_id", "state_id"],
        categorical_columns=["dept_id", "cat_id", "state_id", "event_name_1", "event_type_1"],
        known_future_columns=["snap_CA", "snap_TX", "snap_WI", "sell_price"],
        helper_columns=["wm_yr_wk"],
        price_column="sell_price",
        store_filter=stores,
        joins=[
            JoinSpec(
                dataset_id=cal_id,
                left_on=["ds"],
                right_on=["date"],
                columns=["wm_yr_wk", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"],
            ),
            JoinSpec(
                dataset_id=price_id,
                left_on=["store_code", "item_id", "wm_yr_wk"],
                right_on=["store_id", "item_id", "wm_yr_wk"],
                columns=["sell_price"],
            ),
        ],
    )
    aux = {cal_id: root / "calendar.csv", price_id: root / "sell_prices.csv"}
    return {
        "raw": root / "sales_train_evaluation.csv",
        "mapping": mapping,
        "aux": aux,
        "store_static": {},
        "horizon": 28,
    }


PRESETS = {"rossmann": rossmann, "m5": m5}


# ----------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=PRESETS)
    ap.add_argument("--stores", nargs="*", help="store codes to include (default: all)")
    ap.add_argument(
        "--families",
        nargs="*",
        default=[f.value for f in ModelFamily if f not in (ModelFamily.ARIMA, ModelFamily.SARIMA)],
        choices=[f.value for f in ModelFamily],
    )
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--validation-days", type=int, help="validation window (default = horizon)")
    ap.add_argument("--history-days", type=int)
    ap.add_argument("--no-store-features", action="store_true")
    ap.add_argument("--lgb-rounds", type=int, default=1000)
    ap.add_argument("--torch-epochs", type=int, default=8)
    ap.add_argument("--lgb-objective", default="auto", choices=["auto", "regression", "tweedie", "poisson"])
    ap.add_argument("--quantiles", default="auto", choices=["auto", "true", "false"])
    ap.add_argument(
        "--lgb-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help='extra LightGBM param, JSON-parsed value (repeatable), e.g. target_transform="log1p"',
    )
    ap.add_argument(
        "--tune",
        type=int,
        default=0,
        metavar="N",
        help="Optuna-tune LightGBM / torch families for N trials on an inner split of the training period",
    )
    ap.add_argument(
        "--torch-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help='extra torch_mlp/torch_gru param, JSON-parsed (repeatable), e.g. target_transform="standard"',
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = PRESETS[args.dataset](args.stores)
    mapping: SchemaMapping = cfg["mapping"]
    horizon = args.horizon or cfg["horizon"]
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "stores": args.stores,
        "horizon": horizon,
        "models": {},
    }
    progress = ConsoleProgress()

    with tempfile.TemporaryDirectory() as tmp:
        canon = Path(tmp) / "canonical.parquet"
        print(f"[1/4] ingest {cfg['raw'].name}")
        t0 = time.perf_counter()
        joins = [load_join(j, cfg["aux"][j.dataset_id], "csv") for j in mapping.joins]
        stats = convert_to_parquet(
            cfg["raw"], "csv", mapping, canon, 500_000, lambda f, s: progress.update(f, s), joins=joins
        )
        summary["ingest"] = {k: v for k, v in stats.items() if k != "store_codes"} | {
            "store_count": len(stats["store_codes"]),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(
            f"    {stats['row_count']:,} rows, {stats['series_count']:,} series "
            f"in {summary['ingest']['seconds']}s"
        )

        print("[2/4] diagnostics")
        report = compute_health_report(canon)
        summary["health"] = {"score": report["score"], "issues": [i["code"] for i in report["issues"]]}
        print(f"    score {report['score']}/100 issues={summary['health']['issues']}")

        print("[3/4] features")
        t0 = time.perf_counter()
        data = prepare_data(
            canon,
            mapping.model_dump(mode="json"),
            horizon,
            args.validation_days or horizon,
            None,
            history_days=args.history_days,
            store_static=None if args.no_store_features else cfg["store_static"],
            seq_len=max(FAMILIES[ModelFamily(f)].required_seq_len({}) for f in args.families),
        )
        summary["features"] = {
            "train_rows": len(data.X_tr),
            "val_rows": len(data.X_val),
            "n_features": len(data.spec.feature_columns),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(
            f"    {len(data.X_tr):,} train / {len(data.X_val):,} val rows, "
            f"{len(data.spec.feature_columns)} features in {summary['features']['seconds']}s"
        )

        def enrich(frame: pd.DataFrame) -> pd.DataFrame:
            frame = apply_joins(frame, joins) if joins else frame
            return coerce_covariates(frame.drop(columns=mapping.helper_columns, errors="ignore"), mapping)

        print("[4/4] train + forecast")
        for fam in args.families:
            family = ModelFamily(fam)
            if not FAMILIES[family].is_available():
                print(f"    {fam}: not installed, skipped")
                continue
            quantiles = {"auto": "auto", "true": True, "false": False}[args.quantiles]
            lgb_params = {
                "num_boost_round": args.lgb_rounds,
                "objective": args.lgb_objective,
                "quantiles": quantiles,
            }
            lgb_params |= parse_kv(args.lgb_param)
            nn_params: dict[str, Any] = {"max_epochs": args.torch_epochs, "quantiles": quantiles}
            nn_params |= parse_kv(args.torch_param)
            tune_days = args.validation_days or horizon
            if args.tune and family == ModelFamily.LIGHTGBM:
                lgb_params = tune_family(family, data, lgb_params, args.tune, tune_days)
                summary.setdefault("tuned_params", {})[fam] = lgb_params
            elif args.tune and family in (ModelFamily.TORCH_MLP, ModelFamily.TORCH_GRU):
                nn_params = tune_family(family, data, nn_params, args.tune, tune_days)
                summary.setdefault("tuned_params", {})[fam] = nn_params
            params: dict[str, Any] = {
                ModelFamily.LIGHTGBM: lgb_params,
                ModelFamily.TORCH_MLP: nn_params,
                ModelFamily.TORCH_GRU: nn_params,
                ModelFamily.ENSEMBLE: {"member_params": {"lightgbm": lgb_params, "torch_gru": nn_params}},
            }.get(family, {})
            t0 = time.perf_counter()
            model, logs, preds = _fit_once(family, data, params, "cpu", progress)
            fit_s = time.perf_counter() - t0
            m = evaluate_model(model, data, preds)
            interval = m.get("interval")
            path = Path(tmp) / f"model.{model.artifact_ext}"
            model.save(path)
            loaded = ForecastModel.load(path)
            assert np.allclose(
                loaded.predict(data.X_val.head(500)), model.predict(data.X_val.head(500)), atol=1e-4
            )
            t1 = time.perf_counter()
            out, fsum = run_forecast(
                model=loaded,
                metrics=m,
                parquet_path=canon,
                store_code=None,
                horizon=horizon,
                covariate_defaults={},
                events=[],
                compare_to_baseline=False,
                promo_column=mapping.promo_column,
                open_column=mapping.open_column,
                price_column=mapping.price_column,
                enrich=enrich if joins else None,
                progress=progress,
            )
            entry = {
                "accuracy": 1 - m["wape"],
                **{k: m[k] for k in ("wape", "rmse", "mae", "rmsse", "r2", "rmspe", "bias_pct")},
                "params": {
                    k: v
                    for k, v in model.params.items()
                    if k in ("resolved_objective", "fitted_weights", "best_epoch", "best_iteration")
                }
                | {
                    k: getattr(model, k)
                    for k in ("loss_kind", "target_transform", "quantile_levels")
                    if hasattr(model, k)
                },
                "interval": interval,
                "ensemble": m.get("ensemble"),
                "fit_seconds": round(fit_s, 1),
                "forecast_seconds": round(time.perf_counter() - t1, 1),
                "forecast_rows": len(out),
                "forecast_window": [fsum["start_date"], fsum["end_date"]],
                "forecast_total": fsum["horizon_totals"][f"h{horizon}"]["yhat"],
                "future_covariate_nulls": {
                    c: int(out[c].isna().sum()) for c in ("price",) if c in out and mapping.price_column
                },
            }
            if family in (ModelFamily.LIGHTGBM, ModelFamily.ENSEMBLE):
                fi = model.feature_importance() or {}
                entry["top_features"] = [k for k, _ in sorted(fi.items(), key=lambda kv: -kv[1])[:10]]
            summary["models"][fam] = entry

            def fmt(v: Any) -> str:
                return "n/a" if v is None else f"{v:.4f}"

            iv = interval or {}
            pinball = (
                (iv["pinball_p10"] + iv["pinball_p50"] + iv["pinball_p90"]) / 3
                if "pinball_p50" in iv
                else None
            )
            print(
                f"    {fam:<15} ACC {100 * (1 - m['wape']):.2f}%  WAPE {fmt(m['wape'])}"
                f"  RMSPE {fmt(m['rmspe'])}  RMSSE {fmt(m['rmsse'])}"
                f"  RMSE {m['rmse']:.1f}  MAE {m['mae']:.1f}  R2 {fmt(m['r2'])}  fit {fit_s:.0f}s"
            )
            print(
                f"    {'':<15} interval {iv.get('method', 'n/a')}: P10-P90 cov {fmt(iv.get('coverage'))}"
                f" (holdout {fmt(iv.get('coverage_holdout'))})  mean pinball {fmt(pinball)}"
            )
            if m.get("ensemble"):
                for name, mm in m["ensemble"]["models"].items():
                    print(f"      holdout {name:<20} WAPE {fmt(mm['wape'])}  RMSE {fmt(mm['rmse'])}")
                print(f"      weights {m['ensemble']['weights']}")

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"summary -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
