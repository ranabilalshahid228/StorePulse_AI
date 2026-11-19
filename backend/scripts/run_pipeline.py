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
from app.ml.training import _fit_once, evaluate_model, prepare_data  # noqa: E402
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
# Dataset presets
# ----------------------------------------------------------------------------------------


def rossmann(stores: list[str] | None) -> dict[str, Any]:
    root = DATA / "rossman dataset"
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
    meta = pd.read_csv(root / "store.csv", dtype={"Store": str})
    rename = {
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
    meta = meta.rename(columns=rename).set_index("Store")
    meta["promo2"] = meta["promo2"].astype(bool)
    static = {k: {f: (None if pd.isna(v) else v) for f, v in row.items()} for k, row in meta.iterrows()}
    return {"raw": root / "train.csv", "mapping": mapping, "aux": {}, "store_static": static, "horizon": 42}


def m5(stores: list[str] | None) -> dict[str, Any]:
    root = DATA / "m5"
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
        default=[f.value for f in ModelFamily],
        choices=[f.value for f in ModelFamily],
    )
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--history-days", type=int)
    ap.add_argument("--no-store-features", action="store_true")
    ap.add_argument("--lgb-rounds", type=int, default=1000)
    ap.add_argument("--torch-epochs", type=int, default=8)
    ap.add_argument("--lgb-objective", default="auto", choices=["auto", "regression", "tweedie", "poisson"])
    ap.add_argument("--quantiles", default="auto", choices=["auto", "true", "false"])
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
            horizon,
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
            nn_params = {"max_epochs": args.torch_epochs}
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
                **{k: m[k] for k in ("wape", "rmse", "mae", "rmsse", "r2", "rmspe", "bias_pct")},
                "params": {
                    k: v for k, v in model.params.items() if k in ("resolved_objective", "fitted_weights")
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

            cov = fmt(interval.get("coverage")) if interval else "n/a"
            print(
                f"    {fam:<15} WAPE {fmt(m['wape'])}  RMSSE {fmt(m['rmsse'])}  RMSE {fmt(m['rmse'])}  "
                f"R2 {fmt(m['r2'])}  RMSPE {fmt(m['rmspe'])}  P10-P90 cov {cov}  fit {fit_s:.0f}s"
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
