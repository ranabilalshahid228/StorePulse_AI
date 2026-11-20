"""Standalone training / tuning / evaluation / forecasting CLI for GPU (or CPU) machines.

No web server, database, Redis or Docker: it runs the production ingestion, feature, model and
forecasting code directly and writes everything for a run into one self-contained folder you can
copy back (outputs/ is git-ignored):

    outputs/<run>/
      run.json                     args, git commit, host, python/torch/CUDA versions, timings, leaderboard
      leaderboard.csv              one row per family: accuracy (1 - WAPE), WAPE, RMSPE, RMSSE, ...
      train.log                    console output
      <family>/
        model.pkl | model.pt       checkpoint (loads with app.ml.families.ForecastModel.load)
        config.json                everything needed to evaluate / forecast again (data, mapping, horizon)
        metrics.json               validation metrics incl. intervals, residuals, feature importance
        training_log.json          per-epoch / per-iteration logs
        validation.parquet         ds, store_code, item_id, y, yhat (+ P10/P90 when available)
        forecast.parquet/.csv      next-horizon forecast after the data (unless --no-forecast)
    outputs/_cache/                canonical Parquet per dataset + mapping (re-used across runs)

Examples (from backend/):

    python -m scripts.train_cli train --dataset rossmann --families lightgbm torch_gru ensemble \\
        --horizon 42 --epochs 20 --device cuda --output-dir ../outputs --run-name rossmann_h42
    python -m scripts.train_cli train --dataset rossmann --families torch_mlp --tune 20 --device cuda
    python -m scripts.train_cli train --dataset m5 --stores CA_1 --history-days 730 --families lightgbm
    python -m scripts.train_cli train --dataset custom --train-file sales.csv --mapping-file mapping.json
    python -m scripts.train_cli evaluate --model-dir ../outputs/rossmann_h42/torch_gru
    python -m scripts.train_cli predict --model-dir ../outputs/rossmann_h42/lightgbm --horizon 28 --format csv
    python -m scripts.train_cli list --output-dir ../outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("JWT_SECRET_KEY", "standalone-train-cli-not-used-for-auth-0000")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml.families import FAMILIES, ForecastModel  # noqa: E402
from app.ml.forecasting import run_forecast  # noqa: E402
from app.ml.training import _fit_once, evaluate_model, prepare_data  # noqa: E402
from app.models.enums import ModelFamily  # noqa: E402
from app.schemas.dataset import SchemaMapping  # noqa: E402
from app.services.ingestion import coerce_covariates, convert_to_parquet  # noqa: E402
from app.services.joins import apply_joins, load_join  # noqa: E402
from scripts.run_pipeline import (  # noqa: E402
    DATA,
    PRESETS,
    ConsoleProgress,
    load_store_static,
    parse_kv,
    tune_family,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs"
FAMILY_NAMES = [f.value for f in ModelFamily]


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


class Tee:
    """Mirror stdout into the run's train.log."""

    def __init__(self, path: Path) -> None:
        self.fh = path.open("a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s: str) -> int:
        self.fh.write(s)
        self.fh.flush()
        return self.out.write(s)

    def flush(self) -> None:
        self.fh.flush()
        self.out.flush()


def _json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def resolve_device(requested: str, family: ModelFamily) -> str:
    """auto -> CUDA for GPU-capable families when available; cuda fails fast if it isn't."""
    if not FAMILIES[family].supports_gpu or requested == "cpu":
        return "cpu"
    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if available:
        return "cuda"
    if requested == "cuda":
        raise SystemExit(
            "--device cuda but CUDA is not available (install the CUDA torch wheel, check nvidia-smi / "
            "CUDA_VISIBLE_DEVICES, or use --device cpu)"
        )
    return "cpu"


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "started_at": datetime.now(UTC).isoformat(),
    }
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["git_commit"] = None
    try:
        import lightgbm

        env["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        env["torch"] = None
    return env


def dataset_config(args: argparse.Namespace) -> dict[str, Any]:
    """Raw file, mapping, auxiliary tables and store metadata for --dataset."""
    data_dir = Path(args.data_dir).resolve()
    if args.dataset in PRESETS:
        cfg = PRESETS[args.dataset](args.stores, data_dir)
        aux = {str(k): str(v) for k, v in cfg["aux"].items()}
        return {
            "name": args.dataset,
            "raw": str(cfg["raw"]),
            "mapping": cfg["mapping"].model_dump(mode="json"),
            "aux": aux,
            "store_static": cfg["store_static"],
            "default_horizon": cfg["horizon"],
        }
    if not args.train_file or not args.mapping_file:
        raise SystemExit("--dataset custom needs --train-file and --mapping-file (a SchemaMapping JSON)")
    mapping = SchemaMapping.model_validate(json.loads(Path(args.mapping_file).read_text(encoding="utf-8")))
    if args.stores:
        mapping.store_filter = args.stores
    aux = dict(kv.split("=", 1) for kv in args.aux)
    missing = {str(j.dataset_id) for j in mapping.joins} - set(aux)
    if missing:
        raise SystemExit(f"mapping joins need --aux <dataset_id>=<file> for: {sorted(missing)}")
    return {
        "name": Path(args.train_file).stem,
        "raw": str(Path(args.train_file).resolve()),
        "mapping": mapping.model_dump(mode="json"),
        "aux": {k: str(Path(v).resolve()) for k, v in aux.items()},
        "store_static": load_store_static(Path(args.store_file)) if args.store_file else {},
        "default_horizon": 28,
    }


def _fmt(path: str) -> str:
    return "parquet" if path.lower().endswith(".parquet") else "csv"


def load_joins(ds: dict[str, Any]) -> list:
    mapping = SchemaMapping.model_validate(ds["mapping"])
    return [
        load_join(j, Path(ds["aux"][str(j.dataset_id)]), _fmt(ds["aux"][str(j.dataset_id)]))
        for j in mapping.joins
    ]


def canonical(ds: dict[str, Any], cache_dir: Path, progress: ConsoleProgress) -> Path:
    """Ingest the raw file once per (file, mapping); later runs reuse the cached Parquet."""
    raw = Path(ds["raw"])
    if not raw.exists():
        raise SystemExit(f"data file not found: {raw} (check --data-dir / --train-file)")
    st = raw.stat()
    key = hashlib.sha1(
        json.dumps(
            [str(raw), st.st_size, int(st.st_mtime), ds["mapping"], ds["aux"]], sort_keys=True
        ).encode()
    ).hexdigest()[:12]
    out = cache_dir / f"{ds['name']}-{key}.parquet"
    if out.exists():
        print(f"[data] cached canonical parquet {out.name}")
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] ingesting {raw.name} -> {out.name}")
    t0 = time.perf_counter()
    tmp = out.with_suffix(".tmp.parquet")
    stats = convert_to_parquet(
        raw,
        _fmt(str(raw)),
        SchemaMapping.model_validate(ds["mapping"]),
        tmp,
        500_000,
        lambda f, s: progress.update(f, s),
        joins=load_joins(ds),
    )
    tmp.replace(out)
    print(
        f"[data] {stats['row_count']:,} rows, {stats['series_count']:,} series in"
        f" {time.perf_counter() - t0:.0f}s"
    )
    return out


def family_params(family: ModelFamily, args: argparse.Namespace) -> dict[str, Any]:
    quantiles = {"auto": "auto", "true": True, "false": False}[args.quantiles]
    lgb = {"num_boost_round": args.lgb_rounds, "quantiles": quantiles} | parse_kv(args.lgb_param)
    nn: dict[str, Any] = {"max_epochs": args.epochs, "quantiles": quantiles} | parse_kv(args.torch_param)
    if args.patience is not None:
        nn["patience"] = args.patience
    if family == ModelFamily.LIGHTGBM:
        return lgb
    if family in (ModelFamily.TORCH_MLP, ModelFamily.TORCH_GRU):
        return nn
    if family == ModelFamily.ENSEMBLE:
        members = args.ensemble_members
        member_params = {m: (lgb if m == "lightgbm" else nn) for m in members if m != "seasonal_naive"}
        return {"members": members, "member_params": member_params}
    return {}


def enrich_fn(ds: dict[str, Any]):  # noqa: ANN201
    mapping = SchemaMapping.model_validate(ds["mapping"])
    if not mapping.joins:
        return None
    joins = load_joins(ds)

    def enrich(frame: pd.DataFrame) -> pd.DataFrame:
        frame = apply_joins(frame, joins)
        return coerce_covariates(frame.drop(columns=mapping.helper_columns, errors="ignore"), mapping)

    return enrich


def write_forecast(
    model: ForecastModel, metrics: dict, cfg: dict, horizon: int, out_dir: Path, fmt: str
) -> Path:
    ds = cfg["dataset"]
    mapping = ds["mapping"]
    out, summary = run_forecast(
        model=model,
        metrics=metrics,
        parquet_path=Path(cfg["canonical"]),
        store_code=None,
        horizon=horizon,
        covariate_defaults=cfg.get("covariate_defaults", {}),
        events=[],
        compare_to_baseline=False,
        promo_column=mapping.get("promo_column"),
        open_column=mapping.get("open_column"),
        price_column=mapping.get("price_column"),
        enrich=enrich_fn(ds),
        progress=ConsoleProgress(),
    )
    path = out_dir / f"forecast.{fmt}"
    out.to_parquet(path, index=False) if fmt == "parquet" else out.to_csv(path, index=False)
    _json(summary, out_dir / "forecast_summary.json")
    return path


def _validation_frame(data: Any, model: ForecastModel, preds: np.ndarray) -> pd.DataFrame:
    frame = data.val_frame.copy()
    frame["yhat"] = preds
    q = model.predict_quantiles(data.X_val)
    if q:
        frame["yhat_p10"], frame["yhat_p90"] = q[min(q)], q[max(q)]
    return frame


# ----------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    families = [ModelFamily(f) for f in args.families]
    for f in families:
        if not FAMILIES[f].is_available():
            raise SystemExit(f"{f.value} is not installed (pip install torch for neural families)")
    devices = {f: resolve_device(args.device, f) for f in families}  # fails fast on --device cuda
    out_root = Path(args.output_dir).resolve()
    run_name = args.run_name or f"{args.dataset}_h{args.horizon or 'default'}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = out_root / run_name
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} exists (use --run-name or --overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(run_dir / "train.log")  # type: ignore[assignment]

    env = environment()
    print(
        f"[run] {run_dir}\n[env]"
        f" {json.dumps({k: env.get(k) for k in ('host', 'torch', 'gpus', 'git_commit')})}"
    )
    progress = ConsoleProgress()
    ds = dataset_config(args)
    horizon = args.horizon or ds["default_horizon"]
    validation_days = args.validation_days or horizon
    parquet = canonical(ds, out_root / "_cache", progress)
    seq_len = max(FAMILIES[f].required_seq_len(family_params(f, args)) for f in families)
    store_static = {} if args.no_store_features else ds["store_static"]

    print(f"[features] horizon {horizon}, validation {validation_days} days, seq_len {seq_len}")
    t0 = time.perf_counter()
    data = prepare_data(
        parquet,
        ds["mapping"],
        horizon,
        validation_days,
        None,
        history_days=args.history_days,
        store_static=store_static,
        seq_len=seq_len,
    )
    print(
        f"[features] {len(data.X_tr):,} train / {len(data.X_val):,} val rows in"
        f" {time.perf_counter() - t0:.0f}s"
    )

    board = []
    for family in families:
        fam_dir = run_dir / family.value
        fam_dir.mkdir(exist_ok=True)
        device = devices[family]
        params = family_params(family, args)
        print(f"\n[{family.value}] device {device}")
        if args.tune:
            if family not in (ModelFamily.LIGHTGBM, ModelFamily.TORCH_MLP, ModelFamily.TORCH_GRU):
                print(
                    f"[{family.value}] --tune is supported for lightgbm / torch_mlp / torch_gru; training"
                    f" as is"
                )
            else:
                params = tune_family(family, data, params, args.tune, validation_days, device=device)
        t0 = time.perf_counter()
        model, logs, preds = _fit_once(family, data, params, device, progress)
        fit_s = time.perf_counter() - t0
        metrics = evaluate_model(model, data, preds)
        metrics["fit_seconds"] = round(fit_s, 2)
        ckpt = fam_dir / f"model.{model.artifact_ext}"
        model.save(ckpt)
        _validation_frame(data, model, preds).to_parquet(fam_dir / "validation.parquet", index=False)
        _json(logs, fam_dir / "training_log.json")
        _json(metrics, fam_dir / "metrics.json")
        cfg = {
            "family": family.value,
            "params": model.params,
            "device": device,
            "horizon": horizon,
            "validation_days": validation_days,
            "history_days": args.history_days,
            "seq_len": seq_len,
            "use_store_features": not args.no_store_features,
            "canonical": str(parquet),
            "dataset": ds | {"store_static": store_static},
            "covariate_defaults": parse_kv(args.covariate_default),
            "checkpoint": ckpt.name,
        }
        _json(cfg, fam_dir / "config.json")
        row = {
            "family": family.value,
            "device": device,
            "accuracy": round(1 - metrics["wape"], 5),
            **{k: metrics.get(k) for k in ("wape", "rmspe", "rmsse", "rmse", "mae", "r2", "bias_pct")},
            "interval_coverage": (metrics.get("interval") or {}).get("coverage"),
            "fit_seconds": round(fit_s, 1),
            "checkpoint": str(ckpt.relative_to(run_dir)),
        }
        board.append(row)
        print(
            f"[{family.value}] ACC {100 * row['accuracy']:.2f}%  WAPE {metrics['wape']:.4f}  "
            f"RMSPE {metrics.get('rmspe') or float('nan'):.4f}  RMSSE"
            f" {metrics.get('rmsse') or float('nan'):.4f}  "
            f"fit {fit_s:.0f}s"
        )
        if not args.no_forecast:
            path = write_forecast(
                ForecastModel.load(ckpt), metrics, cfg, horizon, fam_dir, args.forecast_format
            )
            print(f"[{family.value}] forecast -> {path.relative_to(run_dir)}")

    pd.DataFrame(board).sort_values("wape").to_csv(run_dir / "leaderboard.csv", index=False)
    env["finished_at"] = datetime.now(UTC).isoformat()
    _json(
        {
            "args": vars(args),
            "environment": env,
            "dataset": ds["name"],
            "horizon": horizon,
            "validation_days": validation_days,
            "leaderboard": board,
        },
        run_dir / "run.json",
    )
    print(
        f"\n[done] {run_dir}\n"
        + pd.DataFrame(board)[["family", "accuracy", "wape", "rmspe", "fit_seconds"]].to_string()
    )
    return 0


def _load(model_dir: Path) -> tuple[ForecastModel, dict[str, Any]]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return ForecastModel.load(model_dir / cfg["checkpoint"]), cfg


def _ensure_canonical(cfg: dict[str, Any], model_dir: Path, data_dir: str | None = None) -> None:
    """Checkpoints copied to another machine: rebuild the canonical Parquet if the cached one is gone.
    Preset datasets (rossmann, m5) are re-resolved under --data-dir when the recorded raw path (e.g. on the
    GPU host) doesn't exist here; custom datasets need their files at the recorded paths."""
    if Path(cfg["canonical"]).exists():
        return
    ds = cfg["dataset"]
    if not Path(ds["raw"]).exists():
        if ds["name"] not in PRESETS:
            raise SystemExit(
                f"data file {ds['raw']} not found on this machine (custom dataset: copy it there)"
            )
        stores = (ds.get("mapping") or {}).get("store_filter")
        fresh = PRESETS[ds["name"]](stores, Path(data_dir).resolve() if data_dir else DATA)
        ds = ds | {
            "raw": str(fresh["raw"]),
            "mapping": fresh["mapping"].model_dump(mode="json"),
            "aux": {str(k): str(v) for k, v in fresh["aux"].items()},
        }
        cfg["dataset"] = ds
        print(f"[data] recorded data path not found here; using {ds['raw']}")
    cache = model_dir.parent.parent / "_cache"
    cfg["canonical"] = str(canonical(ds, cache, ConsoleProgress()))


def cmd_evaluate(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).resolve()
    model, cfg = _load(model_dir)
    _ensure_canonical(cfg, model_dir, args.data_dir)
    data = prepare_data(
        Path(cfg["canonical"]),
        cfg["dataset"]["mapping"],
        cfg["horizon"],
        cfg["validation_days"],
        None,
        history_days=cfg.get("history_days"),
        store_static=cfg["dataset"].get("store_static") or {},
        seq_len=cfg.get("seq_len", 0),
    )
    preds = np.clip(model.predict(data.X_val), 0, None)
    metrics = evaluate_model(model, data, preds)
    _json(metrics, model_dir / "metrics_eval.json")
    print(
        f"ACC {100 * (1 - metrics['wape']):.2f}%  WAPE {metrics['wape']:.4f}  RMSPE {metrics.get('rmspe')}  "
        f"RMSSE {metrics.get('rmsse')}  -> {model_dir / 'metrics_eval.json'}"
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).resolve()
    model, cfg = _load(model_dir)
    _ensure_canonical(cfg, model_dir, args.data_dir)
    horizon = args.horizon or cfg["horizon"]
    if args.covariate_default:
        cfg["covariate_defaults"] = parse_kv(args.covariate_default)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    path = write_forecast(model, metrics, cfg, horizon, model_dir, args.format)
    print(f"forecast ({horizon} days) -> {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.output_dir).resolve()
    rows = []
    for board in sorted(root.glob("*/leaderboard.csv")):
        df = pd.read_csv(board)
        df.insert(0, "run", board.parent.name)
        rows.append(df)
    if not rows:
        print(f"no runs under {root}")
        return 0
    print(
        pd.concat(rows)[["run", "family", "device", "accuracy", "wape", "rmspe", "fit_seconds"]].to_string(
            index=False
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="ingest, train (optionally tune), evaluate, checkpoint and forecast")
    t.add_argument("--dataset", default="rossmann", choices=[*PRESETS, "custom"])
    t.add_argument("--data-dir", default=str(DATA), help="folder holding 'rossman dataset/' and/or 'm5/'")
    t.add_argument("--train-file", help="custom: sales history CSV/Parquet")
    t.add_argument("--mapping-file", help="custom: SchemaMapping JSON (same as PUT /datasets/{id}/mapping)")
    t.add_argument("--store-file", help="custom: store metadata CSV (Rossmann store.csv format)")
    t.add_argument(
        "--aux", action="append", default=[], metavar="DATASET_ID=FILE", help="custom: join tables"
    )
    t.add_argument("--stores", nargs="*", help="only these store codes")
    t.add_argument("--families", nargs="+", default=["lightgbm"], choices=FAMILY_NAMES)
    t.add_argument("--horizon", type=int, help="days ahead (default: 42 Rossmann, 28 M5/custom)")
    t.add_argument("--validation-days", type=int, help="hold-out window (default = horizon)")
    t.add_argument("--history-days", type=int, help="train on the most recent N days only")
    t.add_argument("--no-store-features", action="store_true")
    t.add_argument("--epochs", type=int, default=20, help="neural max epochs")
    t.add_argument("--patience", type=int, help="neural early-stopping patience (<= 0: fixed epochs)")
    t.add_argument("--lgb-rounds", type=int, default=4000)
    t.add_argument("--tune", type=int, default=0, metavar="TRIALS", help="Optuna trials on an inner split")
    t.add_argument("--quantiles", default="auto", choices=["auto", "true", "false"])
    t.add_argument("--ensemble-members", nargs="+", default=["lightgbm", "torch_gru"])
    t.add_argument("--lgb-param", action="append", default=[], metavar="KEY=VALUE")
    t.add_argument("--torch-param", action="append", default=[], metavar="KEY=VALUE")
    t.add_argument(
        "--covariate-default",
        action="append",
        default=[],
        metavar="COL=VALUE",
        help="future covariate default for forecasts, e.g. Promo=0",
    )
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    t.add_argument("--output-dir", default=str(DEFAULT_OUT))
    t.add_argument("--run-name")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--no-forecast", action="store_true")
    t.add_argument("--forecast-format", default="parquet", choices=["parquet", "csv"])
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="re-score a checkpoint on its validation window")
    e.add_argument("--model-dir", required=True, help="outputs/<run>/<family>")
    e.add_argument("--data-dir", help="local data folder if the run was trained elsewhere (default data/raw)")
    e.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("predict", help="forecast the days after the data with a checkpoint")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", help="local data folder if the run was trained elsewhere (default data/raw)")
    p.add_argument("--horizon", type=int, help="<= the model's trained horizon")
    p.add_argument("--covariate-default", action="append", default=[], metavar="COL=VALUE")
    p.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    p.set_defaults(fn=cmd_predict)

    ls = sub.add_parser("list", help="leaderboards of all runs in an output folder")
    ls.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ls.set_defaults(fn=cmd_list)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
