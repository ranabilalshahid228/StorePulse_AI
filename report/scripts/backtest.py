"""Rolling-origin backtest on Rossmann with the production feature and model code.

For each origin k = 0..K-1 the data are truncated at T_k = last_date - k*H; the last H days before T_k
form the test window and everything earlier is training data. Model selection never uses the test
window ("honest" protocol):

  * seasonal_naive     - weekday-aligned naive baseline
  * lightgbm_honest    - boosting rounds chosen by early stopping on an *inner* window (the H days
                         before the test window), then refit on all training data with that count
  * torch_mlp_honest   - best epoch chosen on the inner window, then refit for that many epochs
  * torch_gru_honest   - same protocol as the MLP
  * ensemble_honest    - LightGBM + GRU, blend weight chosen by WAPE on the inner-window predictions
                         (grid 0, 0.05, ..., 1), applied to the honest test forecasts
  * lightgbm_teststop  - optional (--teststop): the platform's default, early stopping on the test window

Writes <out-dir>/origin_<k>.parquet (ds, store_code, y, one column per forecast). Resumable: finished
origins are skipped.

    cd backend
    # laptop (CPU), LightGBM + baseline, as in the report:
    .venv\\Scripts\\python ..\\report\\scripts\\backtest.py --origins 8 --teststop
    # GPU host, all model families:
    python ../report/scripts/backtest.py --families seasonal_naive lightgbm torch_mlp torch_gru ensemble \\
        --device cuda --out-dir ../report/data/backtest_gpu
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from app.ml.families import FAMILIES  # noqa: E402
from app.ml.training import _fit_once, inner_split, prepare_data  # noqa: E402
from app.models.enums import ModelFamily  # noqa: E402
from app.services.ingestion import convert_to_parquet  # noqa: E402
from scripts.run_pipeline import PRESETS  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "backtest"
NEURAL = {"torch_mlp": ModelFamily.TORCH_MLP, "torch_gru": ModelFamily.TORCH_GRU}


class Quiet:
    def update(self, *a, **k) -> None:  # noqa: D401 - progress sink
        pass

    def sub(self, *a, **k) -> Quiet:
        return self


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def blend_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Weight of `a` in w*a + (1-w)*b minimising WAPE on the inner window (same grid as the platform)."""
    grid = np.linspace(0, 1, 21)
    return float(min(grid, key=lambda w: wape(y, w * a + (1 - w) * b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--rounds", type=int, default=4000, help="LightGBM maximum boosting rounds")
    ap.add_argument("--epochs", type=int, default=30, help="neural maximum epochs")
    ap.add_argument(
        "--families",
        nargs="+",
        default=["seasonal_naive", "lightgbm"],
        choices=["seasonal_naive", "lightgbm", "torch_mlp", "torch_gru", "ensemble"],
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--torch-batch", type=int, default=2048)
    ap.add_argument("--teststop", action="store_true", help="also fit LightGBM with early stopping on the test window")
    ap.add_argument("--stores", nargs="*", help="store codes (smoke tests); default all")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    fams = set(args.families)
    if "ensemble" in fams:
        fams |= {"lightgbm", "torch_gru"}  # the honest blend needs both members
    H = args.horizon
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.txt").write_text(
        f"{platform.node()} | {platform.platform()} | Python {platform.python_version()} | device {args.device}"
        f" | families {sorted(fams)} | epochs {args.epochs} | rounds {args.rounds}\n",
        encoding="utf-8",
    )
    cfg = PRESETS["rossmann"](args.stores)
    mapping = cfg["mapping"]
    seq_len = FAMILIES[ModelFamily.TORCH_GRU].required_seq_len({}) if "torch_gru" in fams else 0
    with tempfile.TemporaryDirectory() as tmp:
        canon = Path(tmp) / "canonical.parquet"
        convert_to_parquet(cfg["raw"], "csv", mapping, canon, 500_000, lambda f, s: None, joins=[])
        full = pd.read_parquet(canon)
        last = pd.Timestamp(full["ds"].max())
        print(f"data to {last.date()}, {len(full):,} rows, families {sorted(fams)}, device {args.device}", flush=True)
        for k in range(args.origins):
            target = out / f"origin_{k}.parquet"
            if target.exists():
                print(f"origin {k}: done, skipped", flush=True)
                continue
            end = last - pd.Timedelta(days=k * H)
            cut = Path(tmp) / f"cut_{k}.parquet"
            full[pd.to_datetime(full["ds"]) <= end].to_parquet(cut, index=False)
            t0 = time.perf_counter()
            data = prepare_data(
                cut, mapping.model_dump(mode="json"), H, H, None, store_static=cfg["store_static"], seq_len=seq_len
            )
            inner = inner_split(data, H)
            frame = data.val_frame[["ds", "store_code", "y"]].copy()
            frame["origin"] = k
            inner_preds: dict[str, np.ndarray] = {}
            notes = []
            if "seasonal_naive" in fams:
                _, _, frame["seasonal_naive"] = _fit_once(ModelFamily.SEASONAL_NAIVE, data, {}, "cpu", Quiet())
            if "lightgbm" in fams:
                base = {"num_boost_round": args.rounds, "objective": "auto", "quantiles": False}
                m_in, _, inner_preds["lightgbm"] = _fit_once(ModelFamily.LIGHTGBM, inner, dict(base), "cpu", Quiet())
                rounds = int(m_in.params.get("best_iteration") or args.rounds)
                honest = dict(base, num_boost_round=rounds, early_stopping_rounds=0)
                _, _, frame["lightgbm_honest"] = _fit_once(ModelFamily.LIGHTGBM, data, honest, "cpu", Quiet())
                frame["rounds_honest"] = rounds
                notes.append(f"lgb rounds {rounds}")
                if args.teststop:
                    m_t, _, frame["lightgbm_teststop"] = _fit_once(
                        ModelFamily.LIGHTGBM, data, dict(base), "cpu", Quiet()
                    )
                    frame["rounds_teststop"] = int(m_t.params.get("best_iteration") or 0)
            for name, family in NEURAL.items():
                if name not in fams:
                    continue
                nn = {"max_epochs": args.epochs, "quantiles": False, "batch_size": args.torch_batch}
                m_in, _, inner_preds[name] = _fit_once(family, inner, dict(nn), args.device, Quiet())
                epochs = int(m_in.params.get("best_epoch") or args.epochs)
                honest = dict(nn, max_epochs=epochs, patience=0)
                _, _, frame[f"{name}_honest"] = _fit_once(family, data, honest, args.device, Quiet())
                frame[f"epochs_{name}"] = epochs
                notes.append(f"{name} epochs {epochs}")
            if "ensemble" in fams:
                w = blend_weight(inner.y_val, inner_preds["lightgbm"], inner_preds["torch_gru"])
                frame["ensemble_honest"] = w * frame["lightgbm_honest"] + (1 - w) * frame["torch_gru_honest"]
                frame["ensemble_weight_lightgbm"] = w
                notes.append(f"blend w_lgb {w:.2f}")
            frame.to_parquet(target, index=False)
            y = frame["y"].to_numpy()
            scores = " ".join(
                f"{c.replace('_honest', '')} {wape(y, frame[c].to_numpy()):.4f}"
                for c in frame.columns
                if c == "seasonal_naive"
                or (c.endswith("_honest") and not c.startswith("rounds"))
                or c == "lightgbm_teststop"
            )
            print(
                f"origin {k}: test {frame['ds'].min().date()}..{frame['ds'].max().date()} n={len(frame):,} | "
                f"WAPE {scores} | {', '.join(notes)} | {time.perf_counter() - t0:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
