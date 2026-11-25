"""Rolling-origin backtest on Rossmann (all stores) with the production feature and model code.

For each origin k = 0..K-1 the data are truncated at T_k = last_date - k*H; the last H days before T_k
form the test window and everything earlier is training data. Per origin three forecasts are made:

  * seasonal_naive          - weekday-aligned naive baseline
  * lightgbm_honest         - boosting rounds chosen by early stopping on an *inner* window (the H days
                              before the test window), then refit on all training data with that count
  * lightgbm_teststop       - the platform's default: early stopping on the test window itself
                              (reported only to quantify the optimism of that protocol)

Writes report/data/backtest/origin_<k>.parquet (ds, store_code, y, one column per forecast) and
report/data/backtest_summary.csv. Resumable: finished origins are skipped.

    cd backend
    .venv\\Scripts\\python ..\\report\\scripts\\backtest.py --origins 8 --horizon 28
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from app.ml.training import _fit_once, inner_split, prepare_data  # noqa: E402
from app.models.enums import ModelFamily  # noqa: E402
from app.services.ingestion import convert_to_parquet  # noqa: E402
from scripts.run_pipeline import PRESETS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "backtest"


class Quiet:
    def update(self, *a, **k) -> None:  # noqa: D401 - progress sink
        pass

    def sub(self, *a, **k) -> Quiet:
        return self


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--rounds", type=int, default=4000)
    args = ap.parse_args()
    H = args.horizon
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PRESETS["rossmann"](None)
    mapping = cfg["mapping"]
    with tempfile.TemporaryDirectory() as tmp:
        canon = Path(tmp) / "canonical.parquet"
        convert_to_parquet(cfg["raw"], "csv", mapping, canon, 500_000, lambda f, s: None, joins=[])
        full = pd.read_parquet(canon)
        last = pd.Timestamp(full["ds"].max())
        print(f"data to {last.date()}, {len(full):,} rows", flush=True)
        for k in range(args.origins):
            target = OUT / f"origin_{k}.parquet"
            if target.exists():
                print(f"origin {k}: done, skipped", flush=True)
                continue
            end = last - pd.Timedelta(days=k * H)
            cut = Path(tmp) / f"cut_{k}.parquet"
            full[pd.to_datetime(full["ds"]) <= end].to_parquet(cut, index=False)
            t0 = time.perf_counter()
            data = prepare_data(cut, mapping.model_dump(mode="json"), H, H, None, store_static=cfg["store_static"])
            base = {"num_boost_round": args.rounds, "objective": "auto", "quantiles": False}
            # Honest: choose the round count on the inner window, then refit on all training data.
            inner = inner_split(data, H)
            m_in, _, _ = _fit_once(ModelFamily.LIGHTGBM, inner, dict(base), "cpu", Quiet())
            rounds = int(m_in.params.get("best_iteration") or args.rounds)
            honest = dict(base, num_boost_round=rounds, early_stopping_rounds=0)
            m_h, _, p_h = _fit_once(ModelFamily.LIGHTGBM, data, honest, "cpu", Quiet())
            # Platform default: early stopping on the evaluated window.
            m_t, _, p_t = _fit_once(ModelFamily.LIGHTGBM, data, dict(base), "cpu", Quiet())
            _, _, p_n = _fit_once(ModelFamily.SEASONAL_NAIVE, data, {}, "cpu", Quiet())
            frame = data.val_frame[["ds", "store_code", "y"]].copy()
            frame["seasonal_naive"] = p_n
            frame["lightgbm_honest"] = p_h
            frame["lightgbm_teststop"] = p_t
            frame["origin"] = k
            frame["rounds_honest"] = rounds
            frame["rounds_teststop"] = int(m_t.params.get("best_iteration") or 0)
            frame.to_parquet(target, index=False)
            y = frame["y"].to_numpy()
            print(
                f"origin {k}: test {frame['ds'].min().date()}..{frame['ds'].max().date()} "
                f"n={len(frame):,} | WAPE naive {wape(y, p_n):.4f} honest {wape(y, p_h):.4f} "
                f"teststop {wape(y, p_t):.4f} | rounds {rounds} vs {frame['rounds_teststop'].iat[0]} "
                f"| {time.perf_counter() - t0:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
