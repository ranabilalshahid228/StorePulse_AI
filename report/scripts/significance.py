"""Significance of forecast-accuracy differences.

Two complementary tests on absolute-error loss (the loss behind WAPE):

1. Diebold-Mariano (DM) test with the Harvey-Leybourne-Newbold small-sample correction.
   Loss differential per day t (summed over stores, scaled by total actual sales):
       d_t = sum_i (|e_A,it| - |e_B,it|) / sum_{i,t} y_it ,   so  sum_t d_t = WAPE_A - WAPE_B.
   DM = dbar / sqrt(LRV / T), LRV = Newey-West long-run variance with Bartlett weights and lag L.
   HLN: DM* = DM * sqrt((T + 1 - 2h + h(h-1)/T) / T), compared with Student t_{T-1}; h = L + 1.
2. Store-level (cluster) bootstrap: resample stores with replacement (B draws) and recompute
   WAPE_A - WAPE_B; percentile 95 % interval. Keeps each store's days together, so serial and
   within-store correlation are preserved.

Inputs: validation.parquet files of the GPU runs (single split) and report/data/backtest/*.parquet
(rolling origins). Outputs: report/data/significance_*.csv.

    cd backend
    .venv\\Scripts\\python ..\\report\\scripts\\significance.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report" / "data"
RNG = np.random.default_rng(20261125)


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    T = len(d)
    lrv = float(d @ d) / T
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * float(d[k:] @ d[:-k]) / T
    return lrv


def dm_test(daily: pd.DataFrame, a: str, b: str, lag: int | None = None) -> dict:
    """`daily`: one row per day with columns abs_<model> (summed absolute error) and y (summed actuals)."""
    total = daily["y"].sum()
    d = ((daily[f"abs_{a}"] - daily[f"abs_{b}"]) / total).to_numpy()
    T = len(d)
    L = lag if lag is not None else int(math.floor(4 * (T / 100) ** (2 / 9)))
    lrv = newey_west_lrv(d, L)
    dm = d.mean() / math.sqrt(lrv / T) if lrv > 0 else float("nan")
    h = L + 1
    hln = dm * math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(hln), df=T - 1)
    return {"T_days": T, "nw_lag": L, "wape_diff": float(d.sum()), "dm": dm, "dm_hln": hln, "p_value": p}


def store_bootstrap(frame: pd.DataFrame, a: str, b: str, draws: int = 2000) -> tuple[float, float]:
    g = frame.assign(ea=(frame["y"] - frame[a]).abs(), eb=(frame["y"] - frame[b]).abs())
    per = g.groupby("store_code")[["ea", "eb", "y"]].sum()
    ea, eb, y = per["ea"].to_numpy(), per["eb"].to_numpy(), per["y"].to_numpy()
    n = len(per)
    idx = RNG.integers(0, n, size=(draws, n))
    diff = ea[idx].sum(1) / y[idx].sum(1) - eb[idx].sum(1) / y[idx].sum(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return float(lo), float(hi)


def daily_losses(frame: pd.DataFrame, models: list[str], day_key: list[str]) -> pd.DataFrame:
    agg = {"y": ("y", "sum")}
    for m in models:
        frame[f"abs_{m}"] = (frame["y"] - frame[m]).abs()
        agg[f"abs_{m}"] = (f"abs_{m}", "sum")
    return frame.groupby(day_key).agg(**agg).reset_index()


def wape(frame: pd.DataFrame, m: str) -> float:
    return float((frame["y"] - frame[m]).abs().sum() / frame["y"].sum())


def single_split() -> None:
    rows = []
    runs = {42: ROOT / "outputs" / "rossmann_h42_gpu", 30: ROOT / "outputs" / "rossmann_h30_all"}
    pairs = {
        42: [("ensemble", "lightgbm"), ("torch_mlp", "lightgbm"), ("torch_gru", "lightgbm")],
        30: [
            ("torch_gru", "lightgbm"),
            ("torch_mlp", "lightgbm"),
            ("torch_gru", "torch_mlp"),
            ("lightgbm", "sarima"),
            ("sarima", "arima"),
        ],
    }
    for h, run in runs.items():
        frame = None
        for fam in sorted({m for p in pairs[h] for m in p}):
            v = pd.read_parquet(run / fam / "validation.parquet")[["ds", "store_code", "y", "yhat"]]
            v = v.rename(columns={"yhat": fam})
            frame = v if frame is None else frame.merge(v, on=["ds", "store_code", "y"])
        models = [c for c in frame.columns if c not in ("ds", "store_code", "y")]
        daily = daily_losses(frame.copy(), models, ["ds"])
        for a, b in pairs[h]:
            lo, hi = store_bootstrap(frame, a, b)
            rows.append(
                {
                    "horizon_days": h,
                    "model_a": a,
                    "model_b": b,
                    "wape_a": wape(frame, a),
                    "wape_b": wape(frame, b),
                    **dm_test(daily, a, b),
                    "boot_lo": lo,
                    "boot_hi": hi,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "significance_single_split.csv", index=False)
    print(out.round(4).to_string(index=False))


def backtest() -> None:
    files = sorted((DATA / "backtest").glob("origin_*.parquet"))
    if not files:
        print("no backtest results yet")
        return
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    models = ["seasonal_naive", "lightgbm_honest", "lightgbm_teststop"]
    per_origin = []
    for k, g in frame.groupby("origin"):
        per_origin.append(
            {
                "origin": int(k),
                "test_start": str(pd.Timestamp(g["ds"].min()).date()),
                "test_end": str(pd.Timestamp(g["ds"].max()).date()),
                "n_rows": len(g),
                "stores": g["store_code"].nunique(),
                "rounds_honest": int(g["rounds_honest"].iat[0]),
                "rounds_teststop": int(g["rounds_teststop"].iat[0]),
                **{f"wape_{m}": wape(g, m) for m in models},
            }
        )
    po = pd.DataFrame(per_origin).sort_values("test_start")
    po.to_csv(DATA / "backtest_per_origin.csv", index=False)
    K = len(po)
    summary = []
    for m in models:
        x = po[f"wape_{m}"].to_numpy()
        half = stats.t.ppf(0.975, K - 1) * x.std(ddof=1) / math.sqrt(K) if K > 1 else float("nan")
        summary.append(
            {
                "model": m,
                "origins": K,
                "mean_wape": x.mean(),
                "ci_lo": x.mean() - half,
                "ci_hi": x.mean() + half,
                "min": x.min(),
                "max": x.max(),
            }
        )
    pd.DataFrame(summary).to_csv(DATA / "backtest_summary.csv", index=False)
    daily = daily_losses(frame.copy(), models, ["origin", "ds"]).sort_values(["origin", "ds"])
    tests = []
    for a, b in (("lightgbm_honest", "seasonal_naive"), ("lightgbm_teststop", "lightgbm_honest")):
        lo, hi = store_bootstrap(frame, a, b)
        diffs = po[f"wape_{a}"] - po[f"wape_{b}"]
        wins = int((diffs < 0).sum())
        tests.append(
            {
                "model_a": a,
                "model_b": b,
                "pooled_wape_a": wape(frame, a),
                "pooled_wape_b": wape(frame, b),
                **dm_test(daily, a, b),
                "boot_lo": lo,
                "boot_hi": hi,
                "origins_a_better": wins,
                "origins": K,
                "sign_test_p": stats.binomtest(wins, K, 0.5).pvalue if K else float("nan"),
            }
        )
    t = pd.DataFrame(tests)
    t.to_csv(DATA / "significance_backtest.csv", index=False)
    print(po.round(4).to_string(index=False))
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(t.round(5).to_string(index=False))


if __name__ == "__main__":
    single_split()
    backtest()
