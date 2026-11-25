"""Staffing as an integer program vs the Autopilot's rule-based shift plan.

Setting (Rossmann demo configuration; planning conventions, not data):
  operating day of 12 hours, customers_per_staff_hour c = 20, minimum 2 staff on the floor,
  hourly wage 15 EUR, three 4-hour shifts with footfall shares 30 / 45 / 25 %, 14 % of daily footfall
  in the peak hour (inside the midday shift). The data are daily, so this hourly profile is assumed.

Hourly requirement for a day with footfall F:   r_t = max(m, ceil(F * p_t / c)),  t = 0..11.

Integer program (shift-scheduling set covering, Dantzig 1954): choose x_j in Z>=0 workers for each
shift template j = (start, length), length in {4, 6, 8} hours, start such that the shift fits:
    minimise   sum_j length_j * x_j                       (paid staff-hours; cost = wage * hours)
    subject to sum_{j covers t} x_j >= r_t   for all t     (coverage of every hour)
Solved per store-day with HiGHS (scipy.optimize.milp).

Plans compared (all from the same forecast, evaluated against what actually happened):
  rule_p50 : Autopilot rule plan (app.services.autopilot.plan_shifts) on the P50 forecast
  ip_p50   : integer program on the P50 forecast
  ip_p90   : integer program on the P90 forecast (service-level protection)
Evaluation uses the actual customer counts of the validation window: shortfall = staff-hours below
the hourly requirement implied by actual footfall.

    cd backend
    .venv\\Scripts\\python ..\\report\\scripts\\staffing_ip.py --days 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.enums import TargetType  # noqa: E402
from app.schemas.autopilot import AutopilotSettings  # noqa: E402
from app.services.autopilot import plan_shifts, shift_layout  # noqa: E402
from app.services.finance import FinanceInputs  # noqa: E402

DATA = ROOT / "report" / "data"
HOURS = 12
CPSH = 20.0
MIN_STAFF = 2
WAGE = 15.0
LENGTHS = (4, 6, 8)

FIN = FinanceInputs(
    target_type=TargetType.SALES,
    gross_margin_pct=0.42,
    cogs_pct=0.55,
    fixed_daily_overhead=600.0,
    promo_discount_pct=0.10,
    hourly_wage=WAGE,
    customers_per_staff_hour=CPSH,
    min_staff_per_shift=MIN_STAFF,
    shift_hours=4.0,
    operating_hours_per_day=float(HOURS),
    avg_transaction_value=None,
    avg_units_per_transaction=None,
)
SETTINGS = AutopilotSettings()


def hourly_profile() -> np.ndarray:
    """Hourly footfall shares consistent with the Autopilot settings (30/45/25 %, 14 % peak hour)."""
    layout = shift_layout(FIN, SETTINGS)
    p = []
    for s in layout:
        n = int(round(s["hours"]))
        if s["name"] == "Midday Peak":
            rest = (s["share"] - SETTINGS.peak_hour_share) / (n - 1)
            block = [rest] * n
            block[1] = SETTINGS.peak_hour_share  # peak in the second midday hour
            p += block
        else:
            p += [s["share"] / n] * n
    p = np.array(p)
    assert len(p) == HOURS and abs(p.sum() - 1) < 1e-9
    return p


PROFILE = hourly_profile()
TEMPLATES = [(s, L) for L in LENGTHS for s in range(HOURS - L + 1)]
A = np.zeros((HOURS, len(TEMPLATES)))
for j, (s, L) in enumerate(TEMPLATES):
    A[s : s + L, j] = 1
COST = np.array([L for _, L in TEMPLATES], dtype=float)


def requirement(footfall: float) -> np.ndarray:
    return np.maximum(MIN_STAFF, np.ceil(footfall * PROFILE / CPSH - 1e-9))


def solve_ip(r: np.ndarray) -> tuple[np.ndarray, float]:
    staff, hours = _solve_cached(tuple(int(v) for v in r))
    return np.array(staff), hours


@lru_cache(maxsize=None)
def _solve_cached(r_key: tuple[int, ...]) -> tuple[tuple[float, ...], float]:
    r = np.array(r_key, dtype=float)
    res = milp(
        COST,
        constraints=LinearConstraint(A, lb=r, ub=np.inf),
        integrality=np.ones(len(COST)),
        bounds=Bounds(0, np.inf),
    )
    if not res.success:
        raise RuntimeError(res.message)
    x = np.round(res.x)
    return tuple(A @ x), float(COST @ x)


def rule_plan(footfall: float) -> tuple[np.ndarray, float]:
    """On-floor staff per hour and paid hours of the platform's rule plan for one open day."""
    days = pd.DataFrame({"ds": [pd.Timestamp("2015-07-01")], "is_open": [True], "footfall": [footfall]})
    shifts, summary = plan_shifts(days, FIN, SETTINGS)
    staff = np.concatenate([np.full(int(round(h)), hc) for h, hc in zip(shifts["hours"], shifts["headcount"])])
    surge = int(summary["surge_staff"].iat[0])
    peak = int(np.argmax(PROFILE))
    staff = staff.astype(float)
    staff[peak : peak + int(SETTINGS.surge_block_hours)] += surge
    return staff, float(summary["labor_hours"].iat[0])


def store_basket(raw: pd.DataFrame, origin: pd.Timestamp, days: int = 182) -> pd.Series:
    """Average basket (sales per customer) per store over the `days` days before the forecast origin
    (the first day of the 42-day validation window): only information available when planning."""
    hist = raw[(raw["ds"] < origin) & (raw["ds"] >= origin - pd.Timedelta(days=days))]
    hist = hist[(hist["Open"] == 1) & (hist["Customers"] > 0)]
    return hist.groupby("store_code")["Sales"].sum() / hist.groupby("store_code")["Customers"].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="last N days of the 42-day validation window")
    args = ap.parse_args()
    raw = pd.read_csv(
        ROOT / "data" / "raw" / "rossman dataset" / "train.csv",
        low_memory=False,
        usecols=["Store", "Date", "Sales", "Customers", "Open"],
    )
    raw["ds"] = pd.to_datetime(raw["Date"])
    raw["store_code"] = raw["Store"].astype(str)
    val = pd.read_parquet(ROOT / "outputs" / "rossmann_h42_gpu" / "ensemble" / "validation.parquet")
    val["ds"] = pd.to_datetime(val["ds"])
    basket = store_basket(raw, val["ds"].min())
    start = val["ds"].max() - pd.Timedelta(days=args.days - 1)
    val = val[val["ds"] >= start]
    # Empirical P90 multiplier of the ensemble (relative-residual band, see report Sec. IV-E).

    m = json.loads((ROOT / "outputs" / "rossmann_h42_gpu" / "ensemble" / "metrics.json").read_text())
    upper = (m.get("metrics", m))["interval"]["upper"]
    df = val.merge(raw[["ds", "store_code", "Customers", "Open"]], on=["ds", "store_code"], how="left")
    df = df[(df["Open"] == 1) & (df["yhat"] > 0)].copy()
    df["basket"] = df["store_code"].map(basket)
    df = df.dropna(subset=["basket"])
    df["f_p50"] = df["yhat"] / df["basket"]
    df["f_p90"] = df["yhat"] * (1 + upper) / df["basket"]
    t0 = time.perf_counter()
    rows = []
    for r in df.itertuples(index=False):
        need = requirement(float(r.Customers))  # requirement implied by what actually happened
        plans = {
            "rule_p50": rule_plan(float(r.f_p50)),
            "ip_p50": solve_ip(requirement(float(r.f_p50))),
            "ip_p90": solve_ip(requirement(float(r.f_p90))),
        }
        out = {
            "ds": r.ds,
            "store_code": r.store_code,
            "actual_customers": r.Customers,
            "required_hours_actual": float(need.sum()),
        }
        for name, (staff, hours) in plans.items():
            out[f"{name}_hours"] = hours
            out[f"{name}_shortfall"] = float(np.maximum(need - staff, 0).sum())
            out[f"{name}_covered_hours_share"] = float((staff >= need).mean())
        rows.append(out)
    res = pd.DataFrame(rows)
    res.to_parquet(DATA / "staffing_ip_storedays.parquet", index=False)
    summary = []
    for name in ("rule_p50", "ip_p50", "ip_p90"):
        hrs = res[f"{name}_hours"]
        summary.append(
            {
                "plan": name,
                "store_days": len(res),
                "paid_hours": hrs.sum(),
                "labour_cost_eur": hrs.sum() * WAGE,
                "hours_per_store_day": hrs.mean(),
                "shortfall_staff_hours": res[f"{name}_shortfall"].sum(),
                "shortfall_per_store_day": res[f"{name}_shortfall"].mean(),
                "store_days_without_shortfall": float((res[f"{name}_shortfall"] == 0).mean()),
                "hours_fully_covered": float(res[f"{name}_covered_hours_share"].mean()),
            }
        )
    s = pd.DataFrame(summary)
    s["cost_vs_rule_pct"] = (s["paid_hours"] / s.loc[s["plan"] == "rule_p50", "paid_hours"].iat[0] - 1) * 100
    s.to_csv(DATA / "staffing_ip_summary.csv", index=False)
    print(
        f"{len(res):,} open store-days from {res['ds'].min().date()} to {res['ds'].max().date()}, "
        f"{time.perf_counter() - t0:.0f}s, P90 multiplier {1 + upper:.3f}"
    )
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
