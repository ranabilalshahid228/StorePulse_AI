"""Cost vs understaffing frontier of the staffing integer program.

Plans the IP on the P50 footfall scaled by a safety factor lambda (lambda = 1 is P50,
lambda = 1 + q90 is the P90 plan), evaluates every plan against realised footfall, and compares the
frontier with the rule-based plan. H3 holds if some lambda gives no more understaffing than the rules
at lower paid hours. Reuses the setting of staffing_ip.py; identical hourly requirements are solved
once (cached).

    cd backend
    .venv\\Scripts\\python ..\\report\\scripts\\staffing_frontier.py
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staffing_ip as base  # noqa: E402

DATA = base.DATA
LAMBDAS = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.117, 1.14]


@lru_cache(maxsize=None)
def solve(r: tuple[int, ...]) -> tuple[tuple[float, ...], float]:
    staff, hours = base.solve_ip(np.array(r, dtype=float))
    return tuple(staff), hours


def main() -> None:
    days = pd.read_parquet(DATA / "staffing_ip_storedays.parquet")[["ds", "store_code", "actual_customers"]]
    val = pd.read_parquet(base.ROOT / "outputs" / "rossmann_h42_gpu" / "ensemble" / "validation.parquet")
    val["ds"] = pd.to_datetime(val["ds"])
    days["ds"] = pd.to_datetime(days["ds"])
    raw = pd.read_csv(
        base.ROOT / "data" / "raw" / "rossman dataset" / "train.csv",
        low_memory=False,
        usecols=["Store", "Date", "Sales", "Customers", "Open"],
    )
    raw["ds"] = pd.to_datetime(raw["Date"])
    raw["store_code"] = raw["Store"].astype(str)
    basket = base.store_basket(raw, val["ds"].min())
    df = days.merge(val[["ds", "store_code", "yhat"]], on=["ds", "store_code"])
    df["f_p50"] = df["yhat"] / df["store_code"].map(basket)
    need = np.stack([base.requirement(c) for c in df["actual_customers"]])
    rows = []
    for lam in LAMBDAS:
        hours, short = 0.0, 0.0
        for k, f in enumerate(df["f_p50"].to_numpy() * lam):
            staff, h = solve(tuple(int(v) for v in base.requirement(float(f))))
            hours += h
            short += float(np.maximum(need[k] - np.array(staff), 0).sum())
        rows.append(
            {"plan": f"ip_lambda_{lam:.3f}", "lambda": lam, "paid_hours": hours, "shortfall_staff_hours": short}
        )
        print(rows[-1], f"(cache {solve.cache_info().currsize})", flush=True)
    rule = pd.read_csv(DATA / "staffing_ip_summary.csv").set_index("plan").loc["rule_p50"]
    out = pd.DataFrame(rows)
    out["cost_vs_rule_pct"] = (out["paid_hours"] / rule["paid_hours"] - 1) * 100
    out["shortfall_vs_rule_pct"] = (out["shortfall_staff_hours"] / rule["shortfall_staff_hours"] - 1) * 100
    out.to_csv(DATA / "staffing_frontier.csv", index=False)
    # Linear interpolation: IP paid hours at the rule plan's understaffing level.
    s, h = out["shortfall_staff_hours"].to_numpy(), out["paid_hours"].to_numpy()
    order = np.argsort(s)
    at_rule = float(np.interp(rule["shortfall_staff_hours"], s[order], h[order]))
    print(f"rule: hours {rule['paid_hours']:.0f}, shortfall {rule['shortfall_staff_hours']:.0f}")
    print(f"IP at equal shortfall: hours {at_rule:.0f} ({(at_rule / rule['paid_hours'] - 1) * 100:+.2f} %)")
    (DATA / "staffing_equal_shortfall.txt").write_text(
        f"rule_paid_hours={rule['paid_hours']:.0f}\nrule_shortfall={rule['shortfall_staff_hours']:.0f}\n"
        f"ip_paid_hours_at_equal_shortfall={at_rule:.0f}\n"
        f"saving_pct={(1 - at_rule / rule['paid_hours']) * 100:.2f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
