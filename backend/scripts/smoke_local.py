"""End-to-end smoke test against a running local stack (scripts/local_stack.py up).

Registers a throwaway workspace, imports Rossmann store metadata, uploads a small store subset of
train.csv, maps it, trains LightGBM on the worker, configures one store and runs a 7-day Autopilot
plan, then downloads its PDF / Excel / CSV. Exits non-zero on the first failure.

    .venv\\Scripts\\python -m scripts.smoke_local --stores 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "rossman dataset"
LOCAL = ROOT / ".local"


def wait(client: httpx.Client, path: str, done: set[str], timeout: float = 1800) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        r = client.get(path)
        r.raise_for_status()
        body = r.json()
        state = (body["status"], body.get("stage"), round(body.get("progress_pct") or 0))
        if state != last:
            print(f"    {path}: {state[0]} {state[1] or ''} {state[2]}%", flush=True)
            last = state
        if body["status"] in done:
            return body
        if body["status"] == "failed":
            raise SystemExit(f"{path} failed: {body.get('error')}")
        time.sleep(2)
    raise SystemExit(f"timeout waiting for {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/api/v1")
    ap.add_argument("--stores", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    subset = LOCAL / "tmp" / "rossmann_subset.csv"
    subset.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW / "train.csv", dtype={"StateHoliday": str}, low_memory=False)
    df[df["Store"].isin(args.stores)].to_csv(subset, index=False)

    email = f"smoke-{secrets.token_hex(4)}@example.com"
    password = secrets.token_urlsafe(16)
    with httpx.Client(base_url=args.api, timeout=120) as c:
        print("[1] register")
        r = c.post(
            "/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": "Smoke Test",
                "organization_name": "Smoke Org",
            },
        )
        r.raise_for_status()
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        (LOCAL / "smoke_login.txt").write_text(f"{email}\n{password}\n")

        print("[2] import store.csv")
        with (RAW / "store.csv").open("rb") as fh:
            r = c.post(
                "/stores/import", files={"file": ("store.csv", fh, "text/csv")}, data={"currency": "EUR"}
            )
        r.raise_for_status()

        print("[3] upload + map sales history")
        with subset.open("rb") as fh:
            r = c.post(
                "/datasets",
                files={"file": ("rossmann_subset.csv", fh, "text/csv")},
                data={"name": "Rossmann subset"},
            )
        r.raise_for_status()
        ds = r.json()
        mapping = ds["suggested_mapping"] | {
            "known_future_columns": ["Open", "Promo", "SchoolHoliday"],
            "categorical_columns": ["StateHoliday"],
            "promo_column": "Promo",
            "open_column": "Open",
        }
        c.put(f"/datasets/{ds['id']}/mapping", json=mapping).raise_for_status()
        wait(c, f"/datasets/{ds['id']}", {"ready"})
        c.get(f"/datasets/{ds['id']}/eda").raise_for_status()

        print("[4] train LightGBM on the worker")
        r = c.post(
            "/model-runs/train",
            json={
                "dataset_id": ds["id"],
                "model_family": "lightgbm",
                "name": "smoke lightgbm",
                "horizon_days": 42,
                "hyperparameters": {"num_boost_round": args.rounds},
            },
        )
        r.raise_for_status()
        run = wait(c, f"/model-runs/{r.json()['id']}", {"succeeded"})
        print(
            f"    validation accuracy {1 - run['metrics']['wape']:.2%},"
            f" fit {run['metrics'].get('fit_seconds')} s"
        )

        print("[5] configure store and run a 7-day Autopilot plan")
        stores: dict[str, dict] = {}
        while True:
            page = c.get(f"/stores?limit=500&offset={len(stores)}").json()
            stores |= {s["code"]: s for s in page["items"]}
            if not page["items"] or len(stores) >= page["total"]:
                break
        store = stores[str(args.stores[0])]
        c.put(
            f"/stores/{store['id']}/financial-config",
            json={
                "gross_margin_pct": "0.42",
                "cogs_pct": "0.55",
                "fixed_daily_overhead": "600",
                "promo_discount_pct": "0.10",
                "hourly_wage": "15",
                "customers_per_staff_hour": "20",
                "min_staff_per_shift": 2,
                "shift_hours": "4",
                "operating_hours_per_day": "12",
            },
        ).raise_for_status()
        r = c.post(
            "/autopilot/runs",
            json={
                "store_id": store["id"],
                "model_run_id": run["id"],
                "horizon": "week",
                "scenario": {"covariate_defaults": {"Promo": 0}, "hourly_wage": 16},
            },
        )
        r.raise_for_status()
        ap_run = wait(c, f"/autopilot/runs/{r.json()['id']}", {"succeeded"})
        k = ap_run["result"]["kpis"]
        print(
            f"    revenue P50 {k['revenue_p50']:,.0f}, net profit P50 {k['net_profit_p50']:,.0f}, "
            f"{len(ap_run['result']['recommendations'])} actions"
        )

        print("[6] exports")
        for name, magic in (("report.pdf", b"%PDF"), ("plan.xlsx", b"PK"), ("plan.csv", b"ds,")):
            r = c.get(f"/autopilot/runs/{ap_run['id']}/{name}")
            r.raise_for_status()
            assert r.content.startswith(magic), name
            print(f"    {name}: {len(r.content):,} bytes")
    print(f"OK - log in to the UI with the account in {LOCAL / 'smoke_login.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
