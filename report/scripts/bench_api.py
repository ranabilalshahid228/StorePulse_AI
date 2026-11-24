"""Measure REST API latency for the endpoints behind the main pages (local stack must be running).

Each endpoint is called once to warm caches, then N times sequentially; wall-clock latency per call is
recorded (client side, loopback, single user). Writes report/data/api_latency_raw.csv and a summary.

    cd backend
    .venv\\Scripts\\python ..\\report\\scripts\\bench_api.py --n 30
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "data"


def login(c: httpx.Client, label: str) -> None:
    text = (ROOT / ".local" / "logins.txt").read_text()
    block = text[text.index(label) :]
    email = re.search(r"email:\s+(\S+)", block).group(1)
    password = re.search(r"password:\s+(\S+)", block).group(1)
    token = c.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]
    c.headers["Authorization"] = "Bearer " + token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    c = httpx.Client(base_url="http://127.0.0.1:8000/api/v1", timeout=120)
    login(c, "Rossmann")
    forecast = next(f for f in c.get("/forecasts").json()["items"] if f["status"] == "succeeded")
    store = c.get("/stores", params={"limit": 1}).json()["items"][0]
    dataset = next(d for d in c.get("/datasets").json()["items"] if d["name"].startswith("Rossmann full"))
    endpoints = {
        "GET /health/ready": "/health/ready",
        "GET /stores (50)": "/stores?limit=50",
        "GET /model-runs": "/model-runs",
        "GET /forecasts/{id}/series": f"/forecasts/{forecast['id']}/series",
        "GET /forecasts/{id}/pnl": f"/forecasts/{forecast['id']}/pnl?store_id={store['id']}",
        "GET /forecasts/{id}/workforce": f"/forecasts/{forecast['id']}/workforce?store_id={store['id']}",
        "GET /datasets/{id}/eda (1.0M rows)": f"/datasets/{dataset['id']}/eda",
    }
    raw, summary = [], []
    for name, path in endpoints.items():
        c.get(path).raise_for_status()  # warm-up
        times = []
        for i in range(args.n):
            t0 = time.perf_counter()
            r = c.get(path)
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            times.append(ms)
            raw.append({"endpoint": name, "call": i + 1, "ms": round(ms, 2), "bytes": len(r.content)})
        q = statistics.quantiles(times, n=20)
        summary.append(
            {
                "endpoint": name,
                "n": args.n,
                "median_ms": round(statistics.median(times), 1),
                "p95_ms": round(q[18], 1),
                "max_ms": round(max(times), 1),
                "response_kb": round(len(r.content) / 1024, 1),
            }
        )
        print(summary[-1], flush=True)
    for name, rows in (("api_latency_raw.csv", raw), ("api_latency.csv", summary)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / "api_latency_env.txt").write_text(
        f"{platform.platform()} | {platform.processor()} | Python {platform.python_version()} | "
        f"loopback, single client, sequential calls, after one warm-up call per endpoint\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
