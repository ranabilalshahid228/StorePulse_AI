"""Collect every measured result used in the report into report/data/*.csv.

Sources (nothing is typed in by hand):
  * outputs/<run>/            GPU runs of scripts/train_cli.py on the OTH im-kigs host (metrics.json per family)
  * report/data/raw_runs/     local CPU reruns of scripts/run_pipeline.py (seasonal naive + LightGBM)
  * the running local stack   job timings and dataset sizes via the REST API (needs `local_stack up`)

    cd backend
    .venv\\Scripts\\python ..\\report\\scripts\\collect_results.py [--no-api]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "data"
RUNS = {"rossmann_h42_gpu": 42, "rossmann_h30_all": 30}
LOCAL = {42: OUT / "raw_runs" / "local_rossmann_h42.json", 30: OUT / "raw_runs" / "local_rossmann_h30.json"}


def write(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"skip {name}: no rows")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name} ({len(rows)} rows)")


def gpu_results() -> None:
    comparison, weeks, weekdays, intervals, ensembles = [], [], [], [], []
    for run, horizon in RUNS.items():
        run_dir = ROOT / "outputs" / run
        env = json.loads((run_dir / "run.json").read_text())["environment"]
        for fam_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            m = json.loads((fam_dir / "metrics.json").read_text())
            m = m.get("metrics", m)
            device = "cuda" if fam_dir.name.startswith("torch") or fam_dir.name == "ensemble" else "cpu"
            comparison.append(
                {
                    "horizon_days": horizon,
                    "model": fam_dir.name,
                    "source": f"GPU host {env['host']} ({env['gpus'][0]}), git {env['git_commit']}",
                    "device": device,
                    "validation_start": m["validation_start"],
                    "validation_end": m["validation_end"],
                    "n_rows": m["n"],
                    "wape": m["wape"],
                    "accuracy": 1 - m["wape"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "rmspe": m["rmspe"],
                    "rmsse": m["rmsse"],
                    "r2": m["r2"],
                    "bias_pct": m["bias_pct"],
                    "fit_seconds": m.get("fit_seconds"),
                }
            )
            iv = m.get("interval") or {}
            intervals.append(
                {
                    "horizon_days": horizon,
                    "model": fam_dir.name,
                    "coverage_in_sample": iv.get("coverage"),
                    "coverage_holdout": iv.get("coverage_holdout"),
                    "mean_width": iv.get("mean_width"),
                }
            )
            for w in m["residuals"]["by_week_ahead"]:
                weeks.append({"horizon_days": horizon, "model": fam_dir.name, **w})
            for d in m["residuals"]["by_weekday"]:
                weekdays.append({"horizon_days": horizon, "model": fam_dir.name, **d})
            if m.get("ensemble"):
                e = m["ensemble"]
                for name, s in e["models"].items():
                    ensembles.append(
                        {
                            "horizon_days": horizon,
                            "variant": name,
                            "weights": json.dumps(e["weights"]),
                            "scored_on": e["compared_on"],
                            "holdout_rows": e["holdout_rows"],
                            "wape": s["wape"],
                            "rmse": s["rmse"],
                            "bias_pct": s["bias_pct"],
                        }
                    )
    for horizon, path in LOCAL.items():
        if not path.exists():
            continue
        local = json.loads(path.read_text())
        for name, s in local["models"].items():
            comparison.append(
                {
                    "horizon_days": horizon,
                    "model": name if name != "lightgbm" else "lightgbm (local CPU rerun)",
                    "source": "local laptop CPU, scripts/run_pipeline.py",
                    "device": "cpu",
                    "validation_start": "",
                    "validation_end": "",
                    "n_rows": "",
                    "wape": s["wape"],
                    "accuracy": s["accuracy"],
                    "rmse": s["rmse"],
                    "mae": s["mae"],
                    "rmspe": s["rmspe"],
                    "rmsse": s["rmsse"],
                    "r2": s["r2"],
                    "bias_pct": s["bias_pct"],
                    "fit_seconds": s["fit_seconds"],
                }
            )
            iv = s.get("interval") or {}
            intervals.append(
                {
                    "horizon_days": horizon,
                    "model": name if name != "lightgbm" else "lightgbm (local CPU rerun)",
                    "coverage_in_sample": iv.get("coverage"),
                    "coverage_holdout": iv.get("coverage_holdout"),
                    "mean_width": iv.get("mean_width"),
                }
            )
    write("model_comparison.csv", comparison)
    write("residuals_by_week_ahead.csv", weeks)
    write("residuals_by_weekday.csv", weekdays)
    write("interval_coverage.csv", intervals)
    write("ensemble_holdout.csv", ensembles)


def api_results() -> None:
    import httpx

    text = (ROOT / ".local" / "logins.txt").read_text()
    blocks = re.split(r"\n(?=\d{4}-\d{2}-\d{2})", text)
    timings, datasets, runs = [], [], []
    for block in blocks:
        m = re.search(r"email:\s+(\S+)", block)
        if not m:
            continue
        label = re.sub(r"^\d{4}-\d{2}-\d{2}(\s+\d{2}:\d{2})?\s+", "", block.splitlines()[0]).strip()
        workspace = label.split()[0]
        password = re.search(r"password:\s+(\S+)", block).group(1)
        c = httpx.Client(base_url="http://127.0.0.1:8000/api/v1", timeout=60)
        r = c.post("/auth/login", json={"email": m.group(1), "password": password})
        if r.status_code != 200:
            continue
        c.headers["Authorization"] = "Bearer " + r.json()["access_token"]
        names = {}
        for d in c.get("/datasets", params={"limit": 100}).json()["items"]:
            names[d["id"]] = d["name"]
            datasets.append(
                {
                    "workspace": workspace,
                    "dataset": d["name"],
                    "role": d.get("role"),
                    "rows": d.get("row_count"),
                    "series": d.get("series_count"),
                    "stores": d.get("store_count"),
                    "date_min": d.get("date_min"),
                    "date_max": d.get("date_max"),
                    "status": d["status"],
                }
            )
        for mr in c.get("/model-runs", params={"limit": 100}).json()["items"]:
            runs.append(
                {
                    "workspace": workspace,
                    "dataset": names.get(mr.get("dataset_id"), ""),
                    "name": mr.get("name") or mr["model_family"],
                    "family": mr["model_family"],
                    "run_type": mr["run_type"],
                    "status": mr["status"],
                    "horizon_days": mr["horizon_days"],
                    "device": mr.get("device"),
                    "store_scope": "single store" if mr.get("store_id") else "all stores",
                    "wape": (mr.get("metrics") or {}).get("wape"),
                    "rmsse": (mr.get("metrics") or {}).get("rmsse"),
                    "fit_seconds": (mr.get("metrics") or {}).get("fit_seconds"),
                    "validation_start": (mr.get("metrics") or {}).get("validation_start"),
                    "validation_end": (mr.get("metrics") or {}).get("validation_end"),
                }
            )
        for kind, path in (("forecast", "/forecasts"), ("autopilot", "/autopilot/runs")):
            for j in c.get(path, params={"limit": 100}).json()["items"]:
                if j["status"] != "succeeded" or not j.get("started_at") or not j.get("finished_at"):
                    continue
                start = datetime.fromisoformat(j["started_at"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(j["finished_at"].replace("Z", "+00:00"))
                timings.append(
                    {
                        "workspace": workspace,
                        "job": kind,
                        "horizon": j.get("horizon_days") or j.get("horizon"),
                        "seconds": round((end - start).total_seconds(), 2),
                        "created_at": j["created_at"][:19],
                    }
                )
    write("datasets.csv", datasets)
    write("model_runs_local.csv", runs)
    write("job_timings.csv", timings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true", help="skip results that need the running local stack")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_results()
    if not args.no_api:
        api_results()


if __name__ == "__main__":
    main()
