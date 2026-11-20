"""Exploratory analysis of a canonical dataset: seasonality, promo lift, holidays and store comparison.

Everything is computed from the stored Parquet (store-day totals; item-level data is summed per store
and day first). Averages use open days when the dataset maps an open flag, so closures don't dilute
weekday or promo effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _f(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 4)


def _lift(a: float, b: float) -> float | None:
    return _f(a / b - 1) if b and not pd.isna(a) and not pd.isna(b) else None


def _find(cols: list[str], *names: str) -> str | None:
    low = {c.lower(): c for c in cols}
    return next((low[n.lower()] for n in names if n.lower() in low), None)


def store_days(parquet_path: Path, mapping: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    d = pads.dataset(parquet_path, format="parquet")
    names = d.schema.names
    roles = {
        "promo": mapping.get("promo_column") if mapping.get("promo_column") in names else None,
        "open": mapping.get("open_column") if mapping.get("open_column") in names else None,
        "school": _find(names, "SchoolHoliday", "school_holiday"),
        "state": _find(names, "StateHoliday", "state_holiday"),
    }
    cols = ["ds", "store_code", "y", *[c for c in ("customers",) if c in names]]
    cols += [c for c in roles.values() if c and c not in cols]
    df = d.to_table(columns=cols).to_pandas()
    df["ds"] = pd.to_datetime(df["ds"])
    agg: dict[str, tuple[str, str]] = {"y": ("y", "sum")}
    if "customers" in df:
        agg["customers"] = ("customers", "sum")
    for role, c in roles.items():
        if c:
            agg[role] = (c, "max" if role in ("promo", "open", "school") else "first")
    days = df.groupby(["store_code", "ds"], as_index=False).agg(**agg)
    for role in ("promo", "open", "school"):
        if role in days:
            days[role] = pd.to_numeric(days[role], errors="coerce").fillna(0) > 0
    if "state" in days:
        days["state"] = days["state"].astype(str).replace({"nan": "0", "0.0": "0", "None": "0"})
    return days, roles


def compute_eda(
    parquet_path: Path,
    mapping: dict[str, Any],
    store_meta: dict[str, dict[str, Any]] | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    days, roles = store_days(parquet_path, mapping)
    open_days = days[days["open"]] if "open" in days else days[days["y"] > 0]
    o = open_days.assign(dow=open_days["ds"].dt.dayofweek, month=open_days["ds"].dt.month)
    overall = float(o["y"].mean()) if len(o) else 0.0

    # Seasonality: mean store-day target by month / weekday, as an index vs the overall mean.
    monthly = [
        {"month": MONTHS[m - 1], "mean": _f(v), "index": _f(v / overall) if overall else None}
        for m, v in o.groupby("month")["y"].mean().items()
    ]
    weekday = []
    for dow in range(7):  # every weekday, also ones the stores never open (mean None, open share 0)
        g = o[o["dow"] == dow]
        row: dict[str, Any] = {
            "dow": WEEKDAYS[dow],
            "mean": _f(g["y"].mean()) if len(g) else None,
            "open_share": None,
        }
        if "promo" in g:
            p, n = g.loc[g["promo"], "y"].mean(), g.loc[~g["promo"], "y"].mean()
            row |= {"promo": _f(p), "no_promo": _f(n), "promo_lift": _lift(p, n)}
        weekday.append(row)
    if "open" in days:
        share = days.assign(dow=days["ds"].dt.dayofweek).groupby("dow")["open"].mean()
        for row in weekday:
            row["open_share"] = _f(share.get(WEEKDAYS.index(row["dow"])))

    trend = (
        days.set_index("ds")["y"]
        .resample("MS")
        .sum()
        .rename("y")
        .reset_index()
        .assign(ds=lambda t: t["ds"].dt.strftime("%Y-%m"))
    )
    yearly = days.groupby(days["ds"].dt.year)["y"].sum()

    promo = None
    if "promo" in o:
        p, n = o.loc[o["promo"], "y"].mean(), o.loc[~o["promo"], "y"].mean()
        promo = {
            "promo_mean": _f(p),
            "no_promo_mean": _f(n),
            "lift": _lift(p, n),
            "promo_day_share": _f(o["promo"].mean()),
        }
        if "customers" in o:
            pc, nc = o.loc[o["promo"], "customers"].mean(), o.loc[~o["promo"], "customers"].mean()
            promo["footfall_lift"] = _lift(pc, nc)
            promo["basket_lift"] = _lift(p / pc if pc else np.nan, n / nc if nc else np.nan)

    holidays: list[dict[str, Any]] = []
    if "school" in o:
        s, ns = o.loc[o["school"], "y"].mean(), o.loc[~o["school"], "y"].mean()
        holidays.append(
            {"holiday": "School holiday", "days": int(o["school"].sum()), "mean": _f(s), "lift": _lift(s, ns)}
        )
    if "state" in days:
        base = o.loc[o["state"] == "0", "y"].mean()
        for code, g in days[days["state"] != "0"].groupby("state"):
            opened = g[g["open"]] if "open" in g else g[g["y"] > 0]
            holidays.append(
                {
                    "holiday": f"State holiday {code}",
                    "days": int(len(g)),
                    "open_share": _f(len(opened) / len(g)) if len(g) else None,
                    "mean": _f(opened["y"].mean()) if len(opened) else None,
                    "lift": _lift(opened["y"].mean(), base) if len(opened) else None,
                }
            )

    # Store comparison.
    per = o.groupby("store_code").agg(mean=("y", "mean"), total=("y", "sum"), open_days=("y", "size"))
    if "customers" in o:
        cust = o.groupby("store_code")[["y", "customers"]].sum()
        per["basket"] = cust["y"] / cust["customers"].where(cust["customers"] > 0)
        per["footfall"] = o.groupby("store_code")["customers"].mean()
    if "promo" in o:
        pm = o[o["promo"]].groupby("store_code")["y"].mean()
        nm = o[~o["promo"]].groupby("store_code")["y"].mean()
        per["promo_lift"] = pm / nm - 1
    per = per.sort_values("mean", ascending=False)

    def rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        for code, r in frame.iterrows():
            meta = (store_meta or {}).get(str(code), {})
            out.append(
                {
                    "store_code": str(code),
                    "store_type": meta.get("store_type"),
                    "assortment": meta.get("assortment"),
                    **{k: _f(v) for k, v in r.items()},
                }
            )
        return out

    segments: list[dict[str, Any]] = []
    if store_meta:
        seg = per.copy()
        seg["store_type"] = [store_meta.get(str(c), {}).get("store_type") for c in seg.index]
        seg["assortment"] = [store_meta.get(str(c), {}).get("assortment") for c in seg.index]
        seg = seg.dropna(subset=["store_type"])
        for (t, a), g in seg.groupby(["store_type", "assortment"], dropna=False):
            segments.append(
                {
                    "segment": f"type {t} / assortment {a}",
                    "stores": int(len(g)),
                    "mean": _f(g["mean"].mean()),
                    "basket": _f(g["basket"].mean()) if "basket" in g else None,
                    "promo_lift": _f(g["promo_lift"].median()) if "promo_lift" in g else None,
                }
            )
        segments.sort(key=lambda s: -(s["mean"] or 0))

    q = per["mean"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]) if len(per) else pd.Series(dtype=float)
    return {
        "store_count": int(per.shape[0]),
        "open_day_mean": _f(overall),
        "columns": roles,
        "monthly": monthly,
        "weekday": weekday,
        "trend": [{"ds": r.ds, "y": _f(r.y)} for r in trend.itertuples()],
        "yearly": [{"year": int(k), "y": _f(v)} for k, v in yearly.items()],
        "promo": promo,
        "holidays": holidays,
        "stores": {
            "top": rows(per.head(top_n)),
            "bottom": rows(per.tail(min(top_n, max(len(per) - top_n, 0))).iloc[::-1])
            if len(per) > top_n
            else [],
            "distribution": {f"p{int(k * 100)}": _f(v) for k, v in q.items()},
        },
        "segments": segments,
    }
