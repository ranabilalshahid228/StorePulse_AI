"""Store Autopilot: forecast a store's window (optionally under a what-if scenario), then footfall,
P&L with P10/P50/P90, a shift-level workforce plan and prioritized, plain-language actions.

Pure computation over stored data, a trained model and the store's configuration (no DB access):
the Celery task loads those and calls `run_autopilot`; tests call it directly on Rossmann.

Data rules (see CLAUDE.md "No mock data"):
* Footfall comes from the forecast itself (customers target) or from the store's own historical
  basket size (sales / customers by weekday x promo); only without a customers history does it fall
  back to the configured average transaction value - and if that is missing too, it fails loudly.
* Operating conventions (three-shift profile, peak-hour share, thresholds) live in
  `AutopilotSettings` on the store's config; every value used is listed in the run's assumptions.
* New / sparse stores (cold start) borrow the daily profile of their most similar stores (same store
  type & assortment, nearest competition distance), scaled to the store's own recent level when it
  has at least two weeks of history.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads

from app.core.exceptions import DomainValidationError
from app.models.enums import TargetType
from app.schemas.autopilot import AutopilotScenario, AutopilotSettings
from app.services.finance import FinanceInputs, _net_revenue

HORIZON_DAYS = {"next_day": 1, "week": 7, "month": 30}
PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
QUANTS = ("p10", "p50", "p90")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ----------------------------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    kind: str
    start: date
    end: date
    history_end: date

    @property
    def forecast_days(self) -> int:
        """Days the model must forecast ahead to cover the window."""
        return (self.end - self.history_end).days

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1


def resolve_window(
    kind: str, history_end: date, model_horizon: int, start: date | None = None, end: date | None = None
) -> Window:
    """Next day / week / month start the day after the last observed date; custom ranges must lie
    inside the model's trained horizon (direct multi-horizon models can't forecast beyond it)."""
    first = history_end + timedelta(days=1)
    if kind in HORIZON_DAYS:
        start, end = first, first + timedelta(days=HORIZON_DAYS[kind] - 1)
    elif kind == "custom":
        if start is None or end is None or end < start:
            raise DomainValidationError("A custom horizon needs start_date <= end_date")
        if start < first:
            raise DomainValidationError(
                f"Custom range must start after the last observed day ({history_end}); "
                f"the first forecastable day is {first}",
                details={"history_end": history_end.isoformat()},
            )
    else:
        raise DomainValidationError(f"Unknown horizon '{kind}'")
    w = Window(kind, start, end, history_end)
    if w.forecast_days > model_horizon:
        raise DomainValidationError(
            f"The window ends {w.forecast_days} days after the data ({history_end}) but the model was "
            f"trained for {model_horizon} days ahead; train a model with horizon_days >= {w.forecast_days}",
            details={"needed_horizon": w.forecast_days, "model_horizon": model_horizon},
        )
    return w


def dataset_end(parquet_path: Path) -> date:
    table = pads.dataset(parquet_path, format="parquet").to_table(columns=["ds"])
    return pd.Timestamp(pc.max(table["ds"]).as_py()).date()


# ----------------------------------------------------------------------------------------
# Scenario -> model inputs
# ----------------------------------------------------------------------------------------


def _find_column(columns: list[str], *names: str) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


@dataclass(frozen=True)
class CovariateColumns:
    known_future: list[str]
    categorical: list[str]
    promo: str | None
    open: str | None

    @property
    def school_holiday(self) -> str | None:
        return _find_column(self.known_future, "SchoolHoliday", "school_holiday")

    @property
    def state_holiday(self) -> str | None:
        return _find_column([*self.categorical, *self.known_future], "StateHoliday", "state_holiday")


def scenario_events(
    scenario: AutopilotScenario, cols: CovariateColumns, window: Window
) -> list[dict[str, Any]]:
    """Levers that change model inputs -> forecasting events (covariate overrides on date ranges)."""
    events = []
    for lv in scenario.levers:
        if lv.end_date < window.start or lv.start_date > window.end:
            raise DomainValidationError(
                f"Lever '{lv.name}' ({lv.start_date}..{lv.end_date}) is outside the window "
                f"{window.start}..{window.end}"
            )
        if not lv.changes_model_inputs:
            continue
        ov: dict[str, float | str] = {}

        def need(col: str | None, label: str, lever: str = lv.name) -> str:
            if col is None:
                raise DomainValidationError(
                    f"Lever '{lever}' sets {label}, but the dataset has no mapped {label} column"
                )
            return col

        if lv.promo is not None:
            ov[need(cols.promo, "promo")] = 1.0 if lv.promo else 0.0
        if lv.open is not None:
            ov[need(cols.open, "open")] = 1.0 if lv.open else 0.0
        if lv.school_holiday is not None:
            ov[need(cols.school_holiday, "SchoolHoliday")] = 1.0 if lv.school_holiday else 0.0
        if lv.state_holiday is not None:
            col = need(cols.state_holiday, "StateHoliday")
            ov[col] = lv.state_holiday if col in cols.categorical else float(lv.state_holiday != "0")
        for c, v in lv.covariates.items():
            if c not in cols.known_future and c not in cols.categorical:
                raise DomainValidationError(
                    f"Lever '{lv.name}' overrides unknown covariate '{c}'",
                    details={"allowed": sorted([*cols.known_future, *cols.categorical])},
                )
            ov[c] = v if c in cols.categorical else float(v)
        events.append(
            {
                "name": lv.name,
                "start_date": lv.start_date.isoformat(),
                "end_date": lv.end_date.isoformat(),
                "store_codes": None,
                "overrides": ov,
            }
        )
    return events


def promo2_meta(meta: dict[str, Any], scenario: AutopilotScenario, start: date) -> dict[str, Any] | None:
    """Store metadata with Promo2 toggled per the scenario (None = no change)."""
    if scenario.promo2 is None:
        return None
    new = dict(meta)
    if scenario.promo2:
        interval = scenario.promo2_interval or meta.get("promo_interval")
        if not interval:
            raise DomainValidationError(
                "promo2_interval is required to enable Promo2 for a store without one (e.g."
                " 'Jan,Apr,Jul,Oct')"
            )
        new["promo_interval"] = interval
        if not (meta.get("promo2") and meta.get("promo2_since_year")):
            iso = start.isocalendar()
            new |= {"promo2_since_week": iso.week, "promo2_since_year": iso.year}
        new["promo2"] = True
    else:
        new["promo2"] = False
    return new


@contextmanager
def store_static_override(model: Any, store_code: str, meta: dict[str, Any] | None) -> Iterator[None]:
    """Temporarily swap one store's metadata in the model spec(s) - incl. ensemble members."""
    if meta is None:
        yield
        return
    specs = {id(m.spec): m.spec for m in [model, *getattr(model, "members", [])]}
    if not any("st_promo2_active" in s.feature_columns for s in specs.values()):
        raise DomainValidationError(
            "This model was trained without store features, so a Promo2 change can't be simulated "
            "(train with use_store_features=true after importing store.csv)"
        )
    saved = {k: s.store_static for k, s in specs.items()}
    try:
        for s in specs.values():
            s.store_static = {**s.store_static, store_code: meta}
        yield
    finally:
        for k, s in specs.items():
            s.store_static = saved[k]


# ----------------------------------------------------------------------------------------
# Forecast frames -> store-day frames (incl. cold start)
# ----------------------------------------------------------------------------------------


def store_days(fc: pd.DataFrame, cols: CovariateColumns) -> pd.DataFrame:
    """Item-level forecast rows of one store -> one row per day."""
    df = fc.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    has_price = "price" in df and df["price"].notna().any()
    if has_price:
        df["price_revenue"] = df["yhat"] * df["price"]
    agg: dict[str, tuple[str, str]] = {
        "yhat": ("yhat", "sum"),
        "yhat_lower": ("yhat_lower", "sum"),
        "yhat_upper": ("yhat_upper", "sum"),
        "is_promo": ("is_promo", "max"),
        "is_open": ("is_open", "max"),
    }
    if has_price:
        agg["price_revenue"] = ("price_revenue", "sum")
    for c in (cols.school_holiday, cols.state_holiday):
        if c and c in df:
            agg[c] = (c, "first")
    days = df.groupby("ds", as_index=False).agg(**agg)
    # P50 only when every series of the day has one (quantile models); else the point forecast is used.
    p50 = df.groupby("ds")["yhat_p50"].agg(lambda v: v.sum() if v.notna().all() else np.nan)
    days["yhat_p50"] = days["ds"].map(p50)
    events = df.dropna(subset=["event"]).groupby("ds")["event"].first() if "event" in df else None
    days["event"] = days["ds"].map(events) if events is not None else None
    return days.sort_values("ds", ignore_index=True)


def cohort_codes(
    store_code: str, meta_all: dict[str, dict[str, Any]], candidates: set[str], size: int
) -> tuple[list[str], list[str]]:
    """Most similar stores: same store_type & assortment (relaxed if too few), then nearest
    competition distance. Returns (codes, matched_on)."""
    me = meta_all.get(store_code, {})
    pool = sorted(c for c in candidates if c != store_code)
    if not pool:
        raise DomainValidationError("Cold start needs other stores in the model's forecast scope")
    group, matched = pool, []
    for keys in (("store_type", "assortment"), ("store_type",), ("assortment",)):
        keys_known = [k for k in keys if me.get(k) is not None]
        if not keys_known:
            continue
        g = [c for c in pool if all(meta_all.get(c, {}).get(k) == me[k] for k in keys_known)]
        if len(g) >= 2:
            group, matched = g, keys_known
            break
    d0 = me.get("competition_distance")
    if d0 is not None:

        def dist(c: str) -> float:
            d = meta_all.get(c, {}).get("competition_distance")
            return abs(math.log1p(float(d)) - math.log1p(float(d0))) if d is not None else float("inf")

        group = sorted(group, key=dist)
        matched = [*matched, "competition_distance"]
    return group[:size], matched


def cohort_profile(
    fc_all: pd.DataFrame,
    codes: list[str],
    cols: CovariateColumns,
    store_hist: pd.DataFrame,
    cohort_hist: pd.DataFrame,
) -> tuple[pd.DataFrame, float, str]:
    """Median daily profile of the cohort, scaled to the store's own level when it has >= 14 open
    days of recent history."""
    per_store = [store_days(fc_all[fc_all["store_code"] == c], cols) for c in codes]
    stack = pd.concat(per_store, ignore_index=True)
    num = ["yhat", "yhat_lower", "yhat_upper", "yhat_p50"]
    days = stack.groupby("ds", as_index=False)[num].median()
    flags = stack.groupby("ds")[["is_open", "is_promo"]].mean() >= 0.5
    days = days.join(flags, on="ds")
    for c in (cols.school_holiday, cols.state_holiday):
        if c and c in stack:
            mode = stack.dropna(subset=[c]).groupby("ds")[c].agg(lambda v: v.mode().iat[0])
            days[c] = days["ds"].map(mode)
    days["event"] = days["ds"].map(stack.dropna(subset=["event"]).groupby("ds")["event"].first())

    scale, note = 1.0, f"Cold start: median daily profile of {len(codes)} similar stores (unscaled)."
    own = _open_days(store_hist)
    if len(own) >= 14:
        recent = own.tail(28)
        ref = _open_days(cohort_hist)
        ref = ref[ref["ds"].isin(recent["ds"])].groupby("ds")["y"].median()
        if len(ref) and ref.mean() > 0:
            scale = float(recent["y"].mean() / ref.mean())
            note = (
                f"Cold start: median daily profile of {len(codes)} similar stores, scaled x{scale:.2f} "
                f"to this store's last {len(recent)} open days."
            )
    for c in num:
        days[c] = days[c] * scale
    return days, scale, note


def _open_days(hist: pd.DataFrame) -> pd.DataFrame:
    """Store-day totals of open days with sales."""
    if hist.empty:
        return pd.DataFrame(columns=["ds", "y", "customers"])
    cols = ["y", "customers"] if "customers" in hist else ["y"]
    d = hist.groupby("ds", as_index=False)[cols].sum(min_count=1)
    return d[d["y"] > 0].sort_values("ds", ignore_index=True)


# ----------------------------------------------------------------------------------------
# Footfall
# ----------------------------------------------------------------------------------------


def basket_profile(hist: pd.DataFrame, promo_column: str | None, lookback_days: int) -> dict[str, Any] | None:
    """Historical basket size (target per customer) by weekday x promo, with fallbacks."""
    if hist.empty or "customers" not in hist or hist["customers"].fillna(0).le(0).all():
        return None
    h = hist[hist["ds"] > hist["ds"].max() - pd.Timedelta(days=lookback_days)]
    aggs = {"y": ("y", "sum"), "customers": ("customers", "sum")}
    if promo_column and promo_column in h:
        aggs["promo"] = (promo_column, "max")
    d = h.groupby("ds", as_index=False).agg(**aggs)
    d = d[(d["customers"] > 0) & (d["y"] > 0)]
    if d.empty:
        return None
    d["dow"] = d["ds"].dt.dayofweek
    d["promo"] = (d["promo"].fillna(0) > 0).astype(int) if "promo" in d else 0

    def ratio(g: pd.DataFrame) -> float:
        return float(g["y"].sum() / g["customers"].sum())

    by_dp = {(int(k[0]), int(k[1])): ratio(g) for k, g in d.groupby(["dow", "promo"]) if len(g) >= 3}
    by_d = {int(k): ratio(g) for k, g in d.groupby("dow") if len(g) >= 3}
    return {"by_dow_promo": by_dp, "by_dow": by_d, "overall": ratio(d), "days": len(d)}


def footfall(
    demand: pd.Series,
    days: pd.DataFrame,
    target_type: TargetType,
    finance: FinanceInputs,
    basket: dict | None,
) -> tuple[pd.Series, pd.Series | None, str]:
    """Expected customers per day from the demand forecast. Returns (footfall, basket, note)."""
    if target_type == TargetType.CUSTOMERS:
        return demand, None, "Forecast target is customers; used directly as footfall."
    if basket:
        dows = days["ds"].dt.dayofweek.to_numpy()
        promos = days["is_promo"].astype(int).to_numpy()
        b = [
            basket["by_dow_promo"].get((int(d), int(p)), basket["by_dow"].get(int(d), basket["overall"]))
            for d, p in zip(dows, promos, strict=True)
        ]
        size = pd.Series(b, index=days.index, dtype="float64")
        unit = "sales" if target_type == TargetType.SALES else "units"
        return (
            demand / size,
            size,
            f"Footfall = forecast {unit} / historical basket size ({unit} per customer by weekday x promo, "
            f"last {basket['days']} trading days; overall {basket['overall']:.2f}).",
        )
    if target_type == TargetType.SALES and finance.avg_transaction_value:
        atv = finance.avg_transaction_value
        return (
            demand / atv,
            pd.Series(atv, index=days.index),
            f"Footfall = sales / configured avg_transaction_value ({atv:.2f}).",
        )
    if target_type == TargetType.UNITS and finance.avg_units_per_transaction:
        upt = finance.avg_units_per_transaction
        return (
            demand / upt,
            pd.Series(upt, index=days.index),
            f"Footfall = units / configured avg_units_per_transaction ({upt:.3f}).",
        )
    raise DomainValidationError(
        "Footfall needs a customers history (map a customers column) or avg_transaction_value / "
        "avg_units_per_transaction on the store's financial config"
    )


# ----------------------------------------------------------------------------------------
# Workforce
# ----------------------------------------------------------------------------------------


def shift_layout(finance: FinanceInputs, settings: AutopilotSettings) -> list[dict[str, Any]]:
    total = sum(s.traffic_share for s in settings.shifts)
    even = finance.operating_hours_per_day / len(settings.shifts)
    return [
        {"name": s.name, "share": s.traffic_share / total, "hours": float(s.hours or even)}
        for s in settings.shifts
    ]


def plan_shifts(
    days: pd.DataFrame, finance: FinanceInputs, settings: AutopilotSettings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`days`: ds, is_open, footfall. Returns (shift rows, per-day staffing summary).

    Per shift: required hours = customers x share / customers_per_staff_hour; headcount =
    max(min_staff, ceil(required / shift hours)), capped by max_staff_per_shift (-> understaffed).
    Peak hour: customers x peak_hour_share need ceil(/ customers_per_staff_hour) staff on the floor;
    any gap over the busiest shift's headcount becomes surge cover (surge_block_hours each)."""
    layout = shift_layout(finance, settings)
    cap = settings.max_staff_per_shift
    cpsh = finance.customers_per_staff_hour
    peak = max(layout, key=lambda s: s["share"] / s["hours"])
    shift_rows, day_rows = [], []
    for d in days.itertuples(index=False):
        if not d.is_open:
            for s in layout:
                shift_rows.append(
                    {
                        "ds": d.ds,
                        "shift": s["name"],
                        "hours": 0.0,
                        "expected_customers": 0.0,
                        "required_hours": 0.0,
                        "headcount": 0,
                        "utilization": None,
                        "status": "closed",
                    }
                )
            day_rows.append(
                {
                    "ds": d.ds,
                    **{f"staff_{s['name']}": 0 for s in layout},
                    "required_hours": 0.0,
                    "labor_hours": 0.0,
                    "peak_hour_customers": 0.0,
                    "peak_staff_needed": 0,
                    "surge_staff": 0,
                    "utilization": None,
                    "staffing_status": "closed",
                }
            )
            continue
        cust_day = float(d.footfall)
        statuses, req_total, sched_total, head = [], 0.0, 0.0, {}
        for s in layout:
            cust = cust_day * s["share"]
            req = cust / cpsh
            need = math.ceil(req / s["hours"] - 1e-9) if req > 0 else 0
            hc = max(finance.min_staff_per_shift, need)
            status = "ok"
            if cap is not None and hc > cap:
                hc, status = cap, "understaffed"
            util = req / (hc * s["hours"]) if hc else None
            if status == "ok" and util is not None and util < settings.overstaff_utilization:
                status = "overstaffed"
            head[s["name"]] = hc
            statuses.append(status)
            req_total += req
            sched_total += hc * s["hours"]
            shift_rows.append(
                {
                    "ds": d.ds,
                    "shift": s["name"],
                    "hours": round(s["hours"], 2),
                    "expected_customers": round(cust, 1),
                    "required_hours": round(req, 2),
                    "headcount": hc,
                    "utilization": round(util, 3) if util is not None else None,
                    "status": status,
                }
            )
        peak_cust = cust_day * settings.peak_hour_share
        peak_need = math.ceil(peak_cust / cpsh - 1e-9) if peak_cust > 0 else 0
        surge = max(0, peak_need - head[peak["name"]])
        if cap is not None and head[peak["name"]] + surge > cap:
            surge = max(0, cap - head[peak["name"]])
            statuses.append("understaffed")
        sched_total += surge * settings.surge_block_hours
        worst = (
            "understaffed"
            if "understaffed" in statuses
            else "overstaffed"
            if all(st == "overstaffed" for st in statuses)
            else "ok"
        )
        day_rows.append(
            {
                "ds": d.ds,
                **{f"staff_{n}": h for n, h in head.items()},
                "required_hours": round(req_total, 2),
                "labor_hours": round(sched_total, 2),
                "peak_hour_customers": round(peak_cust, 1),
                "peak_staff_needed": peak_need,
                "surge_staff": surge,
                "utilization": round(req_total / sched_total, 3) if sched_total else None,
                "staffing_status": worst,
            }
        )
    return pd.DataFrame(shift_rows), pd.DataFrame(day_rows)


# ----------------------------------------------------------------------------------------
# P&L
# ----------------------------------------------------------------------------------------


def compute_pnl(
    days: pd.DataFrame, labor_cost: pd.Series, finance: FinanceInputs, cogs_pct: float, discount_pct: float
) -> pd.DataFrame:
    """`days`: ds, is_open, is_promo, revenue_p10/p50/p90 (realized, post-discount).

    Promo days: list-price revenue = net / (1 - discount); COGS on list-price revenue; labour from
    the shift plan (planned on P50, held fixed across P10/P90); overhead accrues every calendar day.
    Break-even revenue = (labour + overhead) / contribution ratio, where the ratio is
    1 - COGS / (1 - discount) on promo days and 1 - COGS otherwise."""
    promo = days["is_promo"].astype(bool) & days["is_open"].astype(bool)
    d = np.where(promo, discount_pct, 0.0)
    out = pd.DataFrame({"ds": days["ds"], "is_open": days["is_open"].astype(bool), "is_promo": promo})
    labor = labor_cost.to_numpy(dtype="float64")
    for q in QUANTS:
        net = days[f"revenue_{q}"].to_numpy(dtype="float64")
        gross = net / (1 - d)
        cogs = gross * cogs_pct
        out[f"revenue_{q}"] = net
        out[f"net_profit_{q}"] = net - cogs - labor - finance.fixed_daily_overhead
        if q == "p50":
            out["gross_revenue"] = gross
            out["promo_discount"] = gross - net
            out["cogs"] = cogs
            out["gross_profit"] = net - cogs
    out["labor_cost"] = labor
    out["fixed_overhead"] = finance.fixed_daily_overhead
    ratio = 1 - cogs_pct / (1 - d)
    be = (labor + finance.fixed_daily_overhead) / np.where(ratio > 0, ratio, np.nan)
    out["break_even_revenue"] = np.where(out["is_open"] & (ratio > 0), be, np.nan)
    money = [c for c in out.columns if c not in ("ds", "is_open", "is_promo")]
    out[money] = out[money].astype("float64").round(2)
    return out


def pnl_totals(p: pd.DataFrame) -> dict[str, float | None]:
    t: dict[str, float | None] = {
        c: round(float(p[c].sum()), 2)
        for c in (
            "gross_revenue",
            "promo_discount",
            "revenue_p10",
            "revenue_p50",
            "revenue_p90",
            "cogs",
            "gross_profit",
            "labor_cost",
            "fixed_overhead",
            "net_profit_p10",
            "net_profit_p50",
            "net_profit_p90",
        )
    }
    net, gp = t["revenue_p50"] or 0.0, t["gross_profit"] or 0.0
    fixed = (t["labor_cost"] or 0.0) + (t["fixed_overhead"] or 0.0)
    t["gross_margin_pct"] = round(gp / net, 4) if net else None
    t["net_margin_pct"] = round((t["net_profit_p50"] or 0.0) / net, 4) if net else None
    t["break_even_revenue"] = round(fixed / (gp / net), 2) if net and gp > 0 else None
    be = t["break_even_revenue"]
    t["margin_of_safety_pct"] = round((net - be) / net, 4) if be is not None and net else None
    return t


# ----------------------------------------------------------------------------------------
# Anomalies & recommendations
# ----------------------------------------------------------------------------------------


def detect_anomalies(
    plan: pd.DataFrame, hist: pd.DataFrame, promo_column: str | None, threshold: float, scenario_dates: set
) -> list[dict[str, Any]]:
    """Forecast vs the store's recent same-weekday norm, plus data warnings from recent history."""
    out: list[dict[str, Any]] = []
    own = _open_days(hist)
    if own.empty:
        return out
    end = own["ds"].max()
    recent = hist[hist["ds"] > end - pd.Timedelta(days=56)]
    aggs = {"y": ("y", "sum")}
    if promo_column and promo_column in recent:
        aggs["promo"] = (promo_column, "max")
    r = recent.groupby("ds", as_index=False).agg(**aggs)
    r = r[r["y"] > 0]
    r["dow"] = r["ds"].dt.dayofweek
    r["promo"] = (r["promo"].fillna(0) > 0).astype(int) if "promo" in r else 0
    by_dp = r.groupby(["dow", "promo"])["y"].mean().to_dict()
    by_d = r.groupby("dow")["y"].mean().to_dict()
    for row in plan[plan["is_open"]].itertuples(index=False):
        if row.ds in scenario_dates:
            continue  # deliberate what-if change, not an anomaly
        dow = row.ds.dayofweek
        ref = by_dp.get((dow, int(row.is_promo)), by_d.get(dow))
        if not ref:
            continue
        dev = row.demand_p50 / ref - 1
        if abs(dev) > threshold:
            out.append(
                {
                    "kind": "forecast_vs_recent",
                    "ds": row.ds.date().isoformat(),
                    "deviation_pct": round(float(dev), 4),
                    "detail": f"Forecast {row.demand_p50:,.0f} is {dev:+.0%} vs the recent "
                    f"{WEEKDAYS[dow]} {'promo ' if row.is_promo else ''}average of {ref:,.0f}.",
                }
            )
    tail = hist[hist["ds"] > end - pd.Timedelta(days=28)]
    if "is_open_hist" in tail:
        zero_open = tail[(tail["is_open_hist"]) & (tail["y"].fillna(0) == 0)]["ds"].nunique()
        if zero_open:
            out.append(
                {
                    "kind": "zero_sales_while_open",
                    "ds": None,
                    "count": int(zero_open),
                    "detail": f"{zero_open} day(s) in the last 4 weeks were open with zero sales - check "
                    f"for data gaps or refits.",
                }
            )
    last28 = own[own["ds"] > end - pd.Timedelta(days=28)]["y"]
    ly = own[(own["ds"] > end - pd.Timedelta(days=28 + 364)) & (own["ds"] <= end - pd.Timedelta(days=364))]
    if len(last28) >= 10 and len(ly) >= 10 and ly["y"].mean() > 0:
        trend = float(last28.mean() / ly["y"].mean() - 1)
        if abs(trend) >= 0.2:
            out.append(
                {
                    "kind": "trend_vs_last_year",
                    "ds": None,
                    "deviation_pct": round(trend, 4),
                    "detail": f"Last 4 weeks' open-day sales are {trend:+.0%} vs the same weeks last year.",
                }
            )
    return out


def _fmt_dates(ds: list[pd.Timestamp], limit: int = 4) -> str:
    labels = [f"{WEEKDAYS[d.dayofweek]} {d:%d %b}" for d in ds[:limit]]
    more = f" +{len(ds) - limit} more" if len(ds) > limit else ""
    return ", ".join(labels) + more


def _money(v: float, cur: str) -> str:
    return f"{cur} {v:,.0f}"


def recommend(
    *,
    plan: pd.DataFrame,
    shifts: pd.DataFrame,
    baseline: pd.DataFrame | None,
    scenario: AutopilotScenario,
    anomalies: list[dict[str, Any]],
    metrics: dict[str, Any],
    finance: FinanceInputs,
    settings: AutopilotSettings,
    currency: str,
    cold: dict[str, Any] | None,
    cols: CovariateColumns,
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def add(
        priority: str, category: str, title: str, detail: str, dates: list, impact: float | None = None
    ) -> None:
        recs.append(
            {
                "priority": priority,
                "category": category,
                "title": title,
                "detail": detail,
                "dates": [d.date().isoformat() for d in dates],
                "impact": round(float(impact), 2) if impact is not None else None,
            }
        )

    if cold:
        add(
            "high",
            "cold_start",
            "New store: forecast borrowed from similar stores",
            f"{cold['note']} Cohort ({', '.join(cold['matched_on']) or 'all stores'}): "
            f"{', '.join(cold['codes'][:8])}. Re-run once the store has {settings.cold_start_min_days} "
            f"open days of history.",
            [],
        )

    open_ = plan[plan["is_open"]]
    loss = open_[open_["net_profit_p50"] < 0]
    if len(loss):
        add(
            "high",
            "profitability",
            f"Net loss forecast on {len(loss)} open day(s)",
            f"Revenue below break-even on {_fmt_dates(list(loss['ds']))}; combined loss "
            f"{_money(-loss['net_profit_p50'].sum(), currency)}. Review staffing floors and opening hours "
            f"or plan demand drivers for these days.",
            list(loss["ds"]),
            impact=float(loss["net_profit_p50"].sum()),
        )
    risk = open_[(open_["net_profit_p50"] >= 0) & (open_["revenue_p10"] < open_["break_even_revenue"])]
    if len(risk):
        add(
            "medium",
            "profitability",
            f"Downside risk on {len(risk)} day(s)",
            f"At the pessimistic P10 revenue these days fall below break-even"
            f" ({_fmt_dates(list(risk['ds']))}). "
            f"Keep a flexible shift you can release if early trading is weak.",
            list(risk["ds"]),
            impact=float(risk["net_profit_p10"].clip(upper=0).sum()),
        )

    under = shifts[shifts["status"] == "understaffed"]
    if len(under):
        dates = sorted(set(under["ds"]))
        worst = under.sort_values("required_hours", ascending=False).iloc[0]
        add(
            "high",
            "staffing",
            f"Capacity limit reached on {len(dates)} day(s)",
            f"Demand needs more than {settings.max_staff_per_shift} staff per shift (e.g. {worst['shift']}"
            f" on "
            f"{_fmt_dates([worst['ds']], 1)}: {worst['required_hours']:.1f} staff-hours). Expect queues:"
            f" extend "
            f"opening hours, add self-checkout capacity or raise the store's staffing limit.",
            dates,
        )
    surge = plan[plan["surge_staff"] > 0]
    if len(surge):
        extra = float((surge["surge_staff"] * settings.surge_block_hours).sum() * finance.hourly_wage)
        add(
            "medium",
            "staffing",
            f"Add peak-hour cover on {len(surge)} day(s)",
            f"Peak-hour footfall needs up to {int(surge['peak_staff_needed'].max())} staff on the floor;"
            f" schedule "
            f"{int(surge['surge_staff'].max())} floater(s) for {settings.surge_block_hours:g} h around the"
            f" peak on "
            f"{_fmt_dates(list(surge['ds']))} (cost {_money(extra, currency)}, already in the plan).",
            list(surge["ds"]),
            impact=-extra,
        )
    over = shifts[shifts["status"] == "overstaffed"]
    if len(over):
        excess = over["headcount"] * over["hours"] - over["required_hours"]
        saving = float(excess.clip(lower=0).sum() * finance.hourly_wage)
        floor = (over["headcount"] <= finance.min_staff_per_shift).all()
        add(
            "low" if floor else "medium",
            "staffing",
            f"Low utilization in {len(over)} shift(s)",
            f"Scheduled hours are under {settings.overstaff_utilization:.0%} utilized on "
            f"{_fmt_dates(sorted(set(over['ds'])))}"
            + (
                f"; the minimum of {finance.min_staff_per_shift} staff per shift drives this - review the"
                f" floor "
                f"or merge quiet shifts"
                if floor
                else "; trim headcount"
            )
            + f" (up to {_money(saving, currency)} idle labour).",
            sorted(set(over["ds"])),
            impact=saving,
        )

    closed = plan[~plan["is_open"]]
    if len(closed):
        forced = [
            d
            for d in closed["ds"]
            if any(lv.open is False and lv.start_date <= d.date() <= lv.end_date for lv in scenario.levers)
        ]
        add(
            "medium" if forced else "low",
            "closure",
            f"Store closed on {len(closed)} day(s)",
            f"Closed {_fmt_dates(list(closed['ds']))}: forecast sales, footfall and shifts are 0; fixed"
            f" overhead "
            f"of {_money(float(closed['fixed_overhead'].sum()), currency)} still accrues."
            + (" Closure set by the scenario." if forced else ""),
            list(closed["ds"]),
            impact=-float(closed["fixed_overhead"].sum()),
        )
    hol_dates = []
    if cols.state_holiday and cols.state_holiday in plan:
        hol_dates += list(plan[plan[cols.state_holiday].astype(str).isin(["a", "b", "c", "1", "1.0"])]["ds"])
    if cols.school_holiday and cols.school_holiday in plan:
        hol_dates += list(plan[pd.to_numeric(plan[cols.school_holiday], errors="coerce") > 0]["ds"])
    if hol_dates:
        hol = sorted(set(hol_dates))
        sel = plan[plan["ds"].isin(hol) & plan["is_open"]]
        add(
            "medium",
            "holiday",
            f"Holidays in the window ({len(hol)} day(s))",
            f"Public/school holidays on {_fmt_dates(hol)}"
            + (
                f"; open holiday days forecast {_money(float(sel['revenue_p50'].sum()), currency)} revenue - "
                f"confirm opening hours and holiday pay rules."
                if len(sel)
                else "; the store is closed on these days."
            ),
            hol,
        )

    if baseline is not None:
        for lv in scenario.levers:
            mask = (plan["ds"].dt.date >= lv.start_date) & (plan["ds"].dt.date <= lv.end_date)
            if not mask.any():
                continue
            d_rev = float(plan.loc[mask, "revenue_p50"].sum() - baseline.loc[mask, "revenue_p50"].sum())
            d_np = float(plan.loc[mask, "net_profit_p50"].sum() - baseline.loc[mask, "net_profit_p50"].sum())
            dates = list(plan.loc[mask, "ds"])
            if lv.promo:
                if d_np < 0:
                    add(
                        "high",
                        "promo",
                        f"Promo '{lv.name}' is unprofitable",
                        f"It adds {_money(d_rev, currency)} revenue but costs {_money(-d_np, currency)} net"
                        f" profit "
                        f"vs no promo (discount + COGS + extra staff). Reduce the discount depth or target "
                        f"higher-margin days.",
                        dates,
                        impact=d_np,
                    )
                else:
                    add(
                        "low",
                        "promo",
                        f"Promo '{lv.name}' pays off",
                        f"+{_money(d_rev, currency)} revenue and +{_money(d_np, currency)} net profit vs no"
                        f" promo.",
                        dates,
                        impact=d_np,
                    )
            elif lv.demand_pct or lv.price_change_pct or lv.kind in ("local_event", "competitor"):
                add(
                    "medium" if d_np < 0 else "low",
                    "scenario",
                    f"Scenario '{lv.name}': {d_np:+,.0f} {currency} net profit",
                    f"Revenue {d_rev:+,.0f} {currency} vs baseline over {_fmt_dates(dates)}; staffing in"
                    f" the plan "
                    f"already reflects the change.",
                    dates,
                    impact=d_np,
                )

    for a in anomalies:
        if a["kind"] == "forecast_vs_recent":
            continue
        add("medium", "anomaly", a["kind"].replace("_", " ").capitalize(), a["detail"], [])
    devs = [a for a in anomalies if a["kind"] == "forecast_vs_recent"]
    if devs:
        add(
            "medium",
            "anomaly",
            f"Unusual forecast on {len(devs)} day(s)",
            "; ".join(f"{a['ds']}: {a['detail']}" for a in devs[:3])
            + " Check for missing covariates (promo/holiday calendar) before acting.",
            [pd.Timestamp(a["ds"]) for a in devs],
        )
    wape = metrics.get("wape")
    if wape is not None and wape > 0.15:
        add(
            "medium",
            "model",
            f"Forecast uncertainty is high (validation accuracy {1 - wape:.0%})",
            "Treat the staffing plan as provisional and keep on-call cover; consider retraining or tuning.",
            [],
        )
    recs.sort(key=lambda r: (PRIORITY_RANK[r["priority"]], -abs(r["impact"] or 0)))
    return recs


# ----------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------


def _revenue(
    demand: pd.Series, days: pd.DataFrame, target_type: TargetType, finance: FinanceInputs, ratio: pd.Series
) -> tuple[pd.Series, str]:
    """Realized revenue for a demand series (`ratio` = demand / model forecast, for priced units)."""
    price_rev = days["price_revenue"] * ratio if "price_revenue" in days else None
    return _net_revenue(demand.astype("float64"), finance, price_rev)


def _plan(
    days: pd.DataFrame,
    *,
    target_type: TargetType,
    finance: FinanceInputs,
    settings: AutopilotSettings,
    scenario: AutopilotScenario | None,
    basket: dict | None,
) -> dict[str, Any]:
    """Demand levers -> footfall -> shifts -> P&L for one store-day frame."""
    days = days.copy()
    demand_f = pd.Series(1.0, index=days.index)
    price_f = pd.Series(1.0, index=days.index)
    elasticity = scenario.price_elasticity if scenario else None
    for lv in scenario.levers if scenario else []:
        mask = (days["ds"].dt.date >= lv.start_date) & (days["ds"].dt.date <= lv.end_date)
        if lv.demand_pct is not None:
            demand_f[mask] *= 1 + lv.demand_pct
        if lv.price_change_pct is not None:
            price_f[mask] *= 1 + lv.price_change_pct
    volume_f = price_f**elasticity if elasticity is not None else pd.Series(1.0, index=days.index)
    p50_src = days["yhat_p50"].where(days["yhat_p50"].notna(), days["yhat"])
    src = {
        "p10": days["yhat_lower"].fillna(days["yhat"]),
        "p50": p50_src,
        "p90": days["yhat_upper"].fillna(days["yhat"]),
    }
    closed = ~days["is_open"].astype(bool)
    for q in QUANTS:
        d = (src[q] * demand_f * volume_f).where(~closed, 0.0)
        days[f"demand_{q}"] = d
        ratio = (d / days["yhat"].where(days["yhat"] > 0)).fillna(0.0)
        rev, rev_note = _revenue(d, days, target_type, finance, ratio)
        if target_type == TargetType.SALES:
            rev = d  # realized sales at current prices
        days[f"revenue_{q}"] = (rev * price_f).where(~closed, 0.0)
    ff, size, ff_note = footfall(days["demand_p50"], days, target_type, finance, basket)
    days["footfall"] = ff.where(~closed, 0.0)
    for q in ("p10", "p90"):
        days[f"footfall_{q}"] = (
            days[f"demand_{q}"] / size if size is not None else days[f"demand_{q}"]
        ).where(~closed, 0.0)
    days["basket_size"] = (size * price_f) if size is not None else np.nan
    shifts, staffing = plan_shifts(days[["ds", "is_open", "footfall"]], finance, settings)
    staffing["labor_cost"] = (staffing["labor_hours"] * finance.hourly_wage).round(2)
    cogs = scenario.cogs_pct if scenario and scenario.cogs_pct is not None else finance.cogs_pct
    disc = (
        scenario.promo_discount_pct
        if scenario and scenario.promo_discount_pct is not None
        else finance.promo_discount_pct
    )
    pnl = compute_pnl(days, staffing["labor_cost"], finance, cogs, disc)
    plan = days.merge(staffing, on="ds").merge(
        pnl.drop(columns=["is_open", "is_promo", "revenue_p10", "revenue_p50", "revenue_p90", "labor_cost"]),
        on="ds",
    )
    notes = [
        rev_note if target_type != TargetType.SALES else "Forecast target is realized (net) sales.",
        ff_note,
        f"Promo days: list-price revenue = net / (1 - {disc:.1%}); COGS {cogs:.1%} of list-price revenue.",
        f"Fixed overhead {finance.fixed_daily_overhead:,.2f} per calendar day (accrues on closed days).",
    ]
    if elasticity is not None:
        notes.append(f"Price levers change volume by (1 + price change)^{elasticity:g} (price elasticity).")
    elif any(lv.price_change_pct for lv in (scenario.levers if scenario else [])):
        notes.append("Price levers change revenue per sale only; volume held constant (no elasticity given).")
    return {"plan": plan, "shifts": shifts, "totals": pnl_totals(pnl), "notes": notes, "cogs_pct": cogs}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    out = out.astype(object).where(out.notna(), None)
    recs = out.to_dict(orient="records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, np.generic):
                r[k] = v.item()
            if isinstance(r[k], float):
                r[k] = round(r[k], 4) if abs(r[k]) < 10 else round(r[k], 2)
    return recs


def run_autopilot(
    *,
    model: Any,
    metrics: dict[str, Any],
    parquet_path: Path,
    mapping: dict[str, Any],
    store_code: str,
    store_meta: dict[str, dict[str, Any]],
    target_type: TargetType | str,
    finance: FinanceInputs,
    settings: AutopilotSettings,
    scenario: AutopilotScenario,
    horizon: str,
    start: date | None,
    end: date | None,
    currency: str,
    progress: Any,
    model_info: dict[str, Any] | None = None,
    model_comparison: list[dict[str, Any]] | None = None,
    scope_store_code: str | None = None,
    enrich: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    from app.ml import forecasting
    from app.ml.data import load_history

    target_type = TargetType(target_type)
    spec = model.spec
    cols = CovariateColumns(
        list(spec.known_future),
        list(spec.categorical),
        mapping.get("promo_column"),
        mapping.get("open_column"),
    )
    progress.update(0.02, "loading_history", force=True)
    try:
        hist = load_history(parquet_path, store_code)
    except DomainValidationError:
        hist = pd.DataFrame(columns=["ds", "store_code", "item_id", "y"])
    window = resolve_window(horizon, dataset_end(parquet_path), spec.horizon, start, end)
    n_open = int(_open_days(hist)["ds"].nunique()) if len(hist) else 0
    cold = n_open < settings.cold_start_min_days
    if scope_store_code and (cold or scope_store_code != store_code):
        raise DomainValidationError(
            f"The model was trained on store {scope_store_code} only; "
            + (
                "a new store needs a multi-store model for the cold-start fallback"
                if cold
                else "it can't plan another store"
            )
        )

    unknown = set(scenario.covariate_defaults) - set(cols.known_future)
    if unknown:
        raise DomainValidationError(
            "covariate_defaults reference unknown covariates",
            details={"unknown": sorted(unknown), "allowed": cols.known_future},
        )
    events = scenario_events(scenario, cols, window)
    p2 = promo2_meta(store_meta.get(store_code, {}), scenario, window.start)
    fc_kwargs = {
        "metrics": metrics,
        "parquet_path": parquet_path,
        "store_code": None if cold else store_code,
        "horizon": window.forecast_days,
        "covariate_defaults": dict(scenario.covariate_defaults),
        "compare_to_baseline": False,
        "promo_column": cols.promo,
        "open_column": cols.open,
        "price_column": mapping.get("price_column"),
        "scope_store_code": scope_store_code,
        "enrich": enrich,
        "include_covariates": True,
    }
    base_fc, _ = forecasting.run_forecast(
        model=model, events=[], progress=progress.sub(0.05, 0.4), **fc_kwargs
    )
    scen_fc = base_fc
    if scenario.changes_model_inputs:
        with store_static_override(model, store_code, p2):
            scen_fc, _ = forecasting.run_forecast(
                model=model, events=events, progress=progress.sub(0.4, 0.75), **fc_kwargs
            )
    progress.update(0.8, "planning", force=True)

    cold_info = None
    basket_hist = hist
    if cold:
        candidates = set(base_fc["store_code"].astype(str))
        codes, matched = cohort_codes(store_code, store_meta, candidates, settings.cohort_size)
        cohort_hist = load_history(parquet_path, None)
        cohort_hist = cohort_hist[cohort_hist["store_code"].astype(str).isin(codes)]
        base_days, scale, note = cohort_profile(base_fc, codes, cols, hist, cohort_hist)
        scen_days = (
            cohort_profile(scen_fc, codes, cols, hist, cohort_hist)[0]
            if scen_fc is not base_fc
            else base_days
        )
        cold_info = {
            "codes": codes,
            "matched_on": matched,
            "scale": round(scale, 4),
            "note": note,
            "open_days": n_open,
        }
        if basket_profile(hist, cols.promo, settings.basket_lookback_days) is None:
            basket_hist = cohort_hist
    else:
        base_days, scen_days = store_days(base_fc, cols), store_days(scen_fc, cols)

    lo, hi = pd.Timestamp(window.start), pd.Timestamp(window.end)
    base_days = base_days[(base_days["ds"] >= lo) & (base_days["ds"] <= hi)].reset_index(drop=True)
    scen_days = scen_days[(scen_days["ds"] >= lo) & (scen_days["ds"] <= hi)].reset_index(drop=True)
    if base_days.empty:
        raise DomainValidationError(
            f"No forecast rows for store {store_code} in {window.start}..{window.end}: its history ends"
            f" before "
            f"the dataset's last day ({window.history_end})"
        )
    basket = basket_profile(basket_hist, cols.promo, settings.basket_lookback_days)

    common = {"target_type": target_type, "settings": settings, "basket": basket}
    fin_s = (
        dataclasses.replace(finance, hourly_wage=scenario.hourly_wage)
        if scenario.hourly_wage is not None
        else finance
    )
    scen = _plan(scen_days, scenario=scenario, finance=fin_s, **common)
    base = _plan(base_days, scenario=None, finance=finance, **common) if not scenario.is_empty else None
    plan = scen["plan"]
    if base is not None:
        plan["baseline_revenue_p50"] = base["plan"]["revenue_p50"].to_numpy()
        plan["baseline_net_profit_p50"] = base["plan"]["net_profit_p50"].to_numpy()

    hist_flags = hist.copy()
    if cols.open and cols.open in hist_flags:
        hist_flags["is_open_hist"] = hist_flags[cols.open].fillna(1) > 0
    scenario_dates = {
        d
        for lv in scenario.levers
        for d in pd.date_range(lv.start_date, lv.end_date)
        if lv.changes_model_inputs or lv.demand_pct or lv.price_change_pct
    }
    anomalies = detect_anomalies(
        plan,
        hist_flags if not cold else pd.DataFrame(),
        cols.promo,
        settings.anomaly_threshold,
        scenario_dates,
    )
    recs = recommend(
        plan=plan,
        shifts=scen["shifts"],
        baseline=base["plan"] if base is not None else None,
        scenario=scenario,
        anomalies=anomalies,
        metrics=metrics,
        finance=fin_s,
        settings=settings,
        currency=currency,
        cold=cold_info,
        cols=cols,
    )
    progress.update(0.95, "summarizing", force=True)

    totals = scen["totals"]
    layout = shift_layout(finance, settings)
    kpis = {
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "days": window.n_days,
        "open_days": int(plan["is_open"].sum()),
        "revenue_p10": totals["revenue_p10"],
        "revenue_p50": totals["revenue_p50"],
        "revenue_p90": totals["revenue_p90"],
        "net_profit_p10": totals["net_profit_p10"],
        "net_profit_p50": totals["net_profit_p50"],
        "net_profit_p90": totals["net_profit_p90"],
        "gross_margin_pct": totals["gross_margin_pct"],
        "net_margin_pct": totals["net_margin_pct"],
        "break_even_revenue": totals["break_even_revenue"],
        "margin_of_safety_pct": totals["margin_of_safety_pct"],
        "footfall": round(float(plan["footfall"].sum()), 1),
        "footfall_p10": round(float(plan["footfall_p10"].sum()), 1),
        "footfall_p90": round(float(plan["footfall_p90"].sum()), 1),
        "demand_p10": round(float(plan["demand_p10"].sum()), 2),
        "demand_p50": round(float(plan["demand_p50"].sum()), 2),
        "demand_p90": round(float(plan["demand_p90"].sum()), 2),
        "labor_hours": round(float(plan["labor_hours"].sum()), 2),
        "avg_staff_per_open_day": round(
            float(
                plan.loc[plan["is_open"], [f"staff_{s['name']}" for s in layout]].sum(axis=1).mean()
                + plan.loc[plan["is_open"], "surge_staff"].mean()
            ),
            1,
        )
        if plan["is_open"].any()
        else 0.0,
        "labor_cost": totals["labor_cost"],
        "peak_staff_per_shift": int(scen["shifts"]["headcount"].max()) if len(scen["shifts"]) else 0,
        "understaffed_shifts": int((scen["shifts"]["status"] == "understaffed").sum()),
        "overstaffed_shifts": int((scen["shifts"]["status"] == "overstaffed").sum()),
        "cold_start": cold_info is not None,
        "recommendations": {p: sum(r["priority"] == p for r in recs) for p in PRIORITY_RANK},
    }
    impact = None
    if base is not None:
        bt = base["totals"]
        impact = {
            "revenue_p50": round((totals["revenue_p50"] or 0) - (bt["revenue_p50"] or 0), 2),
            "net_profit_p50": round((totals["net_profit_p50"] or 0) - (bt["net_profit_p50"] or 0), 2),
            "labor_cost": round((totals["labor_cost"] or 0) - (bt["labor_cost"] or 0), 2),
            "baseline": bt,
        }
        kpis["scenario_net_profit_delta"] = impact["net_profit_p50"]

    interval = metrics.get("interval") or {}
    assumptions = [
        f"Window {window.start}..{window.end} ({window.kind}); data ends {window.history_end}; model horizon "
        f"{spec.horizon} days.",
        *scen["notes"],
        "Revenue bands: P10/P90 from "
        + (
            "the model's quantile outputs."
            if interval.get("method") == "quantile_model"
            else f"validation residuals (empirical, coverage {interval.get('coverage', float('nan')):.0%})."
            if interval
            else "the point forecast (no interval available)."
        )
        + " Labour is planned on P50 and held fixed across bands.",
        "Shifts: "
        + ", ".join(f"{s['name']} {s['hours']:g} h / {s['share']:.0%} of footfall" for s in layout)
        + f"; {finance.customers_per_staff_hour:g} customers per staff-hour; min"
        f" {finance.min_staff_per_shift} "
        f"staff per open shift"
        + (f"; max {settings.max_staff_per_shift}" if settings.max_staff_per_shift else "; no capacity cap")
        + ".",
        f"Peak hour carries {settings.peak_hour_share:.0%} of daily footfall; gaps are covered by "
        f"{settings.surge_block_hours:g} h floaters. Labour cost = scheduled hours x"
        f" {fin_s.hourly_wage:.2f}.",
        "Closed days (Open = 0): sales, footfall and shifts are 0.",
        "Future covariates: "
        + (
            ", ".join(f"{k}={v:g}" for k, v in scenario.covariate_defaults.items()) + " by default; "
            if scenario.covariate_defaults
            else ""
        )
        + "others repeat the last observed same-weekday value (incl. the promo calendar) unless a lever "
        "or an auxiliary calendar sets them.",
    ]
    if cold_info:
        assumptions.insert(1, cold_info["note"])

    progress.update(1.0, "done", force=True)
    return {
        "window": {
            "kind": window.kind,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "history_end": window.history_end.isoformat(),
            "days": window.n_days,
        },
        "store_code": store_code,
        "currency": currency,
        "target_type": target_type.value,
        "kpis": kpis,
        "totals": totals,
        "scenario_impact": impact,
        "days": _records(plan),
        "shifts": _records(scen["shifts"]),
        "shift_names": [s["name"] for s in layout],
        "recommendations": recs,
        "anomalies": anomalies,
        "cold_start": cold_info,
        "model": model_info or {},
        "model_comparison": model_comparison or [],
        "scenario": scenario.model_dump(mode="json"),
        "assumptions": assumptions,
    }


def summary_of(result: dict[str, Any]) -> dict[str, Any]:
    """Headline numbers stored on the run row for list views."""
    k = result["kpis"]
    keys = ("window_start", "window_end", "days", "revenue_p50", "net_profit_p50", "labor_cost", "footfall")
    return {**{x: k[x] for x in keys}, "cold_start": k["cold_start"], "recommendations": k["recommendations"]}
