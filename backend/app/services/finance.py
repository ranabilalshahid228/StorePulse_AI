"""P&L and workforce planning from a forecast + a store's FinancialStaffingConfig.

Pure functions over real inputs - no defaults are invented. If an input needed to convert the
forecast target into revenue or footfall is missing, a DomainValidationError explains which.

Conventions
-----------
* A `sales` forecast is *realized* (post-discount) revenue, because that is what tills record.
  On promo days list-price revenue is therefore `net / (1 - promo_discount_pct)`, and the
  difference is reported as `promo_discount`.
* COGS is charged on list-price revenue (cost of goods doesn't shrink when you discount).
* Labour cost comes from the workforce plan for the same days.
* `fixed_daily_overhead` applies to every calendar day (rent etc. accrue when closed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from app.core.exceptions import DomainValidationError
from app.models.enums import TargetType


@dataclass(frozen=True)
class FinanceInputs:
    target_type: TargetType
    gross_margin_pct: float
    cogs_pct: float
    fixed_daily_overhead: float
    promo_discount_pct: float
    hourly_wage: float
    customers_per_staff_hour: float
    min_staff_per_shift: int
    shift_hours: float
    operating_hours_per_day: float
    avg_transaction_value: float | None
    avg_units_per_transaction: float | None

    @classmethod
    def from_config(
        cls, cfg: object, target_type: str, avg_transaction_value: float | None = None
    ) -> FinanceInputs:
        def f(v: Decimal | None) -> float | None:
            return None if v is None else float(v)

        atv = f(cfg.avg_transaction_value)  # type: ignore[attr-defined]
        return cls(
            target_type=TargetType(target_type),
            gross_margin_pct=float(cfg.gross_margin_pct),  # type: ignore[attr-defined]
            cogs_pct=float(cfg.cogs_pct),  # type: ignore[attr-defined]
            fixed_daily_overhead=float(cfg.fixed_daily_overhead),  # type: ignore[attr-defined]
            promo_discount_pct=float(cfg.promo_discount_pct),  # type: ignore[attr-defined]
            hourly_wage=float(cfg.hourly_wage),  # type: ignore[attr-defined]
            customers_per_staff_hour=float(cfg.customers_per_staff_hour),  # type: ignore[attr-defined]
            min_staff_per_shift=int(cfg.min_staff_per_shift),  # type: ignore[attr-defined]
            shift_hours=float(cfg.shift_hours),  # type: ignore[attr-defined]
            operating_hours_per_day=float(cfg.operating_hours_per_day),  # type: ignore[attr-defined]
            avg_transaction_value=atv if atv is not None else avg_transaction_value,
            avg_units_per_transaction=f(cfg.avg_units_per_transaction),  # type: ignore[attr-defined]
        )


def _customers(forecast: pd.Series, p: FinanceInputs) -> tuple[pd.Series, str]:
    if p.target_type == TargetType.CUSTOMERS:
        return forecast, "Forecast target is customers; used directly as footfall."
    if p.target_type == TargetType.SALES:
        if not p.avg_transaction_value:
            raise DomainValidationError(
                "avg_transaction_value is required to convert a sales forecast into footfall "
                "(set it on the financial config or map a customers column in the dataset)."
            )
        return (
            forecast / p.avg_transaction_value,
            f"Footfall = sales / avg_transaction_value ({p.avg_transaction_value:.2f}).",
        )
    if not p.avg_units_per_transaction:
        raise DomainValidationError("avg_units_per_transaction is required for a units forecast")
    return forecast / p.avg_units_per_transaction, (
        f"Footfall = units / avg_units_per_transaction ({p.avg_units_per_transaction:.3f})."
    )


def _net_revenue(
    forecast: pd.Series, p: FinanceInputs, price_revenue: pd.Series | None = None
) -> tuple[pd.Series, str]:
    if p.target_type == TargetType.SALES:
        return forecast, "Forecast target is realized (net) sales."
    if p.target_type == TargetType.UNITS and price_revenue is not None and price_revenue.notna().all():
        return price_revenue, "Net revenue = sum over items of forecast units x selling price for that day."
    if not p.avg_transaction_value:
        raise DomainValidationError(f"avg_transaction_value is required to value a {p.target_type} forecast")
    if p.target_type == TargetType.CUSTOMERS:
        return forecast * p.avg_transaction_value, "Net revenue = customers x avg_transaction_value."
    if not p.avg_units_per_transaction:
        raise DomainValidationError("avg_units_per_transaction is required to value a units forecast")
    unit_value = p.avg_transaction_value / p.avg_units_per_transaction
    return forecast * unit_value, f"Net revenue = units x {unit_value:.2f} (ATV / UPT)."


def workforce_plan(days: pd.DataFrame, p: FinanceInputs) -> tuple[pd.DataFrame, list[str]]:
    """`days` columns: ds, yhat, is_open. Returns one row per day."""
    customers, note = _customers(days["yhat"], p)
    shifts = math.ceil(p.operating_hours_per_day / p.shift_hours)
    rows = []
    for (_, d), cust in zip(days.iterrows(), customers, strict=True):
        if not d["is_open"]:
            rows.append(
                {
                    "ds": d["ds"],
                    "is_open": False,
                    "expected_customers": 0.0,
                    "required_staff_hours": 0.0,
                    "shifts": 0,
                    "staff_per_shift": 0,
                    "scheduled_staff_hours": 0.0,
                    "labor_cost": 0.0,
                    "utilization_pct": None,
                }
            )
            continue
        required = float(cust) / p.customers_per_staff_hour
        per_shift = max(
            p.min_staff_per_shift, math.ceil(required / (shifts * p.shift_hours)) if required > 0 else 0
        )
        scheduled = per_shift * shifts * p.shift_hours
        rows.append(
            {
                "ds": d["ds"],
                "is_open": True,
                "expected_customers": round(float(cust), 2),
                "required_staff_hours": round(required, 2),
                "shifts": shifts,
                "staff_per_shift": per_shift,
                "scheduled_staff_hours": round(scheduled, 2),
                "labor_cost": round(scheduled * p.hourly_wage, 2),
                "utilization_pct": round(required / scheduled, 4) if scheduled else None,
            }
        )
    assumptions = [
        note,
        f"{shifts} shift(s)/day = ceil({p.operating_hours_per_day:g} operating h / {p.shift_hours:g} h shifts).",
        f"Required staff-hours = footfall / {p.customers_per_staff_hour:g} customers per staff-hour.",
        f"Staff per shift = max({p.min_staff_per_shift}, ceil(required hours / (shifts x shift hours))).",
        f"Labour cost = scheduled hours x {p.hourly_wage:.2f} hourly wage.",
        "Closed days are not staffed.",
    ]
    return pd.DataFrame(rows), assumptions


def pnl(
    days: pd.DataFrame, labor: pd.DataFrame, p: FinanceInputs
) -> tuple[pd.DataFrame, dict[str, float | None], list[str]]:
    """`days` columns: ds, yhat, is_promo, is_open. `labor` is the workforce_plan output."""
    price_rev = days["price_revenue"].astype(float) if "price_revenue" in days else None
    net, note = _net_revenue(days["yhat"].astype(float), p, price_rev)
    discount = days["is_promo"].astype(bool) & days["is_open"].astype(bool)
    gross = net.where(~discount, net / (1 - p.promo_discount_pct))
    df = pd.DataFrame(
        {
            "ds": days["ds"],
            "forecast_target": days["yhat"].astype(float).round(2),
            "is_promo": days["is_promo"].astype(bool),
            "is_open": days["is_open"].astype(bool),
            "gross_revenue": gross,
            "promo_discount": gross - net,
            "net_revenue": net,
        }
    )
    df["cogs"] = df["gross_revenue"] * p.cogs_pct
    df["gross_profit"] = df["net_revenue"] - df["cogs"]
    df["labor_cost"] = labor["labor_cost"].to_numpy()
    df["fixed_overhead"] = p.fixed_daily_overhead
    df["operating_profit"] = df["gross_profit"] - df["labor_cost"] - df["fixed_overhead"]
    money = [
        "gross_revenue",
        "promo_discount",
        "net_revenue",
        "cogs",
        "gross_profit",
        "labor_cost",
        "fixed_overhead",
        "operating_profit",
    ]
    df[money] = df[money].round(2)

    t = {c: round(float(df[c].sum()), 2) for c in money}
    t["realized_gross_margin_pct"] = (
        round(t["gross_profit"] / t["net_revenue"], 4) if t["net_revenue"] else None
    )
    t["planned_gross_margin_pct"] = p.gross_margin_pct
    t["operating_margin_pct"] = (
        round(t["operating_profit"] / t["net_revenue"], 4) if t["net_revenue"] else None
    )
    assumptions = [
        note,
        f"Promo days: list-price revenue = net / (1 - {p.promo_discount_pct:.2%}).",
        f"COGS = {p.cogs_pct:.2%} of list-price revenue.",
        f"Fixed overhead {p.fixed_daily_overhead:.2f} per calendar day.",
        "Labour cost from the workforce plan for the same days.",
    ]
    return df, t, assumptions
