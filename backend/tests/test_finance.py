from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.exceptions import DomainValidationError
from app.services import finance


def _cfg(**over):  # noqa: ANN003, ANN202
    base = dict(
        gross_margin_pct=0.40, cogs_pct=0.55, fixed_daily_overhead=500, promo_discount_pct=0.20,
        hourly_wage=20, customers_per_staff_hour=10, min_staff_per_shift=2, shift_hours=8,
        operating_hours_per_day=12, avg_transaction_value=None, avg_units_per_transaction=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _days() -> pd.DataFrame:
    return pd.DataFrame({
        "ds": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
        "yhat": [10_000.0, 8_000.0, 0.0],
        "is_promo": [True, False, False],
        "is_open": [True, True, False],
    })


def test_workforce_and_pnl_sales_target() -> None:
    p = finance.FinanceInputs.from_config(_cfg(avg_transaction_value=10), "sales")
    wf, _ = finance.workforce_plan(_days(), p)
    # Day 1: 1000 customers / 10 per hour = 100 h over 2 shifts x 8h -> ceil(100/16)=7 staff/shift
    assert wf.loc[0, "shifts"] == 2
    assert wf.loc[0, "staff_per_shift"] == 7
    assert wf.loc[0, "scheduled_staff_hours"] == 112
    assert wf.loc[0, "labor_cost"] == 2240
    # Closed day is not staffed.
    assert wf.loc[2, "staff_per_shift"] == 0 and wf.loc[2, "labor_cost"] == 0

    days, totals, _ = finance.pnl(_days(), wf, p)
    # Promo day: list price = 10000 / 0.8 = 12500, discount 2500, COGS on list price.
    assert days.loc[0, "gross_revenue"] == pytest.approx(12_500)
    assert days.loc[0, "promo_discount"] == pytest.approx(2_500)
    assert days.loc[0, "cogs"] == pytest.approx(12_500 * 0.55)
    assert days.loc[1, "gross_revenue"] == pytest.approx(8_000)
    # Overhead accrues on all calendar days, including closed ones.
    assert totals["fixed_overhead"] == 1500
    assert totals["net_revenue"] == pytest.approx(18_000)
    expected_op = 18_000 - (12_500 + 8_000) * 0.55 - wf["labor_cost"].sum() - 1500
    assert totals["operating_profit"] == pytest.approx(expected_op, abs=0.01)


def test_min_staff_floor_applies() -> None:
    p = finance.FinanceInputs.from_config(_cfg(avg_transaction_value=1000), "sales")
    wf, _ = finance.workforce_plan(_days(), p)
    assert wf.loc[1, "staff_per_shift"] == 2  # 8 customers would need 1 staff; floor is 2


def test_sales_target_without_atv_is_rejected() -> None:
    p = finance.FinanceInputs.from_config(_cfg(), "sales")
    with pytest.raises(DomainValidationError):
        finance.workforce_plan(_days(), p)


def test_customers_target_values_revenue_with_atv() -> None:
    p = finance.FinanceInputs.from_config(_cfg(avg_transaction_value=12.5), "customers")
    d = _days().assign(yhat=[100.0, 80.0, 0.0])
    wf, _ = finance.workforce_plan(d, p)
    days, _, _ = finance.pnl(d, wf, p)
    assert days.loc[1, "net_revenue"] == pytest.approx(1000)
