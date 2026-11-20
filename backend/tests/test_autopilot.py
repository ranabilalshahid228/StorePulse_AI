"""Store Autopilot: unit tests on hand-checked numbers + the Rossmann end-to-end workflow."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DomainValidationError
from app.models.enums import ModelFamily, TargetType
from app.schemas.autopilot import AutopilotRequest, AutopilotScenario, AutopilotSettings, ScenarioLever
from app.services import autopilot as ap
from app.services import autopilot_report
from app.services.finance import FinanceInputs
from tests.conftest import REPO_ROOT

FIN = FinanceInputs(
    target_type=TargetType.SALES,
    gross_margin_pct=0.4,
    cogs_pct=0.55,
    fixed_daily_overhead=600.0,
    promo_discount_pct=0.1,
    hourly_wage=15.0,
    customers_per_staff_hour=20.0,
    min_staff_per_shift=2,
    shift_hours=4.0,
    operating_hours_per_day=12.0,
    avg_transaction_value=None,
    avg_units_per_transaction=None,
)
COLS = ap.CovariateColumns(["Open", "Promo", "SchoolHoliday"], ["StateHoliday"], "Promo", "Open")


# --- window ------------------------------------------------------------------------------


def test_resolve_window_horizons_and_limits() -> None:
    end = date(2015, 7, 31)
    assert (
        ap.resolve_window("next_day", end, 42).start
        == date(2015, 8, 1)
        == ap.resolve_window("next_day", end, 42).end
    )
    w = ap.resolve_window("week", end, 42)
    assert (w.start, w.end, w.n_days, w.forecast_days) == (date(2015, 8, 1), date(2015, 8, 7), 7, 7)
    assert ap.resolve_window("month", end, 42).end == date(2015, 8, 30)
    c = ap.resolve_window("custom", end, 42, date(2015, 8, 10), date(2015, 8, 20))
    assert (c.n_days, c.forecast_days) == (11, 20)
    with pytest.raises(DomainValidationError, match="trained for 14 days"):
        ap.resolve_window("month", end, 14)
    with pytest.raises(DomainValidationError, match="must start after"):
        ap.resolve_window("custom", end, 42, date(2015, 7, 30), date(2015, 8, 3))


def test_request_validation() -> None:
    sid, mid = "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"
    with pytest.raises(ValueError, match="custom horizon needs"):
        AutopilotRequest(store_id=sid, model_run_id=mid, horizon="custom")
    with pytest.raises(ValueError, match="only used with"):
        AutopilotRequest(store_id=sid, model_run_id=mid, horizon="week", start_date=date(2015, 8, 1))
    with pytest.raises(ValueError, match="shift names"):
        AutopilotSettings(shifts=[{"name": "A", "traffic_share": 0.5}, {"name": "A", "traffic_share": 0.5}])


# --- scenario mapping -------------------------------------------------------------------------


def test_scenario_events_map_levers_to_covariates() -> None:
    w = ap.resolve_window("week", date(2015, 7, 31), 42)
    sc = AutopilotScenario(
        levers=[
            ScenarioLever(
                name="summer promo",
                kind="promo",
                start_date=date(2015, 8, 3),
                end_date=date(2015, 8, 4),
                promo=True,
                school_holiday=True,
                state_holiday="a",
            ),
            ScenarioLever(
                name="shock", start_date=date(2015, 8, 5), end_date=date(2015, 8, 5), demand_pct=-0.2
            ),
        ]
    )
    events = ap.scenario_events(sc, COLS, w)
    assert len(events) == 1  # demand-only levers adjust after the model, they don't re-forecast
    assert events[0]["overrides"] == {"Promo": 1.0, "SchoolHoliday": 1.0, "StateHoliday": "a"}
    assert sc.changes_model_inputs and not sc.is_empty

    bad = AutopilotScenario(
        levers=[ScenarioLever(name="x", start_date=w.start, end_date=w.end, covariates={"Weather": 1})]
    )
    with pytest.raises(DomainValidationError, match="unknown covariate"):
        ap.scenario_events(bad, COLS, w)
    late = AutopilotScenario(
        levers=[
            ScenarioLever(name="late", start_date=date(2015, 9, 1), end_date=date(2015, 9, 2), promo=True)
        ]
    )
    with pytest.raises(DomainValidationError, match="outside the window"):
        ap.scenario_events(late, COLS, w)


def test_promo2_toggle_needs_interval() -> None:
    sc = AutopilotScenario(promo2=True)
    with pytest.raises(DomainValidationError, match="promo2_interval"):
        ap.promo2_meta({"promo2": False}, sc, date(2015, 8, 1))
    meta = ap.promo2_meta(
        {"promo2": False}, sc.model_copy(update={"promo2_interval": "Feb,May,Aug,Nov"}), date(2015, 8, 1)
    )
    assert (
        meta["promo2"] and meta["promo_interval"] == "Feb,May,Aug,Nov" and meta["promo2_since_year"] == 2015
    )
    assert (
        ap.promo2_meta({"promo2": True}, AutopilotScenario(promo2=False), date(2015, 8, 1))["promo2"] is False
    )


# --- footfall, workforce, P&L ---------------------------------------------------------------


def test_basket_footfall_by_weekday_and_promo() -> None:
    ds = pd.date_range("2015-05-04", periods=56)  # 8 full weeks starting Monday
    promo = (ds.dayofweek < 2).astype(int)  # Mon/Tue promo
    hist = pd.DataFrame(
        {"ds": ds, "y": np.where(promo, 1200.0, 1000.0), "customers": 100.0, "Promo": promo, "Open": 1.0}
    )
    basket = ap.basket_profile(hist, "Promo", 182)
    assert basket["by_dow_promo"][(0, 1)] == pytest.approx(12.0)
    assert basket["by_dow_promo"][(3, 0)] == pytest.approx(10.0)
    days = pd.DataFrame({"ds": pd.to_datetime(["2015-06-29", "2015-07-02"]), "is_promo": [True, False]})
    ff, size, note = ap.footfall(pd.Series([2400.0, 1500.0]), days, TargetType.SALES, FIN, basket)
    assert ff.tolist() == pytest.approx([200.0, 150.0]) and "basket size" in note
    with pytest.raises(DomainValidationError, match="Footfall needs"):
        ap.footfall(pd.Series([1.0]), days.head(1), TargetType.SALES, FIN, None)


def test_plan_shifts_rules() -> None:
    days = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2015-08-01", "2015-08-02", "2015-08-03"]),
            "is_open": [True, False, True],
            "footfall": [800.0, 0.0, 40.0],
        }
    )
    settings = AutopilotSettings(max_staff_per_shift=4)
    shifts, staffing = ap.plan_shifts(days, FIN, settings)
    d1 = shifts[shifts["ds"] == "2015-08-01"].set_index("shift")
    # Midday: 800 x 0.45 / 20 = 18 staff-hours over 4 h -> 5 needed, capped at 4 -> understaffed.
    assert d1.loc["Midday Peak", "required_hours"] == pytest.approx(18.0)
    assert d1.loc["Midday Peak", "headcount"] == 4 and d1.loc["Midday Peak", "status"] == "understaffed"
    assert d1.loc["Morning", "headcount"] == 3  # 800 x 0.30 / 20 = 12 h -> 3 staff
    closed = staffing[staffing["ds"] == "2015-08-02"].iloc[0]
    assert closed["labor_hours"] == 0 and closed["staffing_status"] == "closed"
    assert (shifts[shifts["ds"] == "2015-08-02"]["headcount"] == 0).all()
    quiet = shifts[shifts["ds"] == "2015-08-03"]
    assert (quiet["headcount"] == FIN.min_staff_per_shift).all() and (quiet["status"] == "overstaffed").all()
    # Peak hour: 800 x 0.14 = 112 customers / 20 = 6 staff vs 4 on the busiest shift; cap 4 -> no surge.
    assert (
        staffing.iloc[0]["peak_staff_needed"] == 6 and staffing.iloc[0]["staffing_status"] == "understaffed"
    )
    uncapped = ap.plan_shifts(days, FIN, AutopilotSettings())[1].iloc[0]
    assert uncapped["surge_staff"] == 1  # midday 5 staff, peak needs 6
    assert uncapped["labor_hours"] == pytest.approx((3 + 5 + 3) * 4 + 1 * 2)


def test_pnl_promo_break_even_and_bands() -> None:
    days = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2015-08-01", "2015-08-02", "2015-08-03"]),
            "is_open": [True, False, True],
            "is_promo": [True, False, False],
            "revenue_p10": [800.0, 0.0, 900.0],
            "revenue_p50": [900.0, 0.0, 1000.0],
            "revenue_p90": [1000.0, 0.0, 1100.0],
        }
    )
    p = ap.compute_pnl(days, pd.Series([150.0, 0.0, 150.0]), FIN, cogs_pct=0.5, discount_pct=0.1)
    promo = p.iloc[0]
    assert promo["gross_revenue"] == pytest.approx(1000.0) and promo["promo_discount"] == pytest.approx(100.0)
    assert promo["cogs"] == pytest.approx(500.0)
    assert promo["net_profit_p50"] == pytest.approx(900 - 500 - 150 - 600)
    # Break-even: (150 + 600) / (1 - 0.5 / 0.9) on the promo day, / 0.5 on a normal day; none when closed.
    assert promo["break_even_revenue"] == pytest.approx(750 / (1 - 0.5 / 0.9), abs=0.01)
    assert p.iloc[2]["break_even_revenue"] == pytest.approx(1500.0)
    assert np.isnan(p.iloc[1]["break_even_revenue"]) and p.iloc[1]["net_profit_p50"] == -600
    assert (p["net_profit_p10"] <= p["net_profit_p50"]).all() and (
        p["net_profit_p50"] <= p["net_profit_p90"]
    ).all()
    t = ap.pnl_totals(p)
    assert t["revenue_p50"] == 1900 and t["fixed_overhead"] == 1800
    assert t["break_even_revenue"] == pytest.approx(2100 / (t["gross_profit"] / 1900), abs=0.01)


def test_cohort_matches_type_assortment_then_competition() -> None:
    meta = {
        "new": {"store_type": "a", "assortment": "c", "competition_distance": 500},
        "1": {"store_type": "a", "assortment": "c", "competition_distance": 520},
        "2": {"store_type": "a", "assortment": "c", "competition_distance": 9000},
        "3": {"store_type": "b", "assortment": "c", "competition_distance": 500},
        "4": {"store_type": "a", "assortment": "c", "competition_distance": 300},
    }
    codes, matched = ap.cohort_codes("new", meta, {"1", "2", "3", "4"}, 2)
    assert codes == ["1", "4"] and matched == ["store_type", "assortment", "competition_distance"]


# --- Rossmann end-to-end -------------------------------------------------------------------------

STORE_CSV = REPO_ROOT / "data" / "raw" / "rossman dataset" / "store.csv"


@pytest.fixture(scope="module")
def rossmann_ctx(canonical: Path) -> dict:
    if not STORE_CSV.exists():
        pytest.skip("Rossmann store.csv not available")
    from app.ml.training import _fit_once, prepare_data
    from tests.conftest import rossmann_mapping

    meta = pd.read_csv(STORE_CSV, dtype={"Store": str}).set_index("Store")
    meta = meta.rename(
        columns={
            "StoreType": "store_type",
            "Assortment": "assortment",
            "CompetitionDistance": "competition_distance",
            "CompetitionOpenSinceMonth": "competition_open_since_month",
            "CompetitionOpenSinceYear": "competition_open_since_year",
            "Promo2": "promo2",
            "Promo2SinceWeek": "promo2_since_week",
            "Promo2SinceYear": "promo2_since_year",
            "PromoInterval": "promo_interval",
        }
    )
    meta["promo2"] = meta["promo2"].astype(bool)
    static = {
        k: {f: (None if pd.isna(v) else v) for f, v in row.items()}
        for k, row in meta.loc[["1", "2", "3"]].iterrows()
    }
    mapping = rossmann_mapping()
    data = prepare_data(canonical, mapping, 42, 42, None, store_static=static)
    model, _, _ = _fit_once(ModelFamily.LIGHTGBM, data, {"num_boost_round": 150}, "cpu", None)
    return {"model": model, "mapping": mapping, "static": static, "parquet": canonical}


def _run(ctx: dict, progress, **kw) -> dict:  # noqa: ANN001
    args = {
        "model": ctx["model"],
        "metrics": {
            "wape": 0.1,
            "interval": {
                "method": "empirical_relative_residual",
                "coverage": 0.8,
                "lower": -0.12,
                "upper": 0.1,
            },
        },
        "parquet_path": ctx["parquet"],
        "mapping": ctx["mapping"],
        "store_code": "1",
        "store_meta": ctx["static"],
        "target_type": "sales",
        "finance": FIN,
        "settings": AutopilotSettings(),
        "scenario": AutopilotScenario(),
        "horizon": "week",
        "start": None,
        "end": None,
        "currency": "EUR",
        "progress": progress,
        "model_info": {"model_run_id": "m1", "model_family": "lightgbm", "metrics": {"wape": 0.1}},
        "model_comparison": [
            {
                "model_run_id": "m1",
                "name": "lgb",
                "model_family": "lightgbm",
                "run_type": "train",
                "horizon_days": 42,
                "wape": 0.1,
                "rmspe": 0.12,
                "rmsse": 0.2,
                "r2": 0.9,
                "accuracy": 0.9,
                "interval_coverage": 0.8,
            }
        ],
    }
    return ap.run_autopilot(**(args | kw))


def test_rossmann_week_plan_and_exports(rossmann_ctx: dict, progress) -> None:  # noqa: ANN001
    res = _run(rossmann_ctx, progress)
    days = pd.DataFrame(res["days"])
    assert res["window"] == {
        "kind": "week",
        "start": "2015-08-01",
        "end": "2015-08-07",
        "history_end": "2015-07-31",
        "days": 7,
    }
    assert len(days) == 7 and res["cold_start"] is None
    closed = days[~days["is_open"]]
    assert len(closed) == 1 and pd.Timestamp(closed["ds"].iat[0]).dayofweek == 6  # Rossmann store 1: Sundays
    for col in ("revenue_p50", "footfall", "labor_hours", "labor_cost", "staff_Morning", "staff_Midday Peak"):
        assert (closed[col] == 0).all(), col
    open_ = days[days["is_open"]]
    assert (open_["revenue_p10"] <= open_["revenue_p50"]).all()
    assert (open_["revenue_p50"] <= open_["revenue_p90"]).all()
    # Footfall from store 1's own basket size (~8-10 EUR per customer in Rossmann).
    basket = open_["revenue_p50"] / open_["footfall"]
    assert basket.between(5, 15).all()
    assert days["labor_cost"].sum() == pytest.approx(days["labor_hours"].sum() * FIN.hourly_wage, abs=0.05)
    assert res["totals"]["fixed_overhead"] == pytest.approx(7 * 600)
    assert res["kpis"]["revenue_p50"] == pytest.approx(open_["revenue_p50"].sum(), abs=0.1)
    cats = {r["category"] for r in res["recommendations"]}
    assert "closure" in cats
    ranks = [ap.PRIORITY_RANK[r["priority"]] for r in res["recommendations"]]
    assert ranks == sorted(ranks)
    assert len(res["shifts"]) == 7 * 3

    pdf = autopilot_report.executive_pdf(res, title="Autopilot", org_name="Org", store_name="Store 1")
    assert pdf[:4] == b"%PDF" and len(pdf) > 30_000  # embedded charts
    xlsx = autopilot_report.to_xlsx(res)
    names = zipfile.ZipFile(io.BytesIO(xlsx)).namelist()
    assert any("sheet6" in n for n in names)
    sheets = pd.read_excel(io.BytesIO(xlsx), sheet_name=None)
    assert set(sheets) == {
        "Summary",
        "Daily plan",
        "Shift plan",
        "Recommendations",
        "Model comparison",
        "Assumptions",
    }
    assert len(sheets["Daily plan"]) == 7
    csv = autopilot_report.to_csv(res).decode()
    assert csv.splitlines()[0].startswith("ds,is_open,is_promo") and len(csv.splitlines()) == 8


def test_rossmann_whatif_scenario(rossmann_ctx: dict, progress) -> None:  # noqa: ANN001
    calendar = AutopilotScenario(covariate_defaults={"Promo": 0})  # data ends in a promo week
    base = _run(rossmann_ctx, progress, horizon="month", scenario=calendar)
    bdays = pd.DataFrame(base["days"]).set_index("ds")
    # Pick the open non-promo weekdays of the 3rd week to promote, and a local event shock after it.
    target = [d for d, r in bdays.iterrows() if r["is_open"] and not r["is_promo"]][5:7]
    lo, hi = date.fromisoformat(target[0]), date.fromisoformat(target[-1])
    scenario = AutopilotScenario(
        levers=[
            ScenarioLever(name="extra promo", kind="promo", start_date=lo, end_date=hi, promo=True),
            ScenarioLever(
                name="street works",
                kind="local_event",
                start_date=date(2015, 8, 24),
                end_date=date(2015, 8, 25),
                demand_pct=-0.3,
            ),
            ScenarioLever(
                name="close for refit",
                kind="closure",
                start_date=date(2015, 8, 27),
                end_date=date(2015, 8, 27),
                open=False,
            ),
        ],
        covariate_defaults={"Promo": 0},
        promo2=True,
        promo2_interval="Jan,Apr,Jul,Oct",
        cogs_pct=0.5,
    )
    res = _run(rossmann_ctx, progress, horizon="month", scenario=scenario)
    days = pd.DataFrame(res["days"]).set_index("ds")
    assert len(days) == 30 and res["scenario_impact"] is not None
    promo_days = [d.isoformat() for d in pd.date_range(lo, hi).date]
    assert days.loc[promo_days, "is_promo"].all()
    assert (days.loc[promo_days, "revenue_p50"] > days.loc[promo_days, "baseline_revenue_p50"]).all()
    shock = days.loc[["2015-08-24", "2015-08-25"]]
    open_shock = shock[shock["is_open"]]
    assert (open_shock["revenue_p50"] < open_shock["baseline_revenue_p50"] * 0.8).all()
    refit = days.loc["2015-08-27"]
    assert not refit["is_open"] and refit["revenue_p50"] == 0 and refit["labor_hours"] == 0
    cats = {r["category"] for r in res["recommendations"]}
    assert {"promo", "scenario", "closure"} <= cats
    assert any("Closure set by the scenario" in r["detail"] for r in res["recommendations"])
    assert res["kpis"]["scenario_net_profit_delta"] == res["scenario_impact"]["net_profit_p50"]


def test_rossmann_custom_range_and_horizon_limit(rossmann_ctx: dict, progress) -> None:  # noqa: ANN001
    res = _run(rossmann_ctx, progress, horizon="custom", start=date(2015, 8, 10), end=date(2015, 8, 16))
    assert [d["ds"] for d in res["days"]][0] == "2015-08-10" and len(res["days"]) == 7
    with pytest.raises(DomainValidationError, match="trained for 42 days"):
        _run(rossmann_ctx, progress, horizon="custom", start=date(2015, 9, 1), end=date(2015, 9, 20))
    one = _run(rossmann_ctx, progress, horizon="next_day")
    assert len(one["days"]) == 1 and one["window"]["start"] == "2015-08-01"


def test_rossmann_cold_start_store(rossmann_ctx: dict, progress, tmp_path: Path) -> None:  # noqa: ANN001
    """Store 3 with only 20 days of history borrows its cohort's profile, scaled to its own level."""
    df = pd.read_parquet(rossmann_ctx["parquet"])
    df["ds"] = pd.to_datetime(df["ds"])
    keep = (df["store_code"] != "3") | (df["ds"] > df["ds"].max() - pd.Timedelta(days=20))
    sparse = tmp_path / "sparse.parquet"
    df[keep].to_parquet(sparse, index=False)
    res = _run(rossmann_ctx, progress, parquet_path=sparse, store_code="3")
    cold = res["cold_start"]
    assert cold is not None and set(cold["codes"]) <= {"1", "2"} and cold["open_days"] < 90
    assert cold["scale"] > 0 and res["recommendations"][0]["category"] == "cold_start"
    days = pd.DataFrame(res["days"])
    assert len(days) == 7 and (days.loc[days["is_open"], "revenue_p50"] > 0).all()
    assert any("Cold start" in a for a in res["assumptions"])
    # A brand-new store (no rows at all) still gets a plan from its cohort, unscaled.
    none_left = tmp_path / "none.parquet"
    df[df["store_code"] != "3"].to_parquet(none_left, index=False)
    res0 = _run(rossmann_ctx, progress, parquet_path=none_left, store_code="3")
    assert res0["cold_start"]["scale"] == 1.0 and res0["cold_start"]["open_days"] == 0


def test_rossmann_wage_override_and_workforce_kpis(rossmann_ctx: dict, progress) -> None:  # noqa: ANN001
    base = _run(rossmann_ctx, progress)
    res = _run(rossmann_ctx, progress, scenario=AutopilotScenario(hourly_wage=FIN.hourly_wage * 2))
    assert res["kpis"]["labor_hours"] == base["kpis"]["labor_hours"]
    assert res["kpis"]["labor_cost"] == pytest.approx(base["kpis"]["labor_cost"] * 2, abs=0.05)
    assert res["scenario_impact"]["labor_cost"] == pytest.approx(base["kpis"]["labor_cost"], abs=0.05)
    assert any(f"{FIN.hourly_wage * 2:.2f}" in a for a in res["assumptions"])
    k = res["kpis"]
    assert k["footfall_p10"] <= k["footfall"] <= k["footfall_p90"]
    assert k["demand_p10"] <= k["demand_p50"] <= k["demand_p90"]
    assert k["avg_staff_per_open_day"] >= 3 * FIN.min_staff_per_shift
