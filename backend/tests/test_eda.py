"""EDA aggregates and residual diagnostics on the Rossmann subset."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.eda import compute_eda
from tests.conftest import rossmann_mapping


def test_rossmann_eda(canonical: Path) -> None:
    meta = {"1": {"store_type": "c", "assortment": "a"}, "2": {"store_type": "a", "assortment": "a"}}
    eda = compute_eda(canonical, rossmann_mapping(), meta)
    assert eda["store_count"] == 3
    assert eda["columns"] == {
        "promo": "Promo",
        "open": "Open",
        "school": "SchoolHoliday",
        "state": "StateHoliday",
    }
    assert [m["month"] for m in eda["monthly"]] == [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    assert sum(m["index"] for m in eda["monthly"]) / 12 == pytest.approx(1.0, abs=0.15)
    by_dow = {w["dow"]: w for w in eda["weekday"]}
    assert by_dow["Sun"]["open_share"] < 0.1  # Rossmann stores 1-3 close on Sundays
    assert eda["promo"]["lift"] > 0.15 and 0 < eda["promo"]["promo_day_share"] < 0.6
    assert eda["promo"]["footfall_lift"] > 0
    assert {h["holiday"] for h in eda["holidays"]} >= {"School holiday", "State holiday a"}
    assert (
        len(eda["stores"]["top"]) == 3 and eda["stores"]["top"][0]["mean"] >= eda["stores"]["top"][-1]["mean"]
    )
    assert all(s["basket"] and 5 < s["basket"] < 20 for s in eda["stores"]["top"])
    assert {s["segment"] for s in eda["segments"]} == {"type c / assortment a", "type a / assortment a"}
    assert eda["trend"][0]["ds"] == "2013-01" and eda["yearly"][0]["year"] == 2013


def test_residual_diagnostics(canonical: Path) -> None:
    from app.ml.training import _fit_once, evaluate_model, prepare_data
    from app.models.enums import ModelFamily

    data = prepare_data(canonical, rossmann_mapping(), 28, 28, None)
    model, _, preds = _fit_once(ModelFamily.LIGHTGBM, data, {"num_boost_round": 80}, "cpu", None)
    res = evaluate_model(model, data, preds)["residuals"]
    assert [w["week"] for w in res["by_week_ahead"]] == [1, 2, 3, 4]
    assert len(res["by_weekday"]) == 7 and all(w["n"] > 0 for w in res["by_weekday"])
    q = res["pct_error_quantiles"]
    assert q["p05"] <= q["p50"] <= q["p95"]
