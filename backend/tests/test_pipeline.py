"""End-to-end ML pipeline on a real Rossmann subset (no database / broker needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.families import FAMILIES, ForecastModel
from app.ml.forecasting import run_forecast
from app.ml.training import _fit_once, prepare_data
from app.models.enums import ModelFamily
from app.schemas.dataset import SchemaMapping
from app.services.diagnostics import compute_health_report
from app.services.ingestion import convert_to_parquet, profile_upload
from tests.conftest import rossmann_mapping

HORIZON = 42


@pytest.fixture(scope="module")
def canonical(rossmann_subset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("canon") / "canonical.parquet"
    stats = convert_to_parquet(
        rossmann_subset,
        "csv",
        SchemaMapping.model_validate(rossmann_mapping()),
        out,
        200_000,
        lambda f, s: None,
    )
    assert stats["store_codes"] == ["1", "2", "3"]
    assert stats["series_count"] == 3
    return out


def test_profile_suggests_rossmann_mapping(rossmann_subset: Path) -> None:
    profile, mapping = profile_upload(rossmann_subset, "csv")
    assert mapping["date_column"] == "Date"
    assert mapping["target_column"] == "Sales"
    assert mapping["store_column"] == "Store"
    assert mapping["customers_column"] == "Customers"
    assert "Promo" in mapping["known_future_columns"]
    assert mapping["promo_column"] == "Promo"
    assert mapping["open_column"] == "Open"
    assert {c["name"] for c in profile["columns"]} >= {"Store", "Date", "Sales"}


def test_health_report(canonical: Path) -> None:
    report = compute_health_report(canonical)
    assert report["series_count"] == 3
    assert report["duplicate_rows"] == 0
    assert 0 <= report["score"] <= 100
    assert report["daily_total"]


@pytest.mark.parametrize(
    "family",
    [
        ModelFamily.SEASONAL_NAIVE,
        ModelFamily.LIGHTGBM,
        pytest.param(
            ModelFamily.TORCH_MLP,
            marks=pytest.mark.skipif(
                not FAMILIES[ModelFamily.TORCH_MLP].is_available(), reason="torch not installed"
            ),
        ),
    ],
)
def test_train_and_forecast(canonical: Path, family: ModelFamily, progress, tmp_path: Path) -> None:  # noqa: ANN001
    data = prepare_data(canonical, rossmann_mapping(), HORIZON, HORIZON, None)
    params = {ModelFamily.LIGHTGBM: {"num_boost_round": 200}, ModelFamily.TORCH_MLP: {"max_epochs": 10}}.get(
        family, {}
    )
    model, logs, preds = _fit_once(family, data, params, "cpu", progress)
    assert logs and len(preds) == len(data.y_val)
    wape = np.abs(preds - data.y_val).sum() / np.abs(data.y_val).sum()
    assert wape < 0.6, f"{family} WAPE unexpectedly high: {wape:.3f}"

    path = tmp_path / f"model.{FAMILIES[family].artifact_ext}"
    model.save(path)
    loaded = ForecastModel.load(path)
    np.testing.assert_allclose(loaded.predict(data.X_val), model.predict(data.X_val))

    out, summary = run_forecast(
        model=loaded,
        metrics={},
        parquet_path=canonical,
        store_code=None,
        horizon=HORIZON,
        # The data's final week is a promo week, so carry-forward would already set Promo=1;
        # default it to 0 so the event override is a genuine change vs baseline.
        covariate_defaults={"Promo": 0},
        events=[
            {
                "name": "promo-week",
                "start_date": "2015-08-03",
                "end_date": "2015-08-07",
                "store_codes": None,
                "overrides": {"Promo": 1},
            }
        ],
        compare_to_baseline=True,
        promo_column="Promo",
        open_column="Open",
        progress=progress,
    )
    assert len(out) == 3 * HORIZON
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
    assert pd.Timestamp(min(out["ds"])) == pd.Timestamp("2015-08-01")
    assert summary["store_count"] == 3
    if family == ModelFamily.LIGHTGBM:
        impact = summary["event_impact"]["promo-week"]
        assert impact["uplift"] > 0, "promo override should raise forecast sales"


def test_tune_scores_trials_on_inner_split(canonical: Path, progress, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Optuna trials must never see the final validation window; the tuned GBDT trains a fixed
    number of rounds without early stopping on it."""
    import uuid

    from app.ml import training
    from app.services.storage import LocalStorage

    monkeypatch.setattr(training, "get_storage", lambda: LocalStorage(str(tmp_path)))
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    real_fit = training._fit_once

    def spy(family, data, params, device, prog):  # noqa: ANN001, ANN202
        yd = data.X_val["year"].astype(int) * 1000 + data.X_val["dayofyear"].astype(int)
        d = pd.to_datetime(yd.astype(str), format="%Y%j")
        seen.append((d.min(), d.max()))
        return real_fit(family, data, params, device, prog)

    monkeypatch.setattr(training, "_fit_once", spy)
    result = training.tune(
        org_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        parquet_path=canonical,
        mapping=rossmann_mapping(),
        family=ModelFamily.LIGHTGBM,
        base_params={"num_boost_round": 150, "quantiles": False},
        search_space={"num_leaves": {"type": "int", "low": 15, "high": 63}},
        n_trials=2,
        timeout_s=None,
        metric="wape",
        horizon=HORIZON,
        validation_days=HORIZON,
        store_code=None,
        progress=progress,
    )
    final_start = seen[-1][0]
    assert len(seen) == 3 and all(hi < final_start for _, hi in seen[:-1])  # trials never saw final window
    assert result["hyperparameters"]["early_stopping_rounds"] == 0
    assert result["metrics"]["tuning"]["scheme"].startswith("inner split")
