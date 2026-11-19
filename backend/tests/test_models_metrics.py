"""Metrics (RMSSE/R2/pinball), leak-free feature guarantees, Tweedie/quantile LightGBM, GRU and ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.families import FAMILIES, ForecastModel, is_intermittent
from app.ml.features import FeatureSpec, build_features, complete_calendar
from app.ml.metrics import interval_metrics, pinball_loss, regression_metrics, series_scales
from app.ml.training import PreparedData
from app.models.enums import ModelFamily

TORCH = FAMILIES[ModelFamily.TORCH_MLP].is_available()


# --- metrics ---------------------------------------------------------------------------


def test_rmsse_r2_and_pinball() -> None:
    train = pd.DataFrame({"s": ["a"] * 5, "ds": pd.date_range("2020-01-01", periods=5), "y": [0, 0, 1, 2, 3]})
    scale = series_scales(train, ["s"])
    assert scale["a"] == pytest.approx(1.0)  # starts at first non-zero: diffs 1, 1
    m = regression_metrics(
        np.array([4.0, 4.0]), np.array([3.0, 5.0]), series=np.array(["a", "a"]), scale=np.array([1.0, 1.0])
    )
    assert m["rmsse"] == pytest.approx(1.0)
    m2 = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert m2["r2"] == pytest.approx(1.0) and m2["rmsse"] is None
    assert pinball_loss(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball_loss(np.array([10.0]), np.array([12.0]), 0.9) == pytest.approx(0.2)
    iv = interval_metrics(np.array([1.0, 5.0, 9.0]), np.array([0.0, 0.0, 0.0]), np.array([2.0, 6.0, 8.0]))
    assert iv["coverage"] == pytest.approx(2 / 3)


# --- features --------------------------------------------------------------------------


def _panel(n_days: int = 220, seed: int = 0) -> pd.DataFrame:
    """Two stores x two items, intermittent demand, a promo flag and a hierarchy column."""
    rng = np.random.default_rng(seed)
    rows = []
    for store in ("S1", "S2"):
        for item, dept in (("A", "D1"), ("B", "D2")):
            ds = pd.date_range("2021-01-01", periods=n_days)
            promo = (rng.random(n_days) < 0.2).astype(float)
            y = rng.poisson(0.6 + 1.5 * promo) * (rng.random(n_days) < 0.7)
            rows.append(
                pd.DataFrame(
                    {
                        "ds": ds,
                        "store_code": store,
                        "item_id": item,
                        "dept": dept,
                        "y": y.astype(float),
                        "promo": promo,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _spec(seq_len: int = 0) -> FeatureSpec:
    spec = FeatureSpec.from_mapping(
        {
            "known_future_columns": ["promo"],
            "categorical_columns": ["dept"],
            "id_columns": ["dept"],
            "promo_column": "promo",
        },
        horizon=14,
    )
    spec.seq_len = seq_len
    return spec


def test_target_features_are_leak_free() -> None:
    """Scramble every target after a cutoff: no target-derived feature may change for rows whose
    lag origin (t - lag_offset) is on or before the cutoff."""
    df = complete_calendar(_panel())
    spec = _spec(seq_len=28)
    base = build_features(df, spec, fit_categories=True)
    cutoff = pd.Timestamp("2021-05-01")
    scrambled = df.copy()
    after = scrambled["ds"] > cutoff
    scrambled.loc[after, "y"] = np.random.default_rng(1).poisson(5, after.sum()).astype(float)
    spec2 = _spec(seq_len=28)
    spec2.category_levels, spec2.agg_groups = spec.category_levels, spec.agg_groups
    pert = build_features(scrambled, spec2, fit_categories=False)

    safe = base["ds"] <= cutoff + pd.Timedelta(days=spec.lag_offset)
    target_cols = [
        c
        for c in spec.model_columns
        if c.startswith(
            ("lag_", "roll_", "dow_mean", "nz_", "days_since_sale", "agg_", "seq_", "yoy_", "tgt_")
        )
    ]
    assert {"nz_rate_28", "days_since_sale", "agg_store_code", "agg_store_code_dept", "seq_0"} <= set(
        target_cols
    )
    for c in target_cols:
        pd.testing.assert_series_equal(base.loc[safe, c], pert.loc[safe, c], check_names=False, obj=c)
    # Sanity: the scramble does move features for rows past the safe window.
    assert not base.loc[~safe, "roll_mean_28"].equals(pert.loc[~safe, "roll_mean_28"])


def test_intermittency_and_promo_features() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2021-01-01", periods=12),
            "store_code": "S",
            "item_id": "A",
            "y": [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0.0],
            "promo": [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0.0],
        }
    )
    spec = FeatureSpec(horizon=1, lag_offset=1, known_future=["promo"], categorical=[], promo_column="promo")
    f = build_features(df, spec, fit_categories=True)
    # Origin is t-1: at t=2 (origin day 1, sale of 3) -> 0 days; at t=4 (origin day 3) -> 2 days.
    assert f["days_since_sale"].iloc[2] == 0 and f["days_since_sale"].iloc[4] == 2
    assert f["promo_streak"].tolist()[:8] == [0, 1, 2, 3, 0, 0, 1, 2]
    assert f["days_since_promo"].iloc[5] == 2


# --- models ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared() -> PreparedData:
    df = complete_calendar(_panel(n_days=260))
    spec = _spec(seq_len=28)
    f = build_features(df, spec, fit_categories=True)
    f = f[f[f"lag_{spec.lag_offset}"].notna()]
    cut = f["ds"].max() - pd.Timedelta(days=14)
    tr, val = f[f["ds"] <= cut], f[f["ds"] > cut]
    cols = spec.model_columns
    return PreparedData(
        tr[cols],
        tr["y"].to_numpy(),
        val[cols],
        val["y"].to_numpy(),
        val[["ds", "store_code", "item_id", "y"]],
        spec,
        val_scale=np.ones(len(val)),
    )


def test_lightgbm_auto_tweedie_and_quantiles(prepared: PreparedData, tmp_path: Path) -> None:
    assert is_intermittent(prepared.y_tr)
    m = FAMILIES[ModelFamily.LIGHTGBM](prepared.spec, {"num_boost_round": 60, "early_stopping_rounds": 20})
    m.fit(prepared.X_tr, prepared.y_tr, prepared.X_val, prepared.y_val, lambda e, f: None)
    assert m.params["resolved_objective"] == "tweedie"
    q = m.predict_quantiles(prepared.X_val)
    assert q is not None and set(q) == {0.1, 0.5, 0.9}
    assert (q[0.1] <= q[0.5]).all() and (q[0.5] <= q[0.9]).all()
    m.save(tmp_path / "m.pkl")
    again = ForecastModel.load(tmp_path / "m.pkl")
    np.testing.assert_allclose(again.predict_quantiles(prepared.X_val)[0.9], q[0.9])


@pytest.mark.skipif(not TORCH, reason="torch not installed")
def test_gru_poisson_roundtrip(prepared: PreparedData, tmp_path: Path) -> None:
    m = FAMILIES[ModelFamily.TORCH_GRU](prepared.spec, {"max_epochs": 2, "seq_len": 28})
    m.fit(prepared.X_tr, prepared.y_tr, prepared.X_val, prepared.y_val, lambda e, f: None)
    assert m.loss_kind == "poisson"  # auto for intermittent targets
    p = m.predict(prepared.X_val)
    assert np.isfinite(p).all() and (p >= 0).all()
    m.save(tmp_path / "m.pt")
    np.testing.assert_allclose(ForecastModel.load(tmp_path / "m.pt").predict(prepared.X_val), p, rtol=1e-5)


@pytest.mark.skipif(not TORCH, reason="torch not installed")
def test_ensemble_weights_comparison_and_pickle(prepared: PreparedData, tmp_path: Path) -> None:
    params = {
        "members": ["lightgbm", "torch_gru"],
        "member_params": {
            "lightgbm": {"num_boost_round": 40, "quantiles": False},
            "torch_gru": {"max_epochs": 1, "seq_len": 28},
        },
    }
    m = FAMILIES[ModelFamily.ENSEMBLE](prepared.spec, params)
    m.fit(prepared.X_tr, prepared.y_tr, prepared.X_val, prepared.y_val, lambda e, f: None)
    assert m.weights.sum() == pytest.approx(1.0) and (m.weights >= 0).all()
    assert set(m.comparison["models"]) == {"lightgbm", "torch_gru", "equal_weight_blend", "fitted_blend"}
    p = m.predict(prepared.X_val)
    m.save(tmp_path / "e.pkl")  # torch member must survive pickling
    np.testing.assert_allclose(ForecastModel.load(tmp_path / "e.pkl").predict(prepared.X_val), p, rtol=1e-5)


def test_closed_rule_and_log_target(tmp_path: Path) -> None:
    """Closed days (open flag = 0) are excluded from training and forecast as exactly 0;
    target_transform=log1p trains on log scale and predicts on the original scale."""
    rng = np.random.default_rng(3)
    ds = pd.date_range("2021-01-01", periods=300)
    frames = []
    for store in ("1", "2"):
        open_ = (ds.dayofweek != 6).astype(float)
        promo = (rng.random(len(ds)) < 0.4).astype(float)
        y = open_ * (5000 + 1500 * promo + rng.normal(0, 200, len(ds)))
        frames.append(
            pd.DataFrame(
                {"ds": ds, "store_code": store, "item_id": "", "y": y, "Open": open_, "Promo": promo}
            )
        )
    df = complete_calendar(pd.concat(frames, ignore_index=True))
    spec = FeatureSpec.from_mapping(
        {"known_future_columns": ["Open", "Promo"], "open_column": "Open", "promo_column": "Promo"},
        horizon=14,
    )
    f = build_features(df, spec, fit_categories=True)
    f = f[f[f"lag_{spec.lag_offset}"].notna()]
    assert {"roll_mean_open_28", "tgt_promo_lift_91", "dow_mean_13w"} <= set(spec.feature_columns)
    cut = f["ds"].max() - pd.Timedelta(days=14)
    tr, val = f[(f["ds"] <= cut) & (f["Open"] != 0)], f[f["ds"] > cut]
    cols = spec.model_columns
    m = FAMILIES[ModelFamily.LIGHTGBM](spec, {"num_boost_round": 80, "target_transform": "log1p"})
    m.fit(tr[cols], tr["y"].to_numpy(), val[cols], val["y"].to_numpy(), lambda e, f_: None)
    p = m.predict(val[cols])
    closed = val["Open"].to_numpy() == 0
    assert closed.any() and (p[closed] == 0).all()
    assert np.median(p[~closed]) > 1000  # back on the original scale, not log
    m.save(tmp_path / "m.pkl")
    np.testing.assert_allclose(ForecastModel.load(tmp_path / "m.pkl").predict(val[cols]), p)


@pytest.mark.skipif(not TORCH, reason="torch not installed")
@pytest.mark.parametrize("family", [ModelFamily.TORCH_MLP, ModelFamily.TORCH_GRU])
@pytest.mark.parametrize("transform", ["log1p", "standard"])
def test_torch_closed_rule_transforms_and_quantiles(
    family: ModelFamily, transform: str, tmp_path: Path
) -> None:
    """Neural nets: exact zeros on closed days (point and P10/P50/P90), both target transforms decode
    back to sales scale, quantile heads are ordered/non-negative, and GRU windows carry open/promo."""
    rng = np.random.default_rng(5)
    ds = pd.date_range("2021-01-01", periods=300)
    frames = []
    for store in ("1", "2", "3"):
        open_ = (ds.dayofweek != 6).astype(float)
        promo = (rng.random(len(ds)) < 0.4).astype(float)
        y = open_ * (4000 + 1000 * int(store) + 1500 * promo + rng.normal(0, 200, len(ds)))
        frames.append(
            pd.DataFrame(
                {"ds": ds, "store_code": store, "item_id": "", "y": y, "Open": open_, "Promo": promo}
            )
        )
    df = complete_calendar(pd.concat(frames, ignore_index=True))
    spec = FeatureSpec.from_mapping(
        {"known_future_columns": ["Open", "Promo"], "open_column": "Open", "promo_column": "Promo"},
        horizon=14,
    )
    params = {"max_epochs": 4, "target_transform": transform, "quantiles": True, "seq_len": 14}
    spec.seq_len = FAMILIES[family].required_seq_len(params)
    spec.seq_covariates = spec.default_seq_covariates() if spec.seq_len else []
    f = build_features(df, spec, fit_categories=True)
    f = f[f[f"lag_{spec.lag_offset}"].notna()]
    cut = f["ds"].max() - pd.Timedelta(days=14)
    tr, val = f[(f["ds"] <= cut) & (f["Open"] != 0)], f[f["ds"] > cut]
    cols = spec.model_columns
    m = FAMILIES[family](spec, params)
    m.fit(tr[cols], tr["y"].to_numpy(), val[cols], val["y"].to_numpy(), lambda e, f_: None)
    assert m.loss_kind == "mse" and m.target_transform == transform
    assert m.num_log.any(), "sales-scale inputs (lags) should be log-compressed"
    if family == ModelFamily.TORCH_GRU:
        assert m.seq_channels == ["Open", "Promo"]
    p = m.predict(val[cols])
    q = m.predict_quantiles(val[cols])
    closed = val["Open"].to_numpy() == 0
    assert closed.any() and (p[closed] == 0).all()
    assert q is not None and all((v[closed] == 0).all() and (v >= 0).all() for v in q.values())
    assert (q[0.1] <= q[0.5]).all() and (q[0.5] <= q[0.9]).all()
    wape = np.abs(p - val["y"].to_numpy()).sum() / val["y"].sum()
    assert wape < 0.5, f"{family}/{transform} WAPE {wape:.3f}: not decoded back to sales scale?"
    m.save(tmp_path / "m.pt")
    again = ForecastModel.load(tmp_path / "m.pt")
    np.testing.assert_allclose(again.predict(val[cols]), p, rtol=1e-5)
    np.testing.assert_allclose(again.predict_quantiles(val[cols])[0.9], q[0.9], rtol=1e-5)
