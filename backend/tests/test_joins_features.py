"""Auxiliary joins, mapping validation, vectorized calendar completion and store features."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.ml.features import FeatureSpec, build_features, complete_calendar, future_frame
from app.schemas.dataset import JoinSpec, SchemaMapping
from app.services.ingestion import coerce_covariates, convert_to_parquet
from app.services.joins import apply_joins, load_join

CAL, PRICE = uuid.uuid4(), uuid.uuid4()


@pytest.fixture
def m5_like(tmp_path: Path) -> dict:
    """Tiny M5-shaped fixture: 2 items x 10 days wide, a calendar and weekly prices."""
    days = 10
    sales = pd.DataFrame(
        {
            "id": ["A_S1", "B_S1"],
            "item_id": ["A", "B"],
            "dept_id": ["D1", "D2"],
            "store_id": ["S1", "S1"],
            **{f"d_{i}": [i, 2 * i] for i in range(1, days + 1)},
        }
    )
    dates = pd.date_range("2020-01-01", periods=days + 7)  # calendar extends into the future
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 100 + np.arange(len(dates)) // 7,
            "event_name_1": [None] * len(dates),
            "snap": (np.arange(len(dates)) % 2).astype(int),
        }
    )
    cal.loc[12, "event_name_1"] = "SuperBowl"  # a future date
    prices = pd.DataFrame(
        {
            "store_id": ["S1"] * 6,
            "item_id": ["A", "B"] * 3,
            "wm_yr_wk": [100, 100, 101, 101, 102, 102],
            "sell_price": [1.0, 2.0, 1.5, 2.5, 1.25, 2.25],
        }
    )
    paths = {}
    for name, df in (("sales", sales), ("cal", cal), ("prices", prices)):
        paths[name] = tmp_path / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    mapping = SchemaMapping(
        layout="wide",
        wide_value_prefix="d_",
        wide_start_date="2020-01-01",
        store_column="store_id",
        item_column="item_id",
        id_columns=["dept_id"],
        categorical_columns=["dept_id", "event_name_1"],
        known_future_columns=["snap", "sell_price"],
        helper_columns=["wm_yr_wk"],
        price_column="sell_price",
        joins=[
            JoinSpec(
                dataset_id=CAL,
                left_on=["ds"],
                right_on=["date"],
                columns=["wm_yr_wk", "event_name_1", "snap"],
            ),
            JoinSpec(
                dataset_id=PRICE,
                left_on=["store_code", "item_id", "wm_yr_wk"],
                right_on=["store_id", "item_id", "wm_yr_wk"],
                columns=["sell_price"],
            ),
        ],
    )
    joins = [
        load_join(mapping.joins[0], paths["cal"], "csv"),
        load_join(mapping.joins[1], paths["prices"], "csv"),
    ]
    return {"paths": paths, "mapping": mapping, "joins": joins, "tmp": tmp_path}


def test_wide_ingestion_with_chained_joins(m5_like: dict) -> None:
    out = m5_like["tmp"] / "canon.parquet"
    stats = convert_to_parquet(
        m5_like["paths"]["sales"],
        "csv",
        m5_like["mapping"],
        out,
        7,
        lambda f, s: None,
        joins=m5_like["joins"],
    )
    df = pd.read_parquet(out).sort_values(["item_id", "ds"]).reset_index(drop=True)
    assert stats["row_count"] == 20 and stats["series_count"] == 2
    assert "wm_yr_wk" not in df.columns  # helper dropped
    a = df[df["item_id"] == "A"]
    # Week 100 (days 1-7) price 1.0, week 101 (days 8-10) price 1.5 -> chained join via wm_yr_wk.
    assert a["sell_price"].tolist() == [1.0] * 7 + [1.5] * 3
    assert a["snap"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert set(df["dept_id"]) == {"D1", "D2"}
    assert df["y"].tolist()[:3] == [1.0, 2.0, 3.0]


def test_future_frame_uses_real_future_covariates(m5_like: dict) -> None:
    out = m5_like["tmp"] / "canon.parquet"
    mapping = m5_like["mapping"]
    convert_to_parquet(
        m5_like["paths"]["sales"], "csv", mapping, out, 100, lambda f, s: None, joins=m5_like["joins"]
    )
    hist = complete_calendar(pd.read_parquet(out))
    spec = FeatureSpec.from_mapping(mapping.model_dump(mode="json"), horizon=7)

    def enrich(frame: pd.DataFrame) -> pd.DataFrame:
        frame = apply_joins(frame, m5_like["joins"])
        return coerce_covariates(frame.drop(columns=mapping.helper_columns, errors="ignore"), mapping)

    fut = future_frame(hist, spec, 7, {}, enrich=enrich).sort_values(["item_id", "ds"])
    a = fut[fut["item_id"] == "A"].reset_index(drop=True)
    assert a["ds"].iloc[0] == pd.Timestamp("2020-01-11")
    # Calendar row 12 (2020-01-13) is a future event; other days have none (not carried forward).
    assert a.loc[a["ds"] == "2020-01-13", "event_name_1"].iloc[0] == "SuperBowl"
    assert a.loc[a["ds"] != "2020-01-13", "event_name_1"].isna().all()
    assert a["sell_price"].tolist() == [1.5] * 4 + [1.25] * 3  # planned price for week 102
    assert (a["dept_id"] == "D1").all()  # id categorical carried forward


def test_mapping_rejects_bad_joins() -> None:
    with pytest.raises(ValidationError, match="not available yet"):
        SchemaMapping(
            date_column="d",
            target_column="y",
            known_future_columns=["p"],
            helper_columns=["wk"],
            joins=[
                JoinSpec(dataset_id=PRICE, left_on=["wk"], right_on=["wk"], columns=["p"]),
                JoinSpec(dataset_id=CAL, left_on=["ds"], right_on=["date"], columns=["wk"]),
            ],
        )
    with pytest.raises(ValidationError, match="known_future_columns"):
        SchemaMapping(
            date_column="d",
            target_column="y",
            joins=[JoinSpec(dataset_id=CAL, left_on=["ds"], right_on=["date"], columns=["snap"])],
        )


def test_complete_calendar_fills_gaps_and_fast_path() -> None:
    df = pd.DataFrame(
        {
            "ds": ["2020-01-01", "2020-01-04", "2020-01-02", "2020-01-03"],
            "store_code": ["1", "1", "2", "2"],
            "item_id": [None] * 4,
            "y": [1.0, 4.0, 2.0, 3.0],
        }
    )
    full = complete_calendar(df)
    s1 = full[full["store_code"] == "1"]
    assert s1["ds"].dt.day.tolist() == [1, 2, 3, 4]
    assert s1["y"].isna().tolist() == [False, True, True, False]
    again = complete_calendar(full)
    assert len(again) == len(full)


def test_store_static_promo2_and_competition() -> None:
    static = {
        "2": {
            "store_type": "a",
            "assortment": "a",
            "competition_distance": 570.0,
            "competition_open_since_month": 11,
            "competition_open_since_year": 2007,
            "promo2": True,
            "promo2_since_week": 13,
            "promo2_since_year": 2010,
            "promo_interval": "Jan,Apr,Jul,Oct",
        }
    }
    dates = pd.to_datetime(["2010-01-15", "2015-06-15", "2015-07-15"])
    df = pd.DataFrame({"ds": dates, "store_code": "2", "item_id": "", "y": [1.0, 2.0, 3.0]})
    spec = FeatureSpec(horizon=7, lag_offset=7, known_future=[], categorical=[], store_static=static)
    f = build_features(df, spec, fit_categories=True)
    assert f["st_promo2_active"].tolist() == [0.0, 0.0, 1.0]  # before start / June / July
    assert f["st_competition_open_months"].tolist() == [26.0, 91.0, 92.0]
    assert {"st_store_type", "st_assortment", "st_promo2_weeks"} <= set(spec.feature_columns)
