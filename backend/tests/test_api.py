from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from kombu.exceptions import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.forecasts import rank_products
from app.core.exceptions import AppError
from app.main import app
from app.models import ModelRun
from app.workers import dispatch

client = TestClient(app)


def test_openapi_lists_core_routes() -> None:
    paths = client.get("/api/v1/openapi.json").json()["paths"]
    for p in [
        "/api/v1/auth/login",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}/mapping",
        "/api/v1/datasets/{dataset_id}/health",
        "/api/v1/model-runs/train",
        "/api/v1/model-runs/tune",
        "/api/v1/model-runs/leaderboard",
        "/api/v1/forecasts",
        "/api/v1/forecasts/event",
        "/api/v1/forecasts/{forecast_id}/products",
        "/api/v1/forecasts/{forecast_id}/pnl",
        "/api/v1/forecasts/{forecast_id}/workforce",
        "/api/v1/reports/forecasts/{forecast_id}/store-plan.pdf",
        "/api/v1/autopilot/runs",
        "/api/v1/autopilot/runs/{run_id}",
        "/api/v1/autopilot/runs/{run_id}/report.pdf",
        "/api/v1/autopilot/runs/{run_id}/plan.xlsx",
        "/api/v1/autopilot/runs/{run_id}/plan.csv",
    ]:
        assert p in paths, p


def test_unauthenticated_request_returns_error_envelope() -> None:
    r = client.get("/api/v1/stores")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "unauthenticated"
    assert body["error"]["request_id"] == r.headers["x-request-id"]


def test_invalid_token_rejected() -> None:
    r = client.get("/api/v1/stores", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_token"


def test_validation_errors_use_envelope() -> None:
    r = client.post("/api/v1/auth/register", json={"email": "bad"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_dispatch_failure_marks_queued_job_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db = AsyncMock(spec=AsyncSession)
    job = ModelRun(id=uuid.uuid4(), status="queued")

    def unavailable(*args, **kwargs) -> str:
        raise OperationalError("broker unavailable")

    monkeypatch.setattr(dispatch, "enqueue", unavailable)
    with pytest.raises(AppError) as exc:
        await dispatch.enqueue_job(db, job, dispatch.TASK_TRAIN, dispatch.QUEUE_TRAIN)
    assert exc.value.status_code == 503
    assert exc.value.details == {"job_id": str(job.id)}
    statement = db.execute.await_args.args[0].compile()
    assert "model_runs.status =" in str(statement)
    assert {"status": "failed", "stage": "dispatch_failed"}.items() <= statement.params.items()
    db.commit.assert_awaited_once()


def test_forecast_product_ranking_aggregates_skus(tmp_path: Path) -> None:
    path = tmp_path / "forecast.parquet"
    pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "C", "", None],
            "yhat": [3.0, 4.0, 6.0, 7.0, 100.0, 100.0],
        }
    ).to_parquet(path, row_group_size=2)
    result = rank_products(path)
    assert result.product_count == 3
    assert [(item.item_id, item.forecast) for item in result.products] == [("A", 7), ("C", 7), ("B", 6)]
    pd.DataFrame({"item_id": [None, ""], "yhat": [7.0, 10.0]}).to_parquet(path)
    assert rank_products(path).products == []
