from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

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
