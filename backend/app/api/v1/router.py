from fastapi import APIRouter

from app.api.v1 import auth, autopilot, datasets, finance, forecasts, health, model_runs, reports, stores

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(stores.router)
api_router.include_router(datasets.router)
api_router.include_router(model_runs.router)
api_router.include_router(forecasts.router)
api_router.include_router(finance.router)
api_router.include_router(reports.router)
api_router.include_router(autopilot.router)
