"""Application settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App -------------------------------------------------------------------------------
    app_name: str = "StorePulse AI"
    environment: Literal["local", "staging", "production", "test"] = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    # Comma-separated list of allowed origins for CORS.
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    # --- Auth ------------------------------------------------------------------------------
    jwt_secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_minutes: int = 60 * 24 * 7

    # --- PostgreSQL ------------------------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "storepulse"
    postgres_password: str = "storepulse"
    postgres_db: str = "storepulse"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- Redis / Celery --------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    # Broker override, e.g. sqla+postgresql+psycopg://user:pw@host:port/db to use PostgreSQL as the
    # broker (kombu SQLAlchemy transport): no Redis - for lean local runs (scripts/local_stack.py).
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False

    # --- Artifact storage ------------------------------------------------------------------
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_root: str = "./storage"
    storage_cache_dir: str = "./storage/.cache"
    s3_bucket: str = "storepulse-artifacts"
    s3_endpoint_url: str | None = None
    s3_region: str = "eu-central-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None

    # --- Ingestion / ML --------------------------------------------------------------------
    max_upload_mb: int = 2048
    ingestion_chunk_rows: int = 500_000
    training_device: Literal["auto", "cpu", "cuda"] = "auto"
    optuna_default_trials: int = 30
    optuna_storage_url: str | None = None  # e.g. postgresql+psycopg://... to persist studies

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def _pg_dsn(self, driver: str) -> str:
        return (
            f"postgresql+{driver}://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_async(self) -> str:
        """Used by the FastAPI app (asyncpg)."""
        return self._pg_dsn("asyncpg")

    @property
    def database_url_sync(self) -> str:
        """Used by Celery workers and Alembic (psycopg 3)."""
        return self._pg_dsn("psycopg")

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str | None:
        """Task results are never read (job state lives in PostgreSQL); Redis stores them when it is the
        broker, otherwise results are disabled unless CELERY_RESULT_BACKEND is set."""
        if self.celery_result_backend:
            return self.celery_result_backend
        return self.redis_url if self.broker_url.startswith("redis") else None


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
