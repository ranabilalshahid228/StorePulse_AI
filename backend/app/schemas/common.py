from __future__ import annotations

import uuid
from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class PageParams(BaseModel):
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class JobState(ORMModel):
    status: str
    progress_pct: float
    stage: str | None = None
    status_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class JobAccepted(BaseModel):
    id: uuid.UUID
    status: str
    celery_task_id: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict | list | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
