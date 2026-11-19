from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import pyarrow.dataset as pads
from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import AdminPrincipal, AnalystPrincipal, CurrentPrincipal, DbSession
from app.api.v1._progress import progress_stream
from app.core.config import get_settings
from app.core.exceptions import DomainValidationError, PreconditionFailedError
from app.models import Dataset, Store
from app.models.enums import DatasetStatus
from app.schemas.common import JobAccepted, Page, PageParams
from app.schemas.dataset import DatasetHealth, DatasetPreview, DatasetRead, SchemaMapping
from app.services.ingestion import profile_upload
from app.services.storage import ArtifactKeys, get_storage
from app.services.tenancy import get_owned, paginate, scoped
from app.workers import dispatch

router = APIRouter(prefix="/datasets", tags=["datasets"])
ALLOWED_EXT = {".csv": "csv", ".parquet": "parquet"}


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    principal: AnalystPrincipal,
    db: DbSession,
    file: Annotated[UploadFile, File(description="CSV or Parquet file")],
    name: Annotated[str, Form(min_length=1, max_length=200)],
    description: Annotated[str | None, Form()] = None,
    store_id: Annotated[uuid.UUID | None, Form()] = None,
) -> Dataset:
    """Upload a raw file, profile its columns and return a suggested schema mapping.

    The dataset stays in `mapping_required` until the mapping is confirmed via PUT /mapping.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise DomainValidationError(f"Unsupported file type '{ext}'. Use .csv or .parquet")
    if store_id:
        await get_owned(db, Store, store_id, principal.organization_id)

    storage = get_storage()
    ds = Dataset(
        id=uuid.uuid4(),
        organization_id=principal.organization_id,
        store_id=store_id,
        created_by_id=principal.user_id,
        name=name,
        description=description,
        source_filename=file.filename or f"upload{ext}",
        file_format=ALLOWED_EXT[ext],
        raw_path="",
        status=DatasetStatus.UPLOADED.value,
        progress_pct=0.0,
    )
    key = ArtifactKeys.dataset_raw(principal.organization_id, ds.id, ds.source_filename)
    size = await run_in_threadpool(storage.put_stream, key, file.file)
    if size > get_settings().max_upload_mb * 1024 * 1024:
        await run_in_threadpool(storage.delete_prefix, key)
        raise DomainValidationError(f"File exceeds the {get_settings().max_upload_mb} MB upload limit")
    local = await run_in_threadpool(storage.local_path, key)
    profile, suggestion = await run_in_threadpool(profile_upload, local, ds.file_format)

    ds.raw_path = key
    ds.size_bytes = size
    ds.column_profile = profile
    ds.suggested_mapping = suggestion
    ds.status = DatasetStatus.MAPPING_REQUIRED.value
    ds.stage = "awaiting_mapping"
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return ds


@router.get("", response_model=Page[DatasetRead])
async def list_datasets(
    principal: CurrentPrincipal,
    db: DbSession,
    page: PageParams = Depends(),
    store_id: uuid.UUID | None = None,
) -> Page[DatasetRead]:
    stmt = scoped(Dataset, principal.organization_id)
    if store_id:
        stmt = stmt.where(Dataset.store_id == store_id)
    items, total = await paginate(db, stmt.order_by(Dataset.created_at.desc()), page.limit, page.offset)
    return Page(
        items=[DatasetRead.model_validate(i) for i in items],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> Dataset:
    return await get_owned(db, Dataset, dataset_id, principal.organization_id)


@router.get("/{dataset_id}/events")
async def dataset_events(dataset_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession):  # noqa: ANN201
    await get_owned(db, Dataset, dataset_id, principal.organization_id)
    return progress_stream(Dataset, dataset_id, principal.organization_id)


@router.put("/{dataset_id}/mapping", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def confirm_mapping(
    dataset_id: uuid.UUID, mapping: SchemaMapping, principal: AnalystPrincipal, db: DbSession
) -> JobAccepted:
    """Validate the schema mapping and enqueue Parquet conversion + health diagnostics."""
    ds = await get_owned(db, Dataset, dataset_id, principal.organization_id)
    if ds.status in (DatasetStatus.QUEUED, DatasetStatus.PROCESSING):
        raise PreconditionFailedError("Dataset is already being processed")
    known = {c["name"] for c in ds.column_profile.get("columns", [])}
    if mapping.layout == "long":
        missing = sorted(set(mapping.referenced_columns()) - known)
        if missing:
            raise DomainValidationError("Mapped columns not in file", details={"missing": missing})
    ds.schema_mapping = mapping.model_dump(mode="json")
    ds.status = DatasetStatus.QUEUED.value
    ds.stage = "queued"
    ds.progress_pct = 0.0
    ds.error = None
    await db.commit()
    task_id = dispatch.enqueue(dispatch.TASK_INGEST, ds.id, dispatch.QUEUE_INGEST)
    ds.celery_task_id = task_id
    await db.commit()
    return JobAccepted(id=ds.id, status=ds.status, celery_task_id=task_id)


@router.get("/{dataset_id}/health", response_model=DatasetHealth)
async def dataset_health(dataset_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> DatasetHealth:
    ds = await get_owned(db, Dataset, dataset_id, principal.organization_id)
    report = ds.health_report
    return DatasetHealth(dataset_id=ds.id, generated_at=(report or {}).get("generated_at"), report=report)


@router.post(
    "/{dataset_id}/health/recompute", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED
)
async def recompute_health(dataset_id: uuid.UUID, principal: AnalystPrincipal, db: DbSession) -> JobAccepted:
    ds = await get_owned(db, Dataset, dataset_id, principal.organization_id)
    if ds.status != DatasetStatus.READY or not ds.parquet_path:
        raise PreconditionFailedError("Dataset must be ready before diagnostics can run")
    task_id = dispatch.enqueue(dispatch.TASK_HEALTH, ds.id, dispatch.QUEUE_INGEST)
    return JobAccepted(id=ds.id, status=ds.status, celery_task_id=task_id)


def _require_ready(ds: Dataset) -> str:
    if ds.status != DatasetStatus.READY or not ds.parquet_path:
        raise PreconditionFailedError("Dataset is not ready", details={"status": ds.status})
    return ds.parquet_path


def _jsonable(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview(
    dataset_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession, limit: int = Query(50, ge=1, le=1000)
) -> DatasetPreview:
    ds = await get_owned(db, Dataset, dataset_id, principal.organization_id)
    path = await run_in_threadpool(get_storage().local_path, _require_ready(ds))

    def _read() -> pd.DataFrame:
        return pads.dataset(path, format="parquet").head(limit).to_pandas()

    df = await run_in_threadpool(_read)
    return DatasetPreview(columns=list(df.columns), rows=_jsonable(df))


@router.get("/{dataset_id}/series")
async def series(
    dataset_id: uuid.UUID,
    principal: CurrentPrincipal,
    db: DbSession,
    store_code: str | None = None,
    item_id: str | None = None,
) -> dict[str, Any]:
    """Daily aggregated target (and customers if mapped) for EDA charts."""
    ds = await get_owned(db, Dataset, dataset_id, principal.organization_id)
    path = await run_in_threadpool(get_storage().local_path, _require_ready(ds))

    def _read() -> pd.DataFrame:
        filt = None
        if store_code:
            filt = pads.field("store_code") == store_code
        if item_id:
            f2 = pads.field("item_id") == item_id
            filt = f2 if filt is None else filt & f2
        d = pads.dataset(path, format="parquet")
        cols = [c for c in ("ds", "y", "customers") if c in d.schema.names]
        t = d.to_table(columns=cols, filter=filt).to_pandas()
        return t.groupby("ds", as_index=False).sum(numeric_only=True).sort_values("ds")

    df = await run_in_threadpool(_read)
    df["ds"] = pd.to_datetime(df["ds"])
    return {"store_code": store_code, "item_id": item_id, "points": _jsonable(df)}


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: uuid.UUID, principal: AdminPrincipal, db: DbSession) -> Response:
    ds = await get_owned(db, Dataset, dataset_id, principal.organization_id)
    prefix = f"orgs/{principal.organization_id}/datasets/{ds.id}"
    await db.delete(ds)
    await db.commit()
    await run_in_threadpool(get_storage().delete_prefix, prefix)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
