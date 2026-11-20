"""Store Autopilot: plan a store's next day / week / month / custom window (optionally a what-if
scenario) -> footfall, P&L, shift plan, prioritized actions, and PDF / Excel / CSV exports."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.api.deps import AnalystPrincipal, CurrentPrincipal, DbSession
from app.api.v1._progress import progress_stream
from app.core.exceptions import DomainValidationError, PreconditionFailedError
from app.models import AutopilotRun, Dataset, FinancialStaffingConfig, ModelRun, Organization, Store
from app.models.enums import JobStatus, TargetType
from app.schemas.autopilot import AutopilotRequest, AutopilotRunDetail, AutopilotRunRead
from app.schemas.common import JobAccepted, Page, PageParams
from app.services import autopilot_report
from app.services.tenancy import get_owned, paginate, scoped
from app.workers import dispatch

router = APIRouter(prefix="/autopilot", tags=["autopilot"])


@router.post("/runs", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_autopilot_run(
    body: AutopilotRequest, principal: AnalystPrincipal, db: DbSession
) -> JobAccepted:
    """Queue an Autopilot plan. The model must be finished; the store needs a financial config."""
    org = principal.organization_id
    store = await get_owned(db, Store, body.store_id, org)
    run = await get_owned(db, ModelRun, body.model_run_id, org)
    if run.status != JobStatus.SUCCEEDED or not run.artifact_path:
        raise PreconditionFailedError(
            "Model run has not finished successfully", details={"status": run.status}
        )
    if run.store_id and run.store_id != store.id:
        raise DomainValidationError("This model was trained on another single store; use a multi-store model")
    dataset = await get_owned(db, Dataset, run.dataset_id, org)
    cfg = (
        await db.execute(select(FinancialStaffingConfig).where(FinancialStaffingConfig.store_id == store.id))
    ).scalar_one_or_none()
    if cfg is None:
        raise PreconditionFailedError(
            "Set the store's financial & staffing config before running the Autopilot"
        )
    mapping = dataset.schema_mapping or {}
    if (
        store.target_type == TargetType.SALES
        and not mapping.get("customers_column")
        and cfg.avg_transaction_value is None
    ):
        raise DomainValidationError(
            "Footfall needs a customers column in the dataset or avg_transaction_value on the store config"
        )
    sc = body.scenario
    for lv in sc.levers:
        if lv.promo is not None and not mapping.get("promo_column"):
            raise DomainValidationError(
                f"Lever '{lv.name}' toggles promo but the dataset maps no promo column"
            )
        if lv.open is not None and not mapping.get("open_column"):
            raise DomainValidationError(f"Lever '{lv.name}' toggles open but the dataset maps no open column")

    ap = AutopilotRun(
        organization_id=org,
        store_id=store.id,
        model_run_id=run.id,
        dataset_id=dataset.id,
        created_by_id=principal.user_id,
        name=body.name,
        horizon=body.horizon,
        start_date=body.start_date,
        end_date=body.end_date,
        scenario=sc.model_dump(mode="json"),
        summary={},
        result={},
        status=JobStatus.QUEUED.value,
        progress_pct=0.0,
        stage="queued",
    )
    db.add(ap)
    await db.commit()
    ap.celery_task_id = dispatch.enqueue(dispatch.TASK_AUTOPILOT, ap.id, dispatch.QUEUE_FORECAST)
    await db.commit()
    return JobAccepted(id=ap.id, status=ap.status, celery_task_id=ap.celery_task_id)


@router.get("/runs", response_model=Page[AutopilotRunRead])
async def list_autopilot_runs(
    principal: CurrentPrincipal,
    db: DbSession,
    page: PageParams = Depends(),
    store_id: uuid.UUID | None = None,
) -> Page[AutopilotRunRead]:
    stmt = scoped(AutopilotRun, principal.organization_id)
    if store_id:
        stmt = stmt.where(AutopilotRun.store_id == store_id)
    items, total = await paginate(db, stmt.order_by(AutopilotRun.created_at.desc()), page.limit, page.offset)
    return Page(
        items=[AutopilotRunRead.model_validate(i) for i in items],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/runs/{run_id}", response_model=AutopilotRunDetail)
async def get_autopilot_run(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> AutopilotRun:
    return await get_owned(db, AutopilotRun, run_id, principal.organization_id)


@router.get("/runs/{run_id}/events")
async def autopilot_events(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession):  # noqa: ANN201
    await get_owned(db, AutopilotRun, run_id, principal.organization_id)
    return progress_stream(AutopilotRun, run_id, principal.organization_id)


async def _finished(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> AutopilotRun:
    ap = await get_owned(db, AutopilotRun, run_id, principal.organization_id)
    if ap.status != JobStatus.SUCCEEDED or not ap.result:
        raise PreconditionFailedError("Autopilot run has not finished", details={"status": ap.status})
    return ap


def _download(content: bytes, filename: str, media_type: str) -> Response:
    return Response(
        content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


def _stem(ap: AutopilotRun) -> str:
    w = ap.result["window"]
    return f"autopilot_{ap.result['store_code']}_{w['start']}_{w['end']}"


@router.get("/runs/{run_id}/plan.csv")
async def autopilot_csv(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> Response:
    ap = await _finished(run_id, principal, db)
    return _download(autopilot_report.to_csv(ap.result), f"{_stem(ap)}.csv", "text/csv")


@router.get("/runs/{run_id}/plan.xlsx")
async def autopilot_xlsx(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> Response:
    ap = await _finished(run_id, principal, db)
    content = await run_in_threadpool(autopilot_report.to_xlsx, ap.result)
    return _download(
        content, f"{_stem(ap)}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@router.get("/runs/{run_id}/report.pdf")
async def autopilot_pdf(run_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> Response:
    ap = await _finished(run_id, principal, db)
    store = await get_owned(db, Store, ap.store_id, principal.organization_id)
    org = await db.get(Organization, principal.organization_id)
    pdf = await run_in_threadpool(
        autopilot_report.executive_pdf,
        ap.result,
        title=ap.name or f"Store Autopilot - {store.name}",
        org_name=org.name if org else "",
        store_name=store.name,
    )
    return _download(pdf, f"{_stem(ap)}.pdf", "application/pdf")
