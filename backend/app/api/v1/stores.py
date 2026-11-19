from __future__ import annotations

import io
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

from app.api.deps import AdminPrincipal, AnalystPrincipal, CurrentPrincipal, DbSession
from app.core.exceptions import DomainValidationError, NotFoundError
from app.models import FinancialStaffingConfig, Store
from app.models.enums import TargetType
from app.schemas.common import Page, PageParams
from app.schemas.store import (
    FinancialStaffingConfigIn,
    FinancialStaffingConfigRead,
    StoreCreate,
    StoreRead,
    StoreUpdate,
)
from app.services.tenancy import get_owned, paginate, scoped

router = APIRouter(prefix="/stores", tags=["stores"])


def _read(store: Store) -> StoreRead:
    out = StoreRead.model_validate(store)
    out.has_financial_config = store.financial_config is not None
    return out


@router.get("", response_model=Page[StoreRead])
async def list_stores(
    principal: CurrentPrincipal, db: DbSession, page: PageParams = Depends(), q: str | None = None
) -> Page[StoreRead]:
    stmt = scoped(Store, principal.organization_id).options(selectinload(Store.financial_config))
    if q:
        stmt = stmt.where(Store.name.ilike(f"%{q}%") | Store.code.ilike(f"%{q}%"))
    items, total = await paginate(db, stmt.order_by(Store.code), page.limit, page.offset)
    return Page(items=[_read(s) for s in items], total=total, limit=page.limit, offset=page.offset)


@router.post("", response_model=StoreRead, status_code=status.HTTP_201_CREATED)
async def create_store(body: StoreCreate, principal: AdminPrincipal, db: DbSession) -> StoreRead:
    store = Store(organization_id=principal.organization_id, **body.model_dump(mode="json"))
    db.add(store)
    await db.commit()
    await db.refresh(store, ["financial_config"])
    return _read(store)


_IMPORT_ALIASES = {
    # Rossmann store.csv -> Store fields
    "Store": "code",
    "StoreType": "store_type",
    "Assortment": "assortment",
    "CompetitionDistance": "competition_distance",
    "CompetitionOpenSinceMonth": "competition_open_since_month",
    "CompetitionOpenSinceYear": "competition_open_since_year",
    "Promo2": "promo2",
    "Promo2SinceWeek": "promo2_since_week",
    "Promo2SinceYear": "promo2_since_year",
    "PromoInterval": "promo_interval",
}


@router.post("/import", response_model=list[StoreRead])
async def import_stores(
    principal: AdminPrincipal,
    db: DbSession,
    file: UploadFile = File(...),
    currency: str = Form("EUR"),
    target_type: TargetType = Form(TargetType.SALES),
) -> list[StoreRead]:
    """Upsert store metadata from CSV (Rossmann `store.csv` columns or Store field names)."""
    raw = await file.read()
    df = await run_in_threadpool(lambda: pd.read_csv(io.BytesIO(raw), dtype={"Store": str, "code": str}))
    df = df.rename(columns=_IMPORT_ALIASES)
    if "code" not in df.columns:
        raise DomainValidationError("CSV needs a 'code' (or Rossmann 'Store') column")
    allowed = set(StoreUpdate.model_fields) - {"attributes"}
    existing = {s.code: s for s in (await db.execute(scoped(Store, principal.organization_id))).scalars()}
    touched: list[Store] = []
    for rec in df.astype(object).where(df.notna(), None).to_dict(orient="records"):
        code = str(rec.pop("code"))
        fields = {k: v for k, v in rec.items() if k in allowed}
        if "promo2" in fields:
            fields["promo2"] = bool(fields["promo2"])
        fields = StoreUpdate.model_validate(fields).model_dump(exclude_none=True, mode="json")
        store = existing.get(code)
        if store is None:
            store = Store(
                organization_id=principal.organization_id,
                code=code,
                name=f"Store {code}",
                currency=currency.upper(),
                target_type=target_type.value,
                promo2=False,
                timezone="Europe/Berlin",
                attributes={},
            )
            db.add(store)
        for k, v in fields.items():
            setattr(store, k, v)
        touched.append(store)
    await db.commit()
    for s in touched:
        await db.refresh(s, ["financial_config"])
    return [_read(s) for s in touched]


@router.get("/{store_id}", response_model=StoreRead)
async def get_store(store_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession) -> StoreRead:
    store = await get_owned(db, Store, store_id, principal.organization_id)
    await db.refresh(store, ["financial_config"])
    return _read(store)


@router.patch("/{store_id}", response_model=StoreRead)
async def update_store(
    store_id: uuid.UUID, body: StoreUpdate, principal: AdminPrincipal, db: DbSession
) -> StoreRead:
    store = await get_owned(db, Store, store_id, principal.organization_id)
    for k, v in body.model_dump(exclude_unset=True, mode="json").items():
        setattr(store, k, v)
    await db.commit()
    await db.refresh(store, ["financial_config"])
    return _read(store)


@router.delete("/{store_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_store(store_id: uuid.UUID, principal: AdminPrincipal, db: DbSession) -> Response:
    store = await get_owned(db, Store, store_id, principal.organization_id)
    await db.delete(store)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{store_id}/financial-config", response_model=FinancialStaffingConfigRead)
async def get_financial_config(
    store_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession
) -> FinancialStaffingConfig:
    await get_owned(db, Store, store_id, principal.organization_id)
    cfg = (
        await db.execute(select(FinancialStaffingConfig).where(FinancialStaffingConfig.store_id == store_id))
    ).scalar_one_or_none()
    if cfg is None:
        raise NotFoundError("Financial & staffing config not set for this store")
    return cfg


@router.put("/{store_id}/financial-config", response_model=FinancialStaffingConfigRead)
async def upsert_financial_config(
    store_id: uuid.UUID, body: FinancialStaffingConfigIn, principal: AnalystPrincipal, db: DbSession
) -> FinancialStaffingConfig:
    await get_owned(db, Store, store_id, principal.organization_id)
    cfg = (
        await db.execute(select(FinancialStaffingConfig).where(FinancialStaffingConfig.store_id == store_id))
    ).scalar_one_or_none()
    if cfg is None:
        cfg = FinancialStaffingConfig(organization_id=principal.organization_id, store_id=store_id)
        db.add(cfg)
    for k, v in body.model_dump().items():
        setattr(cfg, k, v)
    await db.commit()
    await db.refresh(cfg)
    return cfg
