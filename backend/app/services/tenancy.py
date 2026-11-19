"""Tenant-scoped data access helpers.

Rule: never load a tenant-owned row by primary key alone. Always go through `get_owned`
(or add `organization_id == principal.organization_id` to the query) so one organization
can never read or mutate another's data. Missing and foreign rows both return 404.
"""

from __future__ import annotations

import uuid
from typing import Any, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError

M = TypeVar("M")


async def get_owned(session: AsyncSession, model: type[M], obj_id: uuid.UUID, org_id: uuid.UUID) -> M:
    stmt = select(model).where(model.id == obj_id, model.organization_id == org_id)  # type: ignore[attr-defined]
    obj = (await session.execute(stmt)).scalar_one_or_none()
    if obj is None:
        raise NotFoundError(f"{model.__name__} {obj_id} not found")
    return obj


def scoped(model: type[M], org_id: uuid.UUID) -> Select[Any]:
    return select(model).where(model.organization_id == org_id)  # type: ignore[attr-defined]


async def paginate(
    session: AsyncSession, stmt: Select[Any], limit: int, offset: int
) -> tuple[list[Any], int]:
    total = (
        await session.execute(select(func.count()).select_from(stmt.order_by(None).subquery()))
    ).scalar_one()
    rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars().all()
    return list(rows), int(total)
