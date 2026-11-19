from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError, PermissionDeniedError
from app.core.security import decode_token
from app.db.session import get_db
from app.models import User
from app.models.enums import ROLE_RANK, UserRole

DbSession = Annotated[AsyncSession, Depends(get_db)]
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user: User
    organization_id: uuid.UUID
    role: UserRole

    @property
    def user_id(self) -> uuid.UUID:
        return self.user.id


async def get_principal(
    db: DbSession, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise AuthenticationError("Missing bearer token")
    payload = decode_token(creds.credentials, "access")
    user = await db.get(User, uuid.UUID(payload["sub"]))
    if user is None or not user.is_active:
        raise AuthenticationError("User is inactive or no longer exists")
    # The org in the token must still match the user's org (guards against stale tokens).
    if str(user.organization_id) != payload.get("org"):
        raise AuthenticationError("Token organization mismatch")
    return Principal(user=user, organization_id=user.organization_id, role=UserRole(user.role))


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def require_role(minimum: UserRole) -> Callable[[Principal], Principal]:
    def _check(principal: CurrentPrincipal) -> Principal:
        if ROLE_RANK[principal.role] < ROLE_RANK[minimum]:
            raise PermissionDeniedError(f"Requires role '{minimum.value}' or higher")
        return principal

    return _check


AnalystPrincipal = Annotated[Principal, Depends(require_role(UserRole.ANALYST))]
AdminPrincipal = Annotated[Principal, Depends(require_role(UserRole.ADMIN))]
