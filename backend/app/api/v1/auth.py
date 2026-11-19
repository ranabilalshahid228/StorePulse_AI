from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import AdminPrincipal, CurrentPrincipal, DbSession
from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, ConflictError, PermissionDeniedError
from app.core.security import create_token, decode_token, hash_password, verify_password
from app.models import Organization, User
from app.models.enums import UserRole
from app.schemas.auth import (
    LoginRequest,
    MeResponse,
    OrganizationRead,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserCreate,
    UserRead,
)

router = APIRouter(tags=["auth"])


def _tokens(user: User) -> TokenPair:
    kw = {"user_id": user.id, "organization_id": user.organization_id, "role": user.role}
    return TokenPair(
        access_token=create_token(**kw, token_type="access"),
        refresh_token=create_token(**kw, token_type="refresh"),
        expires_in=get_settings().access_token_expire_minutes * 60,
    )


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "org"


@router.post("/auth/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: DbSession) -> TokenPair:
    """Create a new organization with its first (owner) user."""
    if (await db.execute(select(User.id).where(User.email == body.email.lower()))).first():
        raise ConflictError("Email is already registered")
    org = Organization(
        name=body.organization_name,
        slug=f"{_slugify(body.organization_name)}-{uuid.uuid4().hex[:6]}",
        default_currency=body.default_currency.upper(),
        plan="trial",
        is_active=True,
    )
    db.add(org)
    await db.flush()
    user = User(
        organization_id=org.id,
        email=body.email.lower(),
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=UserRole.OWNER.value,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    return _tokens(user)


@router.post("/auth/login", response_model=TokenPair)
async def login(body: LoginRequest, db: DbSession) -> TokenPair:
    user = (await db.execute(select(User).where(User.email == body.email.lower()))).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise AuthenticationError("Invalid email or password", code="invalid_credentials")
    if not user.is_active:
        raise AuthenticationError("Account is disabled", code="account_disabled")
    return _tokens(user)


@router.post("/auth/refresh", response_model=TokenPair)
async def refresh(body: RefreshRequest, db: DbSession) -> TokenPair:
    payload = decode_token(body.refresh_token, "refresh")
    user = await db.get(User, uuid.UUID(payload["sub"]))
    if user is None or not user.is_active:
        raise AuthenticationError("User is inactive or no longer exists")
    return _tokens(user)


@router.get("/auth/me", response_model=MeResponse)
async def me(principal: CurrentPrincipal, db: DbSession) -> MeResponse:
    org = await db.get(Organization, principal.organization_id)
    return MeResponse(
        user=UserRead.model_validate(principal.user), organization=OrganizationRead.model_validate(org)
    )


@router.get("/users", response_model=list[UserRead])
async def list_users(principal: AdminPrincipal, db: DbSession) -> list[User]:
    stmt = select(User).where(User.organization_id == principal.organization_id).order_by(User.created_at)
    return list((await db.execute(stmt)).scalars())


@router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(body: UserCreate, principal: AdminPrincipal, db: DbSession) -> User:
    if body.role == UserRole.OWNER and principal.role != UserRole.OWNER:
        raise PermissionDeniedError("Only owners can create owners")
    if (await db.execute(select(User.id).where(User.email == body.email.lower()))).first():
        raise ConflictError("Email is already registered")
    user = User(
        organization_id=principal.organization_id,
        email=body.email.lower(),
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role.value,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    return user
