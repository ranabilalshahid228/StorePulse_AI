from __future__ import annotations

import uuid

from pydantic import BaseModel, EmailStr, Field

from app.models.enums import UserRole
from app.schemas.common import ORMModel


class RegisterRequest(BaseModel):
    organization_name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str | None = Field(default=None, max_length=200)
    default_currency: str = Field(default="EUR", min_length=3, max_length=3)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str | None = None
    role: UserRole = UserRole.ANALYST


class UserRead(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    email: EmailStr
    full_name: str | None
    role: UserRole
    is_active: bool


class OrganizationRead(ORMModel):
    id: uuid.UUID
    name: str
    slug: str
    plan: str
    default_currency: str


class MeResponse(BaseModel):
    user: UserRead
    organization: OrganizationRead
