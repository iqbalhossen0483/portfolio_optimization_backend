"""
Auth routes:
  POST /auth/register          — create a new user account
  POST /auth/login             — obtain a JWT access token
  GET  /auth/me                — get own profile
  PUT  /auth/me                — update own email / username / password
  GET  /auth/users             — (admin) list all users
  PUT  /auth/users/{user_id}/role  — (admin) change a user's role
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user, require_admin
from app.core.security import create_access_token, hash_password, verify_password
from app.models.domain import User
from app.models.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UpdateRoleRequest,
    UserListResponse,
    UserProfile,
    UserRole,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserProfile,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    # Check email uniqueness
    existing = await db.execute(select(User).where(User.email == body.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Check username uniqueness
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email=body.email.lower(),
        username=body.username,
        hashed_password=hash_password(body.password),
        role=UserRole.USER.value,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserProfile.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT access token",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    token = create_access_token({"sub": str(user.id), "email": user.email, "role": user.role})
    return TokenResponse(access_token=token, token_type="bearer", role=UserRole(user.role))


@router.get(
    "/me",
    response_model=UserProfile,
    summary="Get your own profile",
)
async def get_me(current_user: User = Depends(get_current_user)) -> UserProfile:
    return UserProfile.model_validate(current_user)


@router.put(
    "/me",
    response_model=UserProfile,
    summary="Update your own profile (email, username, or password)",
)
async def update_me(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    if body.email:
        email = body.email.lower()
        conflict = await db.execute(
            select(User).where(User.email == email, User.id != current_user.id)
        )
        if conflict.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email already in use")
        current_user.email = email

    if body.username:
        conflict = await db.execute(
            select(User).where(User.username == body.username, User.id != current_user.id)
        )
        if conflict.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Username already taken")
        current_user.username = body.username

    if body.password:
        current_user.hashed_password = hash_password(body.password)

    await db.commit()
    await db.refresh(current_user)
    return UserProfile.model_validate(current_user)


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="(Admin) List all users",
)
async def list_users(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UserListResponse:
    result = await db.execute(select(User).order_by(User.id))
    users = result.scalars().all()
    return UserListResponse(
        users=[UserProfile.model_validate(u) for u in users],
        total=len(users),
    )


@router.put(
    "/users/{user_id}/role",
    response_model=UserProfile,
    summary="(Admin) Change a user's role",
)
async def change_user_role(
    user_id: int,
    body: UpdateRoleRequest,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = body.role.value
    await db.commit()
    await db.refresh(user)
    return UserProfile.model_validate(user)
