"""
JWT creation/verification and bcrypt password hashing.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings

cfg = get_settings()

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(data: dict[str, Any]) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=cfg.jwt_expire_minutes)
    payload["exp"] = expire
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """Raises JWTError on invalid or expired token."""
    return jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
