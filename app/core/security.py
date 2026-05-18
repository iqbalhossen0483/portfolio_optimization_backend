"""
JWT creation/verification and bcrypt password hashing.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any

import base64
import hashlib

import bcrypt as _bcrypt

from jose import jwt

from app.config import get_settings

cfg = get_settings()


def _normalize_password(password: str) -> bytes:
    """SHA256 → base64 gives a fixed 44-byte value, safely within bcrypt's 72-byte limit."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return _bcrypt.hashpw(_normalize_password(password), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    hashed_b = hashed.encode("utf-8") if isinstance(hashed, str) else hashed
    return _bcrypt.checkpw(_normalize_password(plain), hashed_b)


def create_access_token(data: dict[str, Any]) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=cfg.jwt_expire_minutes)
    payload["exp"] = expire
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """Raises JWTError on invalid or expired token."""
    return jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
