"""JWT access/refresh token issuance and verification + API-key hashing."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from ..config import get_settings


class TokenError(Exception):
    """Raised when a token is invalid or expired."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: str, *, extra: Optional[dict] = None) -> str:
    return _encode(user_id, "access", get_settings().access_token_ttl_seconds, extra)


def create_refresh_token(user_id: str) -> str:
    return _encode(user_id, "refresh", get_settings().refresh_token_ttl_seconds, None)


def _encode(user_id: str, token_type: str, ttl: int, extra: Optional[dict]) -> str:
    settings = get_settings()
    now = _now()
    payload = {
        "sub": user_id,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl)).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.effective_jwt_secret(), algorithm=settings.jwt_algorithm)


def decode_token(token: str, *, expected_type: Optional[str] = None) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.effective_jwt_secret(), algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token expired") from exc
    except jwt.PyJWTError as exc:
        raise TokenError("invalid token") from exc
    if expected_type and payload.get("type") != expected_type:
        raise TokenError(f"expected {expected_type} token")
    return payload


# -- API keys ---------------------------------------------------------------
_API_KEY_PREFIX = "aif_"


def generate_api_key() -> tuple[str, str, str]:
    """Return (full_key, prefix, sha256_hex).

    The full key is shown to the user exactly once; only the hash is stored.
    """
    body = secrets.token_urlsafe(32)
    full = f"{_API_KEY_PREFIX}{body}"
    prefix = full[: len(_API_KEY_PREFIX) + 6]
    return full, prefix, hash_api_key(full)


def hash_api_key(full_key: str) -> str:
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def is_api_key(value: str) -> bool:
    return value.startswith(_API_KEY_PREFIX)
