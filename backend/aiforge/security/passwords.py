"""Password hashing using bcrypt directly.

We use the ``bcrypt`` library rather than passlib: modern bcrypt (4.x) is
incompatible with passlib 1.7.x. bcrypt has a hard 72-byte input limit, so we
pre-hash the password with SHA-256 and base64-encode it before bcrypt. This
removes the length ceiling without weakening the hash, and is a well-known
pattern (the same approach Django uses for its bcrypt_sha256 hasher).
"""

from __future__ import annotations

import base64
import hashlib

import bcrypt

_ROUNDS = 12


def _prehash(password: str) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    # base64 of 32 bytes is 44 bytes -> well under bcrypt's 72-byte limit.
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=_ROUNDS)
    return bcrypt.hashpw(_prehash(password), salt).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False
