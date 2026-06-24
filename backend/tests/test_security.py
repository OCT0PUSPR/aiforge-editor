"""Tests for security primitives: passwords, JWT, API keys, rate limiting."""

import pytest

from aiforge.security import (
    TokenBucketLimiter,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    is_api_key,
    verify_password,
)
from aiforge.security.tokens import TokenError


def test_password_hash_roundtrip():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong", h)


def test_password_long_input():
    # bcrypt's 72-byte limit must not break long passwords (sha256 pre-hash).
    long = "a" * 500
    assert verify_password(long, hash_password(long))


def test_jwt_roundtrip_and_type_enforcement():
    access = create_access_token("user1")
    assert decode_token(access, expected_type="access")["sub"] == "user1"
    refresh = create_refresh_token("user1")
    with pytest.raises(TokenError):
        decode_token(refresh, expected_type="access")


def test_jwt_tampered_rejected():
    token = create_access_token("user1")
    with pytest.raises(TokenError):
        decode_token(token + "tamper", expected_type="access")


def test_api_key_generation_and_hash():
    full, prefix, key_hash = generate_api_key()
    assert is_api_key(full)
    assert full.startswith(prefix)
    assert hash_api_key(full) == key_hash
    assert hash_api_key("aif_other") != key_hash


def test_token_bucket_limiter():
    limiter = TokenBucketLimiter()
    # 3 rpm: first 3 allowed, 4th denied (bucket starts full at capacity=rpm).
    for _ in range(3):
        assert limiter.allow("k", rpm=3)
    assert not limiter.allow("k", rpm=3)
    # Different key has its own bucket.
    assert limiter.allow("other", rpm=3)


def test_limiter_zero_rpm_unlimited():
    limiter = TokenBucketLimiter()
    for _ in range(100):
        assert limiter.allow("k", rpm=0)
