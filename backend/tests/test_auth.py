"""Auth flow tests: register, login, refresh, me, api keys."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_register_login_refresh_me(client):
    r = client.post(
        "/api/auth/register",
        json={"email": "bob@example.com", "username": "bob", "password": "supersecret"},
    )
    assert r.status_code == 201, r.text
    tokens = r.json()
    assert tokens["access_token"] and tokens["refresh_token"]

    # login
    r = client.post("/api/auth/login", json={"username": "bob", "password": "supersecret"})
    assert r.status_code == 200
    access = r.json()["access_token"]
    refresh = r.json()["refresh_token"]

    # me
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert r.status_code == 200
    assert r.json()["username"] == "bob"

    # refresh
    r = client.post("/api/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_login_wrong_password(client):
    client.post(
        "/api/auth/register",
        json={"email": "c@example.com", "username": "carol", "password": "rightpass1"},
    )
    r = client.post("/api/auth/login", json={"username": "carol", "password": "wrongpass"})
    assert r.status_code == 401


def test_duplicate_registration_rejected(client):
    body = {"email": "d@example.com", "username": "dave", "password": "password99"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409


def test_short_password_rejected(client):
    r = client.post(
        "/api/auth/register",
        json={"email": "e@example.com", "username": "eve", "password": "short"},
    )
    assert r.status_code == 422  # pydantic validation


def test_api_key_auth(auth_client):
    # Create an API key and use it as a bearer token.
    r = auth_client.post("/api/keys", json={"name": "ci"})
    assert r.status_code == 201, r.text
    full = r.json()["key"]
    assert full.startswith("aif_")

    # Use the key (drop the JWT header) on an authed endpoint.
    fresh_headers = {"Authorization": f"Bearer {full}"}
    r = auth_client.get("/api/auth/me", headers=fresh_headers)
    assert r.status_code == 200

    # Revoke it; now it fails.
    key_id = r2 = auth_client.get("/api/keys").json()[0]["id"]
    assert auth_client.delete(f"/api/keys/{key_id}").status_code == 204
    r = auth_client.get("/api/auth/me", headers=fresh_headers)
    assert r.status_code == 401
