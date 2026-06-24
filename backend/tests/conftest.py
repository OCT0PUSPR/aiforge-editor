"""Shared pytest fixtures.

Each test gets an isolated temp data dir + sqlite DB and a fresh app, so tests
are independent and never touch real data.
"""

import shutil
from pathlib import Path

import pytest

SAMPLE = Path(__file__).parent / "sample_project"


@pytest.fixture()
def workspace_root(tmp_path):
    """A fresh FS workspace seeded with a copy of the sample project."""
    root = tmp_path / "ws"
    shutil.copytree(SAMPLE, root)
    return root


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    """Settings pointing at an isolated temp data dir + sqlite db."""
    monkeypatch.setenv("AIFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AIFORGE_DATABASE_URL", f"sqlite:///{tmp_path / 'data' / 'test.db'}")
    monkeypatch.setenv("AIFORGE_BACKEND", "mock")
    monkeypatch.setenv(
        "AIFORGE_JWT_SECRET", "test-secret-do-not-use-in-prod-at-least-32-bytes-long!!"
    )
    monkeypatch.setenv("AIFORGE_RAG_PERSIST", "false")
    from aiforge.config import reset_settings_cache
    from aiforge.db.session import reset_engine

    reset_settings_cache()
    reset_engine()
    from aiforge.config import get_settings

    s = get_settings()
    yield s
    reset_settings_cache()
    reset_engine()


@pytest.fixture()
def app(settings):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from aiforge.api.server import create_app

    return create_app(settings)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_client(client):
    """A TestClient with a registered user + default workspace, auth header set."""
    r = client.post(
        "/api/auth/register",
        json={"email": "alice@example.com", "username": "alice", "password": "password123"},
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    # Find the default workspace.
    ws = client.get("/api/workspaces").json()
    client.workspace_id = ws[0]["id"]  # type: ignore[attr-defined]
    return client
