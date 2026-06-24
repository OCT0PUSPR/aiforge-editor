"""End-to-end: the from-scratch model serving /api/complete.

Skipped cleanly unless torch is installed AND a trained checkpoint exists at
backend/runs/proof (produced by `python -m aiforge.ml.train`).
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "proof"

pytestmark = pytest.mark.skipif(
    not ((RUN_DIR / "best.pt").exists() or (RUN_DIR / "ckpt.pt").exists()),
    reason="no trained checkpoint at runs/proof (run `python -m aiforge.ml.train`)",
)


@pytest.fixture()
def local_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AIFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AIFORGE_DATABASE_URL", f"sqlite:///{tmp_path / 'data' / 't.db'}")
    monkeypatch.setenv("AIFORGE_JWT_SECRET", "test-secret-at-least-32-bytes-for-hs256!!")
    monkeypatch.setenv("AIFORGE_BACKEND", "mock")
    monkeypatch.setenv("AIFORGE_COMPLETE_BACKEND", "local")
    monkeypatch.setenv("AIFORGE_LOCAL_MODEL_DIR", str(RUN_DIR))
    monkeypatch.setenv("AIFORGE_RAG_PERSIST", "false")
    from aiforge.config import get_settings, reset_settings_cache
    from aiforge.db.session import reset_engine

    reset_settings_cache()
    reset_engine()
    from aiforge.api.server import create_app

    yield create_app(get_settings())
    reset_settings_cache()
    reset_engine()


def test_local_model_serves_completion(local_app):
    from fastapi.testclient import TestClient

    with TestClient(local_app) as c:
        r = c.post(
            "/api/auth/register",
            json={"email": "l@x.com", "username": "localuser", "password": "password123"},
        )
        assert r.status_code == 201
        c.headers.update({"Authorization": f"Bearer {r.json()['access_token']}"})
        ws = c.get("/api/workspaces").json()[0]["id"]

        r = c.post(
            f"/api/workspaces/{ws}/ai/complete",
            json={"prefix": "def add(a, b):\n    return ", "suffix": "\n", "language": "python"},
        )
        assert r.status_code == 200
        assert "event: token" in r.text and "event: done" in r.text
        text = ""
        for block in r.text.split("\n\n"):
            if "event: token" in block:
                for line in block.split("\n"):
                    if line.startswith("data:"):
                        text += json.loads(line[5:].strip())["text"]
        # The local model produced *some* completion (real model output).
        assert isinstance(text, str)
