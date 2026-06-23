"""FastAPI endpoint tests via TestClient, using the offline MockLLM backend.

Guarded so the suite still passes (skips) if FastAPI / httpx are not installed.
"""
import json

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by starlette's TestClient

from fastapi.testclient import TestClient  # noqa: E402

from aiforge.api.server import create_app  # noqa: E402
from aiforge.config import Settings  # noqa: E402


@pytest.fixture()
def client(workspace_root):
    settings = Settings()
    settings.backend = "mock"
    settings.workspace_root = str(workspace_root)
    app = create_app(settings)
    return TestClient(app)


def _sse_text(raw: str) -> str:
    """Concatenate the 'token' event payloads from an SSE response body."""
    out = []
    for block in raw.split("\n\n"):
        if "event: token" in block:
            for line in block.splitlines():
                if line.startswith("data:"):
                    out.append(json.loads(line[len("data:"):].strip())["text"])
    return "".join(out)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "mock"


def test_tree(client):
    r = client.get("/api/tree")
    assert r.status_code == 200
    assert r.json()["type"] == "dir"


def test_read_and_save_file(client):
    r = client.get("/api/file", params={"path": "src/calculator.py"})
    assert r.status_code == 200
    assert "fibonacci" in r.json()["content"]

    r = client.put("/api/file", json={"path": "src/calculator.py", "content": "# changed\n"})
    assert r.status_code == 200
    r = client.get("/api/file", params={"path": "src/calculator.py"})
    assert r.json()["content"] == "# changed\n"


def test_file_traversal_rejected(client):
    r = client.get("/api/file", params={"path": "../../../etc/passwd"})
    assert r.status_code == 400


def test_create_and_delete_file(client):
    r = client.post("/api/file", json={"path": "scratch.txt", "content": "x"})
    assert r.status_code == 200
    r = client.delete("/api/file", params={"path": "scratch.txt"})
    assert r.status_code == 200


def test_index_and_search(client):
    r = client.post("/api/index")
    assert r.status_code == 200
    assert r.json()["files"] >= 2

    r = client.get("/api/search", params={"q": "fibonacci", "k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results
    assert any(res["path"] == "src/calculator.py" for res in results)


def test_complete_sse(client):
    r = client.post(
        "/api/complete",
        json={"prefix": "def greet(name):\n", "suffix": "", "language": "python"},
    )
    assert r.status_code == 200
    body = r.text
    assert "event: token" in body
    assert "event: done" in body
    assert _sse_text(body)  # produced some completion text


def test_chat_sse_with_references(client):
    client.post("/api/index")
    r = client.post(
        "/api/chat",
        json={"question": "How does password hashing work?", "history": []},
    )
    assert r.status_code == 200
    body = r.text
    assert "event: token" in body
    assert "event: meta" in body  # references payload
    assert "src/auth.py" in body


def test_edit_propose_and_apply(client):
    r = client.post(
        "/api/edit",
        json={"path": "src/calculator.py", "instruction": "add a header comment"},
    )
    assert r.status_code == 200
    proposal = r.json()
    assert proposal["changed"] is True

    r = client.post(
        "/api/edit/apply",
        json={"path": "src/calculator.py", "diff": proposal["diff"]},
    )
    assert r.status_code == 200
    new_content = r.json()["new_content"]
    assert new_content.startswith("# Edited by aiforge MockLLM")

    # Confirm it actually persisted.
    r = client.get("/api/file", params={"path": "src/calculator.py"})
    assert r.json()["content"].startswith("# Edited by aiforge MockLLM")


def test_edit_apply_full_content(client):
    r = client.post(
        "/api/edit/apply",
        json={"path": "src/auth.py", "new_content": "print('rewritten')\n"},
    )
    assert r.status_code == 200
    r = client.get("/api/file", params={"path": "src/auth.py"})
    assert r.json()["content"] == "print('rewritten')\n"
