"""API integration tests via TestClient (offline MockLLM, isolated sqlite)."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _sse_text(raw: str) -> str:
    out = []
    for block in raw.split("\n\n"):
        if "event: token" in block:
            for line in block.splitlines():
                if line.startswith("data:"):
                    out.append(json.loads(line[len("data:") :].strip())["text"])
    return "".join(out)


def _ws(client):
    return client.workspace_id


def _seed(client, files):
    ws = _ws(client)
    for path, content in files.items():
        client.post(f"/api/workspaces/{ws}/files", json={"path": path, "content": content})


def test_health_and_ready_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is True
    m = client.get("/metrics")
    assert m.status_code == 200
    assert "aiforge_http_requests_total" in m.text


def test_unauthenticated_rejected(client):
    # Random workspace id, no auth header.
    r = client.get("/api/workspaces/abc/files/tree")
    assert r.status_code == 401


def test_file_crud_and_traversal(auth_client):
    ws = _ws(auth_client)
    base = f"/api/workspaces/{ws}/files"
    # create
    r = auth_client.post(base, json={"path": "src/x.py", "content": "print(1)\n"})
    assert r.status_code == 200, r.text
    # read
    r = auth_client.get(base, params={"path": "src/x.py"})
    assert r.json()["content"] == "print(1)\n"
    # tree
    assert auth_client.get(f"{base}/tree").json()["type"] == "dir"
    # rename
    r = auth_client.post(f"{base}/rename", json={"src": "src/x.py", "dst": "src/y.py"})
    assert r.status_code == 200
    assert auth_client.get(base, params={"path": "src/y.py"}).status_code == 200
    # delete
    assert auth_client.delete(base, params={"path": "src/y.py"}).status_code == 200
    # traversal blocked
    assert auth_client.get(base, params={"path": "../../../etc/passwd"}).status_code == 400


def test_file_quota_rejects_large_file(auth_client, settings, monkeypatch):
    ws = _ws(auth_client)
    base = f"/api/workspaces/{ws}/files"
    big = "x" * (settings.max_file_bytes + 10)
    r = auth_client.post(base, json={"path": "big.txt", "content": big})
    assert r.status_code == 413


def test_rag_index_and_search(auth_client):
    _seed(
        auth_client,
        {
            "src/auth.py": "import hashlib\n\ndef hash_password(pw, salt):\n    return hashlib.sha256((salt+pw).encode()).hexdigest()\n",
            "src/calc.py": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        },
    )
    ws = _ws(auth_client)
    r = auth_client.post(f"/api/workspaces/{ws}/rag/index")
    assert r.status_code == 200
    assert r.json()["files"] >= 2
    res = auth_client.get(
        f"/api/workspaces/{ws}/rag/search", params={"q": "hash a password with salt", "k": 3}
    ).json()["results"]
    assert res and res[0]["path"] == "src/auth.py"


def test_rag_reindex_incremental(auth_client):
    _seed(auth_client, {"a.py": "def a():\n    return 1\n"})
    ws = _ws(auth_client)
    auth_client.post(f"/api/workspaces/{ws}/rag/index")
    r = auth_client.post(f"/api/workspaces/{ws}/rag/reindex").json()
    assert r["unchanged"] >= 1


def test_complete_sse(auth_client):
    ws = _ws(auth_client)
    r = auth_client.post(
        f"/api/workspaces/{ws}/ai/complete",
        json={"prefix": "def greet(name):\n", "suffix": "", "language": "python"},
    )
    assert r.status_code == 200
    assert "event: token" in r.text and "event: done" in r.text
    assert _sse_text(r.text)


def test_chat_sse_with_refs_and_persistence(auth_client):
    _seed(auth_client, {"src/auth.py": "def hash_password(pw, salt):\n    return pw + salt\n"})
    ws = _ws(auth_client)
    auth_client.post(f"/api/workspaces/{ws}/rag/index")
    r = auth_client.post(
        f"/api/workspaces/{ws}/ai/chat",
        json={"question": "How does password hashing work?"},
    )
    assert r.status_code == 200
    assert "event: token" in r.text
    assert "event: meta" in r.text
    assert "src/auth.py" in r.text


def test_edit_propose_apply_persist_and_undo(auth_client):
    _seed(auth_client, {"src/m.py": "def m():\n    return 1\n"})
    ws = _ws(auth_client)
    base = f"/api/workspaces/{ws}/ai"
    proposal = auth_client.post(
        f"{base}/edit", json={"path": "src/m.py", "instruction": "add header"}
    ).json()
    assert proposal["changed"] is True
    assert proposal["usage"]["input_tokens"] > 0

    applied = auth_client.post(
        f"{base}/edit/apply",
        json={"path": "src/m.py", "diff": proposal["diff"], "instruction": "add header"},
    ).json()
    assert applied["new_content"].startswith("# Edited by aiforge MockLLM")

    # File persisted.
    content = auth_client.get(f"/api/workspaces/{ws}/files", params={"path": "src/m.py"}).json()[
        "content"
    ]
    assert content.startswith("# Edited by aiforge MockLLM")

    # History + undo.
    hist = auth_client.get(f"{base}/edit/history").json()["history"]
    assert hist
    edit_id = hist[0]["id"]
    undo = auth_client.post(f"{base}/edit/undo/{edit_id}")
    assert undo.status_code == 200
    restored = auth_client.get(f"/api/workspaces/{ws}/files", params={"path": "src/m.py"}).json()[
        "content"
    ]
    assert restored == "def m():\n    return 1\n"


def test_multifile_edit(auth_client):
    _seed(
        auth_client,
        {"src/a.py": "def a():\n    return 1\n", "src/b.py": "def b():\n    return 2\n"},
    )
    ws = _ws(auth_client)
    base = f"/api/workspaces/{ws}/ai"
    proposal = auth_client.post(
        f"{base}/edit/multi",
        json={"paths": ["src/a.py", "src/b.py"], "instruction": "add header to both"},
    ).json()
    assert proposal["changed"] is True
    applied = auth_client.post(
        f"{base}/edit/apply",
        json={"diff": proposal["diff"], "multifile": True, "instruction": "headers"},
    ).json()
    assert set(applied["applied"]) == {"src/a.py", "src/b.py"}
    for p in ("src/a.py", "src/b.py"):
        c = auth_client.get(f"/api/workspaces/{ws}/files", params={"path": p}).json()["content"]
        assert c.startswith("# Edited by aiforge MockLLM")
