"""Workspace isolation tests: one user cannot touch another's workspace."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _register(client, name):
    r = client.post(
        "/api/auth/register",
        json={"email": f"{name}@x.com", "username": name, "password": "password123"},
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    ws = client.get("/api/workspaces", headers={"Authorization": f"Bearer {token}"}).json()
    return token, ws[0]["id"]


def test_users_cannot_access_each_others_workspaces(client):
    alice_token, alice_ws = _register(client, "alice")
    bob_token, bob_ws = _register(client, "bob")
    assert alice_ws != bob_ws

    # Alice writes a secret in her workspace.
    h_a = {"Authorization": f"Bearer {alice_token}"}
    r = client.post(
        f"/api/workspaces/{alice_ws}/files",
        json={"path": "secret.txt", "content": "alice-only"},
        headers=h_a,
    )
    assert r.status_code == 200

    # Bob tries to read Alice's workspace -> 404 (not even visible).
    h_b = {"Authorization": f"Bearer {bob_token}"}
    r = client.get(f"/api/workspaces/{alice_ws}/files", params={"path": "secret.txt"}, headers=h_b)
    assert r.status_code == 404

    # Bob's workspace does not see Alice's file.
    r = client.get(f"/api/workspaces/{bob_ws}/files", params={"path": "secret.txt"}, headers=h_b)
    assert r.status_code == 404


def test_multiple_workspaces_are_filesystem_isolated(client):
    token, ws1 = _register(client, "carol")
    h = {"Authorization": f"Bearer {token}"}
    # Create a second workspace.
    r = client.post("/api/workspaces", json={"name": "Project Two"}, headers=h)
    assert r.status_code == 201
    ws2 = r.json()["id"]

    # Same relative path, different content per workspace.
    client.post(f"/api/workspaces/{ws1}/files", json={"path": "f.txt", "content": "one"}, headers=h)
    client.post(f"/api/workspaces/{ws2}/files", json={"path": "f.txt", "content": "two"}, headers=h)

    c1 = client.get(f"/api/workspaces/{ws1}/files", params={"path": "f.txt"}, headers=h).json()
    c2 = client.get(f"/api/workspaces/{ws2}/files", params={"path": "f.txt"}, headers=h).json()
    assert c1["content"] == "one"
    assert c2["content"] == "two"


def test_cannot_delete_last_workspace(client):
    token, ws1 = _register(client, "dave")
    h = {"Authorization": f"Bearer {token}"}
    r = client.delete(f"/api/workspaces/{ws1}", headers=h)
    assert r.status_code == 400
