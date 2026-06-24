"""Workspace sandbox + path-traversal protection tests."""

import pytest

from aiforge.workspace.files import (
    NotFoundError,
    PathTraversalError,
    Workspace,
    WorkspaceError,
)


def test_read_write_roundtrip(workspace_root):
    ws = Workspace(workspace_root)
    ws.write("src/new.py", "print('hi')\n")
    assert ws.read("src/new.py") == "print('hi')\n"


def test_create_then_conflict(workspace_root):
    ws = Workspace(workspace_root)
    ws.create("notes.txt", "a")
    with pytest.raises(WorkspaceError):
        ws.create("notes.txt", "b")


def test_delete_file(workspace_root):
    ws = Workspace(workspace_root)
    ws.write("temp.txt", "x")
    assert ws.exists("temp.txt")
    ws.delete("temp.txt")
    assert not ws.exists("temp.txt")


def test_delete_missing_raises(workspace_root):
    ws = Workspace(workspace_root)
    with pytest.raises(NotFoundError):
        ws.delete("does-not-exist.txt")


@pytest.mark.parametrize(
    "evil",
    [
        "../secret.txt",
        "../../etc/passwd",
        "src/../../escape.py",
        "a/b/../../../c",
        "/../escape",  # leading slash stripped, then '..' escapes
    ],
)
def test_path_traversal_blocked(workspace_root, evil):
    ws = Workspace(workspace_root)
    with pytest.raises(PathTraversalError):
        ws.resolve(evil)


def test_leading_slash_is_workspace_relative(workspace_root):
    ws = Workspace(workspace_root)
    # A leading slash is treated as workspace-relative, NOT filesystem-absolute,
    # so "/etc/passwd" maps safely inside the workspace rather than escaping.
    assert ws.resolve("/src/auth.py") == (ws.root / "src" / "auth.py")
    assert ws.resolve("/etc/passwd") == (ws.root / "etc" / "passwd")


def test_symlink_escape_blocked(workspace_root, tmp_path):
    import os

    ws = Workspace(workspace_root)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = ws.root / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(PathTraversalError):
        ws.resolve("link.txt")


def test_tree_and_list(workspace_root):
    ws = Workspace(workspace_root)
    tree = ws.tree().to_dict()
    assert tree["type"] == "dir"
    files = ws.list_files()
    assert "src/calculator.py" in files
    assert "src/auth.py" in files


def test_tree_ignores_pycache(workspace_root):
    ws = Workspace(workspace_root)
    (ws.root / "__pycache__").mkdir()
    (ws.root / "__pycache__" / "x.pyc").write_text("junk")
    names = [c["name"] for c in ws.tree().to_dict()["children"]]
    assert "__pycache__" not in names
