"""Shared pytest fixtures."""
import shutil
from pathlib import Path

import pytest

SAMPLE = Path(__file__).parent / "sample_project"


@pytest.fixture()
def workspace_root(tmp_path):
    """A fresh workspace seeded with a copy of the sample project."""
    root = tmp_path / "ws"
    shutil.copytree(SAMPLE, root)
    return root
