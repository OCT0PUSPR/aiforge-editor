"""Corpus building + FIM-aware dataloader.

The corpus is assembled from permissively-licensed Python sources:
- selected files from this repository (``aiforge`` package),
- selected modules from the Python standard library (PSF-licensed),
- any extra ``*.py`` under a user-provided directory.

Each document is tokenized, FIM-transformed (per :mod:`aiforge.ml.fim`), and the
results are concatenated and packed into fixed-size training blocks. Two
non-overlapping splits (train/val) are produced.

The dataloader yields ``(x, y)`` batches where ``y`` is ``x`` shifted by one.
``torch`` is imported here, so this module is only imported when training.
"""

from __future__ import annotations

import random
import sysconfig
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch

from .config import TrainConfig
from .fim import FimTokens, apply_fim
from .tokenizer import CodeTokenizer

# A curated set of stdlib modules that are readable, self-contained, and
# representative of idiomatic Python. All PSF-licensed.
_STDLIB_MODULES = [
    "dataclasses",
    "json/__init__",
    "json/decoder",
    "json/encoder",
    "argparse",
    "collections/__init__",
    "functools",
    "itertools",
    "typing",
    "enum",
    "abc",
    "contextlib",
    "heapq",
    "bisect",
    "queue",
    "string",
    "textwrap",
    "difflib",
    "pprint",
    "copy",
    "random",
    "statistics",
    "fractions",
    "decimal",
    "csv",
    "configparser",
    "pathlib",
    "tempfile",
    "shutil",
    "io",
    "base64",
    "hashlib",
    "secrets",
    "uuid",
    "datetime",
    "calendar",
    "operator",
    "numbers",
    "html/parser",
    "urllib/parse",
    "http/client",
    "logging/__init__",
]


def _read_text(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.strip():
        return None
    return text


# Directory names within the stdlib we skip (tests, vendored, platform cruft).
_STDLIB_SKIP_DIRS = {
    "test",
    "tests",
    "idlelib",
    "turtledemo",
    "tkinter",
    "lib2to3",
    "ensurepip",
    "venv",
    "site-packages",
    "__pycache__",
    "encodings",
}


def collect_corpus(
    extra_dir: Optional[str] = None,
    *,
    repo_root: Optional[str] = None,
    broad: bool = True,
    max_chars_per_doc: int = 40_000,
) -> List[str]:
    """Gather code documents from stdlib + this repo (+ optional extra dir).

    With ``broad=True`` (default) we walk a large, deterministic slice of the
    standard library -- hundreds of PSF-licensed modules -- giving the model
    enough real data to generalise. ``broad=False`` uses only the small curated
    module list (faster, for smoke tests).
    """
    docs: List[str] = []

    # 1. Python standard library (the running interpreter's stdlib).
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    if broad:
        for py in sorted(stdlib.rglob("*.py")):
            parts = set(py.parts)
            if parts & _STDLIB_SKIP_DIRS:
                continue
            if py.name.startswith("test_") or py.name == "conftest.py":
                continue
            text = _read_text(py)
            if text and 200 < len(text) <= max_chars_per_doc:
                docs.append(text)
    else:
        for mod in _STDLIB_MODULES:
            text = _read_text(stdlib / f"{mod}.py")
            if text:
                docs.append(text)

    # 2. This repository's own backend code.
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    aiforge_pkg = root / "aiforge"
    if aiforge_pkg.is_dir():
        for py in sorted(aiforge_pkg.rglob("*.py")):
            if "ml" in py.parts and py.name in ("model.py",):
                continue  # skip the heaviest torch file to keep corpus tidy
            text = _read_text(py)
            if text:
                docs.append(text)

    # 3. Optional extra directory of permissive code.
    if extra_dir:
        for py in sorted(Path(extra_dir).rglob("*.py")):
            text = _read_text(py)
            if text:
                docs.append(text)

    return docs


def build_token_stream(
    docs: Sequence[str],
    tokenizer: CodeTokenizer,
    fim: FimTokens,
    rng: random.Random,
    *,
    fim_rate: float,
    spm_rate: float,
) -> List[int]:
    """Tokenize + FIM-transform each doc, concatenated into one id stream."""
    stream: List[int] = []
    for doc in docs:
        ids = tokenizer.encode(doc)
        if not ids:
            continue
        stream.extend(apply_fim(ids, fim, rng, fim_rate=fim_rate, spm_rate=spm_rate))
    return stream


class FimDataset:
    """Packs a FIM token stream into fixed-size blocks and serves batches."""

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def batch(
        self, batch_size: int, device: "torch.device", generator: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(self)
        ix = torch.randint(0, max(1, n), (batch_size,), generator=generator)
        # Vectorized gather: build (batch, block) index windows in one shot.
        offsets = torch.arange(self.block_size)
        idx = ix[:, None] + offsets[None, :]
        x = self.data[idx]
        y = self.data[idx + 1]
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def make_datasets(
    docs: Sequence[str],
    tokenizer: CodeTokenizer,
    cfg: TrainConfig,
    block_size: int,
    *,
    val_fraction: float = 0.1,
) -> Tuple[FimDataset, FimDataset]:
    """Build train/val datasets from documents using the FIM transform."""
    rng = random.Random(cfg.seed)
    stream = build_token_stream(
        docs, tokenizer, tokenizer.fim, rng, fim_rate=cfg.fim_rate, spm_rate=cfg.spm_rate
    )
    data = torch.tensor(stream, dtype=torch.long)
    n_val = int(len(data) * val_fraction)
    train_data = data[:-n_val] if n_val > 0 else data
    val_data = data[-n_val:] if n_val > 0 else data[: max(block_size + 2, 1)]
    return (
        FimDataset(train_data, block_size),
        FimDataset(val_data, block_size),
    )


def iter_batches(
    dataset: FimDataset,
    batch_size: int,
    device: "torch.device",
    generator: torch.Generator,
    steps: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(steps):
        yield dataset.batch(batch_size, device, generator)
