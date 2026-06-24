"""Offline-capable codebase RAG: chunk, embed, store, and search.

Pipeline
--------
1. **Walk** the workspace, skipping ignored dirs and binary-ish files.
2. **Chunk** each file by code structure -- top-level ``def`` / ``class`` for
   Python, brace/heuristic blocks for other languages -- falling back to fixed
   line windows. Every chunk records its file path and 1-based line span.
3. **Embed** each chunk with a deterministic :class:`HashingEmbedder` (no
   network, no model download). An optional sentence-transformers embedder can
   be enabled via config when the dependency is present.
4. **Store** embeddings in a small in-memory numpy cosine store.
5. **Search** returns the top-k chunks for a query with file + line metadata.

The whole thing runs with only ``numpy`` -- and even degrades to a pure-Python
cosine fallback if numpy is missing -- so it works offline and in tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

try:  # numpy is the fast path; we fall back to pure python if absent.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None  # type: ignore[assignment]

from ..workspace.files import Workspace

# Extensions we treat as indexable source text.
_TEXT_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".kt",
    ".swift",
    ".scala",
    ".sh",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".css",
    ".sql",
}


@dataclasses.dataclass
class Chunk:
    """A unit of retrievable code with provenance."""

    path: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    symbol: Optional[str] = None  # function/class name if known

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "text": self.text,
        }


@dataclasses.dataclass
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d["score"] = round(self.score, 6)
        return d


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def _tokenize(text: str) -> List[str]:
    """Tokenize code/text into identifiers, numbers, and symbols, lowercased.

    CamelCase and snake_case identifiers are also split so that ``getUserName``
    contributes ``get``/``user``/``name`` features -- this makes lexical search
    match on sub-words, which matters a lot for code.
    """
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text):
        low = raw.lower()
        tokens.append(low)
        # Split snake_case and camelCase identifiers into sub-tokens.
        if "_" in raw:
            tokens.extend(p for p in raw.split("_") if p)
        for part in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", raw):
            pl = part.lower()
            if pl != low:
                tokens.append(pl)
    return tokens


@runtime_checkable
class Embedder(Protocol):
    """Protocol every embedder satisfies."""

    name: str

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]: ...


class HashingEmbedder:
    """Deterministic hashing embedder (feature hashing + L2 normalisation).

    Maps token frequencies into a fixed-dimensional vector using a stable hash.
    No training, no network -- identical input always yields an identical
    vector, which is what the determinism test asserts.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket_sign(self, token: str) -> tuple[int, float]:
        # MD5 here is a fast, stable hash for *feature hashing* (embeddings),
        # not for security. usedforsecurity=False documents that intent.
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            bucket, sign = self._bucket_sign(token)
            vec[bucket] += sign
        # L2 normalise so cosine similarity reduces to a dot product.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def _maybe_sentence_transformer(model_name: str):
    """Return a sentence-transformers embedder if the dependency is installed.

    Returns ``None`` (so callers fall back to the hashing embedder) when the
    package isn't available. When present it runs a real CPU model.
    """
    try:  # pragma: no cover - optional, only when ST is installed
        from sentence_transformers import SentenceTransformer
    except Exception:  # noqa: BLE001 - torch import can fail in many ways
        return None

    class _STEmbedder:  # pragma: no cover - optional path
        name = "sentence-transformers"

        def __init__(self) -> None:
            self._model = SentenceTransformer(model_name, device="cpu")
            self.dim = self._model.get_sentence_embedding_dimension()

        def embed(self, text: str) -> List[float]:
            return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]

        def embed_batch(self, texts):
            if not texts:
                return []
            return [
                [float(x) for x in row]
                for row in self._model.encode(
                    list(texts), normalize_embeddings=True, show_progress_bar=False
                )
            ]

    return _STEmbedder()


def get_embedder(name: str, *, dim: int = 512, st_model: str = "") -> Embedder:
    """Resolve an embedder by name.

    ``sentence-transformers`` is used when requested *and available*, otherwise
    we transparently fall back to the deterministic :class:`HashingEmbedder`.
    """
    if name in ("sentence-transformers", "st"):
        emb = _maybe_sentence_transformer(st_model or "sentence-transformers/all-MiniLM-L6-v2")
        if emb is not None:
            return emb
    return HashingEmbedder(dim=dim)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_code(path: str, text: str, *, window: int = 60, overlap: int = 10) -> List[Chunk]:
    """Split a file into structural chunks with line provenance.

    Python files are split on top-level ``def``/``class`` boundaries (each
    definition becomes a chunk). Everything else -- and any leftover lines --
    is split into overlapping fixed line windows.
    """
    lines = text.splitlines()
    if not lines:
        return []
    if path.endswith(".py"):
        chunks = _chunk_python(path, lines)
        if chunks:
            return chunks
    return _chunk_windows(path, lines, window=window, overlap=overlap)


def _chunk_python(path: str, lines: List[str]) -> List[Chunk]:
    # Find top-level def/class lines (no leading indentation).
    boundaries: List[tuple[int, Optional[str]]] = []
    def_re = re.compile(r"^(?:async\s+)?(def|class)\s+(\w+)")
    for i, line in enumerate(lines):
        if line and not line[0].isspace():
            m = def_re.match(line)
            if m:
                boundaries.append((i, m.group(2)))
    if not boundaries:
        return []
    chunks: List[Chunk] = []
    # Preamble (imports, module docstring) before the first definition.
    if boundaries[0][0] > 0:
        chunks.append(
            Chunk(
                path=path,
                start_line=1,
                end_line=boundaries[0][0],
                text="\n".join(lines[: boundaries[0][0]]),
                symbol="<module>",
            )
        )
    for idx, (start, name) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        chunks.append(
            Chunk(
                path=path,
                start_line=start + 1,
                end_line=end,
                text="\n".join(lines[start:end]),
                symbol=name,
            )
        )
    return chunks


def _chunk_windows(path: str, lines: List[str], *, window: int, overlap: int) -> List[Chunk]:
    if window <= 0:
        window = 60
    step = max(1, window - max(0, overlap))
    chunks: List[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        end = min(i + window, n)
        chunks.append(
            Chunk(
                path=path,
                start_line=i + 1,
                end_line=end,
                text="\n".join(lines[i:end]),
            )
        )
        if end >= n:
            break
        i += step
    return chunks


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
def _cosine_py(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class CosineStore:
    """A tiny in-memory cosine-similarity store over embedded chunks."""

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []
        self._matrix: object = None  # numpy matrix cache (Any)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunk: Chunk, vector: Sequence[float]) -> None:
        self.chunks.append(chunk)
        self._vectors.append(list(vector))
        self._matrix = None

    def remove_path(self, path: str) -> int:
        """Drop all chunks belonging to ``path``. Returns the number removed."""
        keep_chunks: List[Chunk] = []
        keep_vectors: List[List[float]] = []
        removed = 0
        for chunk, vec in zip(self.chunks, self._vectors):
            if chunk.path == path:
                removed += 1
            else:
                keep_chunks.append(chunk)
                keep_vectors.append(vec)
        if removed:
            self.chunks = keep_chunks
            self._vectors = keep_vectors
            self._matrix = None
        return removed

    def vectors(self) -> List[List[float]]:
        return [list(v) for v in self._vectors]

    def clear(self) -> None:
        self.chunks.clear()
        self._vectors.clear()
        self._matrix = None

    def search(self, query_vec: Sequence[float], k: int) -> List[SearchResult]:
        if not self._vectors:
            return []
        if _np is not None:
            return self._search_numpy(query_vec, k)
        return self._search_python(query_vec, k)

    def _search_numpy(self, query_vec: Sequence[float], k: int) -> List[SearchResult]:
        if self._matrix is None:
            self._matrix = _np.asarray(self._vectors, dtype="float32")
        mat: "Any" = self._matrix
        q = _np.asarray(query_vec, dtype="float32")
        # Vectors are L2-normalised by the embedder; renormalise defensively.
        mat_norms = _np.linalg.norm(mat, axis=1)
        q_norm = _np.linalg.norm(q)
        denom = mat_norms * (q_norm if q_norm else 1.0)
        denom[denom == 0] = 1.0
        scores = (mat @ q) / denom
        k = min(k, len(self.chunks))
        top = _np.argsort(-scores)[:k]
        return [SearchResult(self.chunks[int(i)], float(scores[int(i)])) for i in top]

    def _search_python(self, query_vec: Sequence[float], k: int) -> List[SearchResult]:
        scored = [
            SearchResult(chunk, _cosine_py(query_vec, vec))
            for chunk, vec in zip(self.chunks, self._vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: min(k, len(scored))]


# --------------------------------------------------------------------------
# Indexer
# --------------------------------------------------------------------------
def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RagIndexer:
    """Builds and queries a RAG index over a :class:`Workspace`.

    Supports **incremental** re-indexing: per-file content hashes are tracked so
    a re-index only re-embeds files whose content changed (and drops deleted
    files). The index can be **persisted** to and loaded from disk so it
    survives restarts.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        embed_dim: int = 512,
        chunk_lines: int = 60,
        chunk_overlap: int = 10,
        embedder: str = "hashing",
        use_sentence_transformers: bool = False,
        st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: Optional[str] = None,
    ) -> None:
        self.workspace = workspace
        self.chunk_lines = chunk_lines
        self.chunk_overlap = chunk_overlap
        if use_sentence_transformers and embedder == "hashing":
            embedder = "sentence-transformers"
        self.embedder = get_embedder(embedder, dim=embed_dim, st_model=st_model)
        self.store = CosineStore()
        self.file_count = 0
        # path -> content hash, for incremental dedup.
        self._hashes: Dict[str, str] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None

    @property
    def chunk_count(self) -> int:
        return len(self.store)

    # -- building -----------------------------------------------------------
    def build(self, paths: Optional[Iterable[str]] = None) -> dict:
        """Full (re)build of the index from scratch. Returns a stats dict."""
        self.store.clear()
        self._hashes.clear()
        self.file_count = 0
        file_list = list(paths) if paths is not None else self.workspace.list_files()
        for rel in file_list:
            self._index_file(rel)
        self._maybe_persist()
        return self.stats()

    def reindex(self, paths: Optional[Iterable[str]] = None) -> dict:
        """Incrementally re-index: only re-embed changed files; drop deleted.

        Returns a stats dict including ``{added, updated, removed, unchanged}``.
        """
        current = set(paths if paths is not None else self.workspace.list_files())
        current = {p for p in current if Path(p).suffix in _TEXT_EXTS}
        previous = set(self._hashes.keys())

        added = updated = removed = unchanged = 0

        # Remove chunks for deleted files.
        for rel in previous - current:
            self._drop_file(rel)
            removed += 1

        for rel in current:
            try:
                text = self.workspace.read(rel)
            except Exception:
                continue
            h = _content_hash(text)
            if self._hashes.get(rel) == h:
                unchanged += 1
                continue
            if rel in self._hashes:
                self._drop_file(rel)
                updated += 1
            else:
                added += 1
            self._index_text(rel, text, h)

        self.file_count = len({c.path for c in self.store.chunks})
        self._maybe_persist()
        stats = self.stats()
        stats.update(
            {"added": added, "updated": updated, "removed": removed, "unchanged": unchanged}
        )
        return stats

    def _index_file(self, rel: str) -> None:
        if Path(rel).suffix not in _TEXT_EXTS:
            return
        try:
            text = self.workspace.read(rel)
        except Exception:
            return
        self.file_count += 1
        self._index_text(rel, text, _content_hash(text))

    def _index_text(self, rel: str, text: str, content_hash: str) -> None:
        chunks = chunk_code(rel, text, window=self.chunk_lines, overlap=self.chunk_overlap)
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_batch(texts) if texts else []
        for chunk, vec in zip(chunks, vectors):
            self.store.add(chunk, vec)
        self._hashes[rel] = content_hash

    def _drop_file(self, rel: str) -> None:
        self.store.remove_path(rel)
        self._hashes.pop(rel, None)

    # -- searching ----------------------------------------------------------
    def search(self, query: str, k: int = 6) -> List[SearchResult]:
        if not query.strip() or len(self.store) == 0:
            return []
        q_vec = self.embedder.embed(query)
        return self.store.search(q_vec, k)

    # -- stats / persistence ------------------------------------------------
    def stats(self) -> dict:
        return {
            "files": self.file_count,
            "chunks": self.chunk_count,
            "embedder": self.embedder.name,
        }

    def _maybe_persist(self) -> None:
        if self._persist_dir is not None:
            self.save(self._persist_dir)

    def save(self, directory: Union[str, Path]) -> None:
        """Persist chunks, vectors, and hashes to ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": self.embedder.name,
            "file_count": self.file_count,
            "hashes": self._hashes,
            "chunks": [c.to_dict() for c in self.store.chunks],
            "vectors": self.store.vectors(),
        }
        (directory / "index.json").write_text(json.dumps(payload), encoding="utf-8")

    def load(self, directory: Union[str, Path]) -> bool:
        """Load a persisted index. Returns False if none exists / incompatible."""
        path = Path(directory) / "index.json"
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if payload.get("embedder") != self.embedder.name:
            return False  # embedder changed; vectors are incomparable
        self.store.clear()
        self._hashes = dict(payload.get("hashes", {}))
        self.file_count = int(payload.get("file_count", 0))
        for cd, vec in zip(payload.get("chunks", []), payload.get("vectors", [])):
            chunk = Chunk(
                path=cd["path"],
                start_line=cd["start_line"],
                end_line=cd["end_line"],
                text=cd["text"],
                symbol=cd.get("symbol"),
            )
            self.store.add(chunk, vec)
        return True
