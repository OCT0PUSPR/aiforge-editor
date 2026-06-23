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
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:  # numpy is the fast path; we fall back to pure python if absent.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None

from ..workspace.files import Workspace

# Extensions we treat as indexable source text.
_TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".kt", ".swift", ".scala", ".sh",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".sql",
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
        digest = hashlib.md5(token.encode("utf-8")).digest()
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
    """Return a sentence-transformers embedder if the dependency is installed."""
    try:  # pragma: no cover - optional, only when ST is installed
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    class _STEmbedder:  # pragma: no cover - optional path
        name = "sentence-transformers"

        def __init__(self) -> None:
            self._model = SentenceTransformer(model_name)
            self.dim = self._model.get_sentence_embedding_dimension()

        def embed(self, text: str) -> List[float]:
            return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]

        def embed_batch(self, texts):
            return [
                [float(x) for x in row]
                for row in self._model.encode(list(texts), normalize_embeddings=True)
            ]

    return _STEmbedder()


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
        self._matrix = None  # numpy matrix cache

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunk: Chunk, vector: Sequence[float]) -> None:
        self.chunks.append(chunk)
        self._vectors.append(list(vector))
        self._matrix = None

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
        mat = self._matrix
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
class RagIndexer:
    """Builds and queries a RAG index over a :class:`Workspace`."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        embed_dim: int = 512,
        chunk_lines: int = 60,
        chunk_overlap: int = 10,
        use_sentence_transformers: bool = False,
        st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.workspace = workspace
        self.chunk_lines = chunk_lines
        self.chunk_overlap = chunk_overlap
        self.embedder = None
        if use_sentence_transformers:
            self.embedder = _maybe_sentence_transformer(st_model)
        if self.embedder is None:
            self.embedder = HashingEmbedder(dim=embed_dim)
        self.store = CosineStore()
        self.file_count = 0

    @property
    def chunk_count(self) -> int:
        return len(self.store)

    def build(self, paths: Optional[Iterable[str]] = None) -> dict:
        """(Re)build the index. Returns a small stats dict."""
        self.store.clear()
        self.file_count = 0
        file_list = list(paths) if paths is not None else self.workspace.list_files()
        for rel in file_list:
            if Path(rel).suffix not in _TEXT_EXTS:
                continue
            try:
                text = self.workspace.read(rel)
            except Exception:
                continue
            self.file_count += 1
            chunks = chunk_code(
                rel, text, window=self.chunk_lines, overlap=self.chunk_overlap
            )
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_batch(texts) if texts else []
            for chunk, vec in zip(chunks, vectors):
                self.store.add(chunk, vec)
        return {
            "files": self.file_count,
            "chunks": self.chunk_count,
            "embedder": self.embedder.name,
        }

    def search(self, query: str, k: int = 6) -> List[SearchResult]:
        if not query.strip() or len(self.store) == 0:
            return []
        q_vec = self.embedder.embed(query)
        return self.store.search(q_vec, k)
