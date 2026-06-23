"""RAG indexer, hashing embedder, and search tests."""
from aiforge.rag.indexer import HashingEmbedder, RagIndexer, chunk_code
from aiforge.workspace.files import Workspace


def test_hashing_embedder_deterministic():
    emb = HashingEmbedder(dim=256)
    a = emb.embed("def add(a, b): return a + b")
    b = emb.embed("def add(a, b): return a + b")
    assert a == b
    assert len(a) == 256


def test_hashing_embedder_normalised():
    emb = HashingEmbedder(dim=128)
    vec = emb.embed("hello world fibonacci sequence")
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_hashing_embedder_distinguishes_text():
    emb = HashingEmbedder(dim=512)
    auth = emb.embed("hash password salt sha256 verify session token")
    calc = emb.embed("fibonacci add subtract calculator total number")
    # Different content should not be identical vectors.
    assert auth != calc


def test_chunk_python_splits_on_definitions():
    src = (
        "import os\n\n"
        "def first():\n    return 1\n\n"
        "def second():\n    return 2\n\n"
        "class Thing:\n    def method(self):\n        return 3\n"
    )
    chunks = chunk_code("m.py", src)
    symbols = [c.symbol for c in chunks]
    assert "first" in symbols
    assert "second" in symbols
    assert "Thing" in symbols
    # Line spans are 1-based and ordered.
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_window_fallback_for_non_python():
    text = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_code("data.txt", text, window=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 50


def test_index_and_search_finds_right_file(workspace_root):
    ws = Workspace(workspace_root)
    idx = RagIndexer(ws, embed_dim=512)
    stats = idx.build()
    assert stats["files"] >= 2
    assert stats["chunks"] >= 4

    # A query about password hashing should rank the auth module on top.
    results = idx.search("hash a password with a salt", k=3)
    assert results
    top = results[0].chunk
    assert top.path == "src/auth.py"

    # A query about Fibonacci should rank the calculator module on top.
    results = idx.search("compute the nth fibonacci number", k=3)
    assert results
    assert results[0].chunk.path == "src/calculator.py"


def test_search_empty_index_returns_empty(tmp_path):
    ws = Workspace(tmp_path / "empty")
    idx = RagIndexer(ws)
    idx.build()
    assert idx.search("anything", k=5) == []


def test_search_results_have_line_provenance(workspace_root):
    ws = Workspace(workspace_root)
    idx = RagIndexer(ws)
    idx.build()
    results = idx.search("SessionManager create session token", k=2)
    assert results
    d = results[0].to_dict()
    assert "start_line" in d and "end_line" in d and "path" in d and "score" in d
