"""AI feature tests against the offline MockLLM backend."""

from aiforge.ai import chat as chat_feature
from aiforge.ai import completion as completion_feature
from aiforge.ai.edit import apply_edit, apply_full_content, propose_edit
from aiforge.llm import MockLLM, collect
from aiforge.rag.indexer import RagIndexer
from aiforge.workspace.files import Workspace


def test_mock_completion_streams_text():
    backend = MockLLM()
    chunks = list(
        completion_feature.complete(
            backend, prefix="def greet(name):\n", suffix="", language="python"
        )
    )
    text = "".join(chunks)
    assert len(chunks) >= 1  # it streamed in pieces
    assert text  # non-empty completion


def test_mock_completion_deterministic():
    backend = MockLLM()
    a = collect(completion_feature.complete(backend, prefix="x = ", suffix=""))
    b = collect(completion_feature.complete(backend, prefix="x = ", suffix=""))
    assert a == b


def test_mock_chat_returns_references(workspace_root):
    ws = Workspace(workspace_root)
    idx = RagIndexer(ws)
    idx.build()
    backend = MockLLM()
    stream, results = chat_feature.chat(backend, idx, question="How does password hashing work?")
    answer = collect(stream)
    assert answer
    assert results  # RAG retrieved context
    assert any(r.chunk.path == "src/auth.py" for r in results)


def test_propose_edit_produces_diff(workspace_root):
    ws = Workspace(workspace_root)
    backend = MockLLM()
    proposal = propose_edit(
        backend, ws, path="src/calculator.py", instruction="add a header comment"
    )
    assert proposal.changed
    assert "--- a/src/calculator.py" in proposal.diff or proposal.diff.startswith("---")
    # MockLLM prepends a header comment deterministically.
    assert proposal.new_content.startswith("# Edited by aiforge MockLLM")


def test_apply_edit_writes_workspace(workspace_root):
    ws = Workspace(workspace_root)
    backend = MockLLM()
    proposal = propose_edit(
        backend, ws, path="src/calculator.py", instruction="add a header comment"
    )
    result = apply_edit(
        ws,
        path="src/calculator.py",
        diff=proposal.diff,
        expected_original=proposal.original,
    )
    on_disk = ws.read("src/calculator.py")
    assert on_disk == result.new_content
    assert on_disk.startswith("# Edited by aiforge MockLLM")
    # The reverse diff restores the original.
    from aiforge.ai.diff import apply_unified_diff

    assert apply_unified_diff(on_disk, result.reverse_diff) == proposal.original


def test_apply_full_content(workspace_root):
    ws = Workspace(workspace_root)
    apply_full_content(ws, path="src/calculator.py", new_content="print('replaced')\n")
    assert ws.read("src/calculator.py") == "print('replaced')\n"
