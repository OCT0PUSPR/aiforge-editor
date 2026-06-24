"""aiforge -- the AI engine behind the aiforge-editor.

A small, dependency-light backend that provides AI features over a sandboxed
workspace: inline completion, codebase chat with RAG, and agentic edits with a
real unified-diff apply engine. Everything works offline via the deterministic
:class:`~aiforge.llm.mock_backend.MockLLM` backend.
"""

__version__ = "0.1.0"
