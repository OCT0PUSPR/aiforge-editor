"""Tests for the local model backend wiring (graceful fallback + FIM parse)."""

from aiforge.llm import get_backend
from aiforge.llm.local_backend import local_model_exists


def test_local_model_exists_false_for_missing(tmp_path):
    assert local_model_exists(None) is False
    assert local_model_exists(str(tmp_path / "nope")) is False


def test_get_backend_local_falls_back_to_mock(tmp_path):
    # No checkpoint at this dir -> backend factory must degrade to mock.
    backend = get_backend("local", local_model_dir=str(tmp_path / "missing"))
    assert backend.name == "mock"


def test_local_model_exists_true_when_files_present(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "model_config.json").write_text("{}")
    (run / "tokenizer.json").write_text("{}")
    (run / "best.pt").write_text("x")
    assert local_model_exists(str(run)) is True


def test_fim_prompt_extraction_regex():
    # The local backend extracts <prefix>/<suffix> from the completion prompt.
    from aiforge.llm.local_backend import _PREFIX_RE, _SUFFIX_RE

    prompt = (
        "Complete the code.\n<prefix>\ndef f():\n</prefix>\n<suffix>\n    return 1\n</suffix>\n"
    )
    pm = _PREFIX_RE.search(prompt)
    sm = _SUFFIX_RE.search(prompt)
    assert pm and "def f():" in pm.group(1)
    assert sm and "return 1" in sm.group(1)
