"""Unified-diff parser/applier round-trip and edge-case tests."""
import pytest

from aiforge.ai.diff import (
    DiffError,
    apply_unified_diff,
    make_unified_diff,
    parse_unified_diff,
    reverse_diff,
)


def test_parse_simple_diff():
    diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old line\n"
        "+new line\n"
        " unchanged\n"
    )
    patches = parse_unified_diff(diff)
    assert len(patches) == 1
    assert patches[0].target_path() == "foo.py"
    assert len(patches[0].hunks) == 1


def test_apply_replaces_line():
    original = "old line\nunchanged\n"
    diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old line\n"
        "+new line\n"
        " unchanged\n"
    )
    result = apply_unified_diff(original, diff)
    assert result == "new line\nunchanged\n"


def test_apply_insertion():
    original = "line1\nline2\nline3\n"
    diff = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        "+inserted\n"
        " line2\n"
        " line3\n"
    )
    result = apply_unified_diff(original, diff)
    assert result == "line1\ninserted\nline2\nline3\n"


def test_apply_deletion():
    original = "keep1\ndrop\nkeep2\n"
    diff = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -1,3 +1,2 @@\n"
        " keep1\n"
        "-drop\n"
        " keep2\n"
    )
    result = apply_unified_diff(original, diff)
    assert result == "keep1\nkeep2\n"


def test_generate_and_apply_roundtrip():
    old = "alpha\nbeta\ngamma\ndelta\n"
    new = "alpha\nBETA\ngamma\nepsilon\ndelta\n"
    diff = make_unified_diff("x.txt", old, new)
    assert apply_unified_diff(old, diff) == new


def test_reverse_diff_undoes_change():
    old = "one\ntwo\nthree\nfour\nfive\n"
    new = "one\nTWO\nthree\nFOUR\nfive\n"
    fwd = make_unified_diff("x.txt", old, new)
    applied = apply_unified_diff(old, fwd)
    assert applied == new
    # Now reverse: applying the reverse diff to `new` restores `old`.
    rev = reverse_diff(fwd)
    restored = apply_unified_diff(new, rev)
    assert restored == old


def test_multi_hunk_diff():
    old = "\n".join(str(i) for i in range(1, 21)) + "\n"
    new_lines = [str(i) for i in range(1, 21)]
    new_lines[1] = "TWO"
    new_lines[17] = "EIGHTEEN"
    new = "\n".join(new_lines) + "\n"
    diff = make_unified_diff("nums.txt", old, new)
    assert len(parse_unified_diff(diff)[0].hunks) >= 1
    assert apply_unified_diff(old, diff) == new


def test_file_creation_diff():
    diff = (
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+print('hello')\n"
        "+print('world')\n"
    )
    patches = parse_unified_diff(diff)
    assert patches[0].is_creation
    result = apply_unified_diff("", diff)
    assert result == "print('hello')\nprint('world')\n"


def test_fuzzy_relocate_tolerates_offset():
    # Diff was computed against a slightly older version with two extra leading
    # lines; the applier should still locate the hunk context.
    old_for_diff = "ctx_a\ntarget\nctx_b\n"
    new = "ctx_a\nTARGET\nctx_b\n"
    diff = make_unified_diff("f.txt", old_for_diff, new)
    drifted = "header1\nheader2\nctx_a\ntarget\nctx_b\n"
    result = apply_unified_diff(drifted, diff)
    assert "TARGET" in result
    assert result.startswith("header1\nheader2\n")


def test_mismatched_context_raises():
    original = "completely\ndifferent\ncontent\n"
    diff = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-expected line\n"
        "+replacement\n"
        " also expected\n"
    )
    with pytest.raises(DiffError):
        apply_unified_diff(original, diff)


def test_malformed_diff_raises():
    with pytest.raises(DiffError):
        parse_unified_diff("not a diff at all\njust text\n")
