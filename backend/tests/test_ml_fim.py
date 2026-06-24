"""FIM transform tests (torch-free, always run)."""

import random

from aiforge.ml.fim import (
    EOT,
    MID,
    PRE,
    SPECIAL_TOKENS,
    SUF,
    FimTokens,
    apply_fim,
    build_infill_prompt,
    make_fim_tokens,
    split_three,
    stop_on_eot,
)


def _fim():
    return FimTokens(pre=1, suf=2, mid=3, eot=0, pad=4)


def test_special_tokens_present():
    assert PRE in SPECIAL_TOKENS and SUF in SPECIAL_TOKENS
    assert MID in SPECIAL_TOKENS and EOT in SPECIAL_TOKENS


def test_split_three_partitions():
    rng = random.Random(0)
    ids = list(range(20))
    for _ in range(50):
        a, b, c = split_three(ids, rng)
        assert a + b + c == ids  # exact partition, order preserved


def test_apply_fim_psm_layout():
    rng = random.Random(1)
    fim = _fim()
    out = apply_fim(list(range(10, 30)), fim, rng, fim_rate=1.0, spm_rate=0.0)
    # PSM: <PRE> ... <SUF> ... <MID> ... <EOT>
    assert out[0] == fim.pre
    assert fim.suf in out and fim.mid in out
    assert out[-1] == fim.eot
    # sentinel ordering PRE < SUF < MID
    assert out.index(fim.pre) < out.index(fim.suf) < out.index(fim.mid)


def test_apply_fim_spm_layout():
    rng = random.Random(2)
    fim = _fim()
    out = apply_fim(list(range(10, 40)), fim, rng, fim_rate=1.0, spm_rate=1.0)
    # SPM starts <PRE> <SUF> ...
    assert out[0] == fim.pre and out[1] == fim.suf
    assert out[-1] == fim.eot


def test_apply_fim_passthrough_when_rate_zero():
    rng = random.Random(3)
    fim = _fim()
    ids = list(range(5, 15))
    out = apply_fim(ids, fim, rng, fim_rate=0.0, spm_rate=0.5)
    assert out == ids + [fim.eot]  # unchanged + EOT


def test_build_infill_prompt_and_stop():
    fim = _fim()
    prompt = build_infill_prompt([10, 11], [20, 21], fim)
    assert prompt == [fim.pre, 10, 11, fim.suf, 20, 21, fim.mid]
    gen = [30, 31, fim.eot, 99]
    assert stop_on_eot(gen, fim) == [30, 31]


def test_make_fim_tokens_from_dict():
    vocab = {EOT: 0, PRE: 1, SUF: 2, MID: 3, "<|pad|>": 4}
    fim = make_fim_tokens(vocab)
    assert fim.pre == 1 and fim.eot == 0
