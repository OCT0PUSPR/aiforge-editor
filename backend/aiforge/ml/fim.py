"""Fill-in-the-Middle (FIM) transformation.

Implements the PSM / SPM document transform from "Efficient Training of Language
Models to Fill in the Middle" (Bavarian et al., 2022). A document is split at two
random points into (prefix, middle, suffix), then re-arranged with sentinel
tokens so the model learns to *infill* rather than only continue left-to-right:

  PSM:  <PRE> prefix <SUF> suffix <MID> middle <EOT>
  SPM:  <PRE> <SUF> suffix <MID> prefix middle <EOT>

At inference, an editor sends a prefix (text before the cursor) and a suffix
(text after). We prompt the model with ``<PRE> prefix <SUF> suffix <MID>`` and it
generates the ``middle`` (the code to insert at the cursor), stopping at <EOT>.

This module is pure-Python and torch-free, so it is unit-tested without any ML
dependency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Special-token surface strings (added to the BPE vocab as atomic tokens).
PRE = "<|fim_prefix|>"
SUF = "<|fim_suffix|>"
MID = "<|fim_middle|>"
EOT = "<|endoftext|>"
PAD = "<|pad|>"

SPECIAL_TOKENS = [EOT, PRE, SUF, MID, PAD]


@dataclass
class FimTokens:
    """Resolved integer ids for the FIM/control tokens."""

    pre: int
    suf: int
    mid: int
    eot: int
    pad: int


def split_three(ids: Sequence[int], rng: random.Random) -> "tuple[List[int], List[int], List[int]]":
    """Split ``ids`` into (prefix, middle, suffix) at two random cut points."""
    n = len(ids)
    if n < 2:
        return list(ids), [], []
    # Two distinct sorted cut points in [0, n].
    a = rng.randint(0, n)
    b = rng.randint(0, n)
    lo, hi = (a, b) if a <= b else (b, a)
    return list(ids[:lo]), list(ids[lo:hi]), list(ids[hi:])


def apply_fim(
    ids: Sequence[int],
    fim: FimTokens,
    rng: random.Random,
    *,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
) -> List[int]:
    """Maybe transform a token sequence into a FIM example.

    With probability ``fim_rate`` the document is rearranged into PSM or SPM
    order (SPM chosen with probability ``spm_rate``); otherwise it is returned
    unchanged (so the model still trains on ordinary left-to-right text). The
    returned sequence always ends with the EOT token.
    """
    if rng.random() >= fim_rate or len(ids) < 4:
        return list(ids) + [fim.eot]

    prefix, middle, suffix = split_three(ids, rng)
    if rng.random() < spm_rate:
        # SPM: suffix before prefix.
        out = [fim.pre, fim.suf] + suffix + [fim.mid] + prefix + middle
    else:
        # PSM: prefix, suffix, middle.
        out = [fim.pre] + prefix + [fim.suf] + suffix + [fim.mid] + middle
    out.append(fim.eot)
    return out


def build_infill_prompt(
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    fim: FimTokens,
) -> List[int]:
    """Build the inference prompt for infilling at a cursor (PSM order).

    The model continues after ``<MID>`` to produce the middle text.
    """
    return [fim.pre] + list(prefix_ids) + [fim.suf] + list(suffix_ids) + [fim.mid]


def stop_on_eot(generated: Sequence[int], fim: FimTokens) -> List[int]:
    """Truncate a generated id list at the first EOT / control token."""
    out: List[int] = []
    control = {fim.eot, fim.pre, fim.suf, fim.mid, fim.pad}
    for tok in generated:
        if tok in control:
            break
        out.append(tok)
    return out


def make_fim_tokens(token_to_id) -> FimTokens:
    """Resolve :class:`FimTokens` from a ``token -> id`` lookup (callable or dict)."""
    get = token_to_id.get if hasattr(token_to_id, "get") else token_to_id

    def _id(tok: str) -> int:
        value = get(tok)
        if value is None:
            raise KeyError(f"special token not in vocab: {tok}")
        return int(value)

    return FimTokens(pre=_id(PRE), suf=_id(SUF), mid=_id(MID), eot=_id(EOT), pad=_id(PAD))


def chunk_sequence(
    ids: Sequence[int], block_size: int, eot: Optional[int] = None
) -> List[List[int]]:
    """Pack a long id list into fixed-size training blocks.

    Used after applying FIM per document and concatenating with EOT separators.
    The final, short remainder is dropped (standard LM packing).
    """
    blocks: List[List[int]] = []
    for i in range(0, len(ids) - block_size, block_size):
        blocks.append(list(ids[i : i + block_size]))
    return blocks
