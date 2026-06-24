"""Byte-level BPE tokenizer for code, with FIM special tokens.

Wraps HuggingFace ``tokenizers`` (used *only* for BPE, as permitted) to train a
byte-level BPE on a code corpus and expose a small, stable API the rest of the
ML package uses. The FIM/control tokens (``<|fim_prefix|>`` etc.) are registered
as atomic special tokens so they never get split by BPE.

The dependency is import-guarded: importing this module fails clearly if
``tokenizers`` is absent, but it is never imported at backend startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from .fim import SPECIAL_TOKENS, FimTokens, make_fim_tokens


def _require_tokenizers():
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'tokenizers' package is required for BPE training/loading. "
            "Install it with: pip install -r requirements-train.txt"
        ) from exc


class CodeTokenizer:
    """A trained byte-level BPE tokenizer with FIM special tokens."""

    def __init__(self, tokenizer) -> None:
        self._tk = tokenizer
        self.fim: FimTokens = make_fim_tokens(self.token_to_id)

    # -- construction -------------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        *,
        vocab_size: int = 8192,
        min_frequency: int = 2,
    ) -> "CodeTokenizer":
        """Train a byte-level BPE tokenizer on ``texts``."""
        _require_tokenizers()
        from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
        from tokenizers.models import BPE

        tk = Tokenizer(BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tk.train_from_iterator(texts, trainer=trainer)
        return cls(tk)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CodeTokenizer":
        _require_tokenizers()
        from tokenizers import Tokenizer

        tk = Tokenizer.from_file(str(path))
        return cls(tk)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tk.save(str(path))

    # -- vocab --------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tk.token_to_id(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self._tk.id_to_token(idx)

    # -- encode / decode ----------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self._tk.encode(text).ids

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        return self._tk.decode(list(ids), skip_special_tokens=skip_special)
