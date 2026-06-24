"""Inference: load a trained checkpoint and run FIM infill / completion.

:class:`CodeCompleter` is the runtime object the local backend uses. It loads a
checkpoint + tokenizer, builds the PSM infill prompt from a (prefix, suffix)
pair, samples the middle, and decodes it -- stopping at the EOT/control tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Union

import torch

from .config import ModelConfig
from .fim import build_infill_prompt, stop_on_eot
from .model import CodeTransformer, select_device
from .tokenizer import CodeTokenizer


class CodeCompleter:
    """Runtime wrapper around a trained model for code infill/completion."""

    def __init__(
        self,
        model: CodeTransformer,
        tokenizer: CodeTokenizer,
        device: "torch.device",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def load(
        cls, run_dir: Union[str, Path], *, device: str = "auto", checkpoint: str = "best.pt"
    ) -> "CodeCompleter":
        run = Path(run_dir)
        dev = select_device(device)
        cfg = ModelConfig.load(run / "model_config.json")
        tokenizer = CodeTokenizer.load(run / "tokenizer.json")
        model = CodeTransformer(cfg)
        ckpt_path = run / checkpoint
        if not ckpt_path.exists():
            ckpt_path = run / "ckpt.pt"
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state["model"])
        model.to(dev)
        return cls(model, tokenizer, dev)

    @property
    def block_size(self) -> int:
        return self.model.cfg.block_size

    # -- infill -------------------------------------------------------------
    def infill(
        self,
        prefix: str,
        suffix: str = "",
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.6,
        top_k: Optional[int] = 40,
    ) -> str:
        """Return the predicted text to insert between ``prefix`` and ``suffix``."""
        fim = self.tokenizer.fim
        prefix_ids = self.tokenizer.encode(prefix)
        suffix_ids = self.tokenizer.encode(suffix)
        prompt = build_infill_prompt(prefix_ids, suffix_ids, fim)
        # Truncate the prompt to fit the context window, keeping the tail.
        prompt = prompt[-(self.block_size - max_new_tokens - 1) :]
        idx = torch.tensor([prompt], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=fim.eot
        )
        gen_ids = out[0, len(prompt) :].tolist()
        middle_ids = stop_on_eot(gen_ids, fim)
        return self.tokenizer.decode(middle_ids)

    def complete(
        self,
        prefix: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.6,
        top_k: Optional[int] = 40,
    ) -> str:
        """Left-to-right completion (no suffix)."""
        return self.infill(
            prefix, "", max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )

    def stream(
        self,
        prefix: str,
        suffix: str = "",
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.6,
        top_k: Optional[int] = 40,
    ) -> Iterator[str]:
        """Yield the completion incrementally (token-group decoded)."""
        # Simple approach: generate fully, then chunk the decoded text. (The
        # tiny proof model is fast enough that true per-token streaming adds
        # little; the server wraps this in SSE regardless.)
        text = self.infill(
            prefix, suffix, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )
        for i in range(0, len(text), 8):
            yield text[i : i + 8]
