"""A decoder-only Transformer for code, implemented from scratch in PyTorch.

Everything load-bearing is written here -- RMSNorm, rotary position embeddings
(RoPE), causal multi-head self-attention, the SwiGLU-ish MLP, and the
generation loop. No ``transformers``/HF modeling code is used.

``torch`` is imported at module top, but the package's ``__init__`` import-guards
this module so the backend (and CI without torch) never imports it accidentally.

References implemented by hand:
- RMSNorm: Zhang & Sennrich, 2019.
- RoPE: Su et al., 2021 (rotary embeddings applied to Q/K).
- Pre-norm decoder block: GPT-NeoX / LLaMA style.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, then cast back.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin tables for rotary embeddings.

    Returns tensors of shape ``(seq_len, head_dim)`` where the second half mirrors
    the first (the standard interleaved-by-half RoPE layout).
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)  # (seq_len, half)
    emb = torch.cat([angles, angles], dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(B, H, T, D)`` with cached cos/sin ``(T, D)``."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        # (B, T, C) -> (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Flash/SDPA when available; causal mask handled internally.
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    """Gated MLP (SwiGLU), as in LLaMA."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """Pre-norm decoder block: x + Attn(Norm(x)); x + MLP(Norm(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class CodeTransformer(nn.Module):
    """Decoder-only Transformer language model with RoPE + RMSNorm."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE cache (lazily (re)built per device/length). Plain attributes
        # (non-persistent, recomputed per device) rather than buffers.
        self._rope_len = 0
        self._cos: torch.Tensor = torch.empty(0)
        self._sin: torch.Tensor = torch.empty(0)

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, *, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_weights:
            n -= self.tok_emb.weight.numel()
        return n

    def _rope(self, seq_len: int, device, dtype):
        if seq_len > self._rope_len or self._cos.device != device or self._cos.dtype != dtype:
            cos, sin = build_rope_cache(
                max(seq_len, self.cfg.block_size),
                self.cfg.head_dim(),
                self.cfg.rope_theta,
                device,
                dtype,
            )
            self._cos, self._sin = cos, sin
            self._rope_len = self._cos.shape[0]
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> "tuple[torch.Tensor, Optional[torch.Tensor]]":
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope(T, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        # Inference: only compute logits for the last position to save compute.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` tokens, stopping at ``eos_id``."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                vals, _ = torch.topk(logits, k)
                logits[logits < vals[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if temperature <= 1e-6:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx


def select_device(prefer: str = "auto") -> "torch.device":
    """Pick a device: MPS > CUDA > CPU (override with an explicit name)."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
