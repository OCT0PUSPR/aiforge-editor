"""Model and training configuration dataclasses (torch-free)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass
class ModelConfig:
    """Decoder-only Transformer hyperparameters."""

    vocab_size: int = 8192
    block_size: int = 256  # max context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = True

    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 24
    grad_accum: int = 1
    max_steps: int = 2000
    warmup_steps: int = 100
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 50
    checkpoint_interval: int = 500
    fim_rate: float = 0.5
    spm_rate: float = 0.5
    seed: int = 1337
    log_interval: int = 20

    def to_dict(self) -> dict:
        return asdict(self)
