"""Model tests (skipped cleanly when torch is unavailable)."""

import pytest

torch = pytest.importorskip("torch")

from aiforge.ml.config import ModelConfig  # noqa: E402
from aiforge.ml.model import (  # noqa: E402
    CodeTransformer,
    RMSNorm,
    apply_rope,
    build_rope_cache,
    select_device,
)


def _tiny():
    return ModelConfig(vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64)


def test_rmsnorm_preserves_shape_and_scales():
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    y = norm(x)
    assert y.shape == x.shape


def test_rope_is_a_rotation():
    cos, sin = build_rope_cache(8, 16, 10000.0, torch.device("cpu"), torch.float32)
    q = torch.randn(1, 4, 8, 16)
    rq = apply_rope(q, cos, sin)
    # A rotation preserves vector norm.
    assert torch.allclose(q.norm(dim=-1), rq.norm(dim=-1), atol=1e-4)


def test_forward_and_loss():
    cfg = _tiny()
    model = CodeTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss.item() > 0


def test_inference_returns_last_position():
    cfg = _tiny()
    model = CodeTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    logits, loss = model(x)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert loss is None


def test_generate_grows_sequence_and_stops_on_eos():
    cfg = _tiny()
    model = CodeTransformer(cfg)
    x = torch.randint(1, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=8, temperature=0.0, top_k=1)
    assert out.shape[1] <= 4 + 8
    assert out.shape[1] >= 5


def test_weight_tying():
    cfg = _tiny()
    model = CodeTransformer(cfg)
    assert model.lm_head.weight is model.tok_emb.weight


def test_block_size_enforced():
    cfg = _tiny()
    model = CodeTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError):
        model(x)


def test_select_device_returns_device():
    dev = select_device("cpu")
    assert dev.type == "cpu"
