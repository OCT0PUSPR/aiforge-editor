"""Export a trained checkpoint to ONNX for portable, dependency-light serving.

The exported graph takes ``input_ids`` of shape ``(batch, seq)`` and returns the
next-token ``logits`` of shape ``(batch, 1, vocab)`` (the model only computes the
last position at inference). ``seq`` is a dynamic axis so a single export serves
any context length up to ``block_size``.

Run:

    python -m aiforge.ml.export_onnx --run runs/proof --out runs/proof/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from .config import ModelConfig
from .model import CodeTransformer


def export(run_dir: str, out_path: str, *, opset: int = 17) -> str:
    run = Path(run_dir)
    cfg = ModelConfig.load(run / "model_config.json")
    model = CodeTransformer(cfg)
    ckpt_path = run / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # Wrap to return only logits (ONNX export wants tensor outputs).
    class _Wrapper(torch.nn.Module):
        def __init__(self, m: CodeTransformer) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            logits, _ = self.m(input_ids)
            return logits

    wrapper = _Wrapper(model)
    example = torch.randint(0, cfg.vocab_size, (1, min(16, cfg.block_size)))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(out),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    return str(out)


def verify(onnx_path: str, run_dir: str) -> dict:
    """Run the ONNX model once and compare against the torch model's logits."""
    import numpy as np
    import onnxruntime as ort

    run = Path(run_dir)
    cfg = ModelConfig.load(run / "model_config.json")
    model = CodeTransformer(cfg)
    ckpt_path = run / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run / "ckpt.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        torch_logits, _ = model(ids)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"input_ids": ids.numpy().astype(np.int64)})[0]

    diff = float(np.abs(onnx_logits - torch_logits.numpy()).max())
    torch_pred = int(torch_logits[0, -1].argmax())
    onnx_pred = int(onnx_logits[0, -1].argmax())
    return {
        "max_abs_logit_diff": diff,
        "argmax_match": torch_pred == onnx_pred,
        "onnx_output_shape": list(onnx_logits.shape),
    }


def main(argv: Optional[list] = None) -> dict:
    p = argparse.ArgumentParser(description="Export the aiforge model to ONNX")
    p.add_argument("--run", default="runs/proof")
    p.add_argument("--out", default=None)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)
    out_path = args.out or str(Path(args.run) / "model.onnx")
    path = export(args.run, out_path)
    print(f"exported ONNX -> {path}")
    result = {"onnx_path": path}
    if args.verify:
        result.update(verify(path, args.run))
        print("verification:", result)
    return result


if __name__ == "__main__":
    main()
