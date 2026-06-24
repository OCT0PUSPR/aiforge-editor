"""Scale-up training: a larger code LM on a subset of The Stack (GPU).

This trains the SAME from-scratch architecture (:class:`CodeTransformer`) at a
larger size on a streamed subset of `bigcode/the-stack-dedup` (or any HF dataset
of code), instead of the bundled stdlib corpus. It is intended for a CUDA GPU.

It is import-guarded and **optional**: it needs ``datasets`` (and a GPU to be
practical). The bundled proof model (``aiforge.ml.train``) requires none of
this.

Example (single GPU)::

    pip install -r requirements-train.txt datasets
    python -m aiforge.ml.train_scaleup \\
        --out runs/stack-small \\
        --dataset bigcode/the-stack-dedup --data-dir data/python \\
        --max-docs 50000 --vocab-size 32000 --block-size 1024 \\
        --n-layer 12 --n-head 12 --n-embd 768 \\
        --batch-size 16 --grad-accum 8 --max-steps 20000

The architecture is unchanged; only the corpus, tokenizer size, model width, and
schedule scale up. Mixed precision (bf16/fp16) is enabled automatically on CUDA.
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def stream_stack_python(dataset: str, data_dir: str, max_docs: int) -> List[str]:
    """Stream code documents from a HuggingFace dataset (e.g. The Stack)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pip install datasets to stream The Stack. See requirements-train.txt"
        ) from exc

    ds = load_dataset(dataset, data_dir=data_dir, split="train", streaming=True)
    docs: List[str] = []
    for row in ds:
        content = row.get("content") or row.get("text") or ""
        if content and 200 < len(content) < 50_000:
            docs.append(content)
        if len(docs) >= max_docs:
            break
    return docs


def main(argv: Optional[list] = None) -> dict:
    p = argparse.ArgumentParser(description="Scale-up training on The Stack subset")
    p.add_argument("--out", default="runs/stack-small")
    p.add_argument("--dataset", default="bigcode/the-stack-dedup")
    p.add_argument("--data-dir", default="data/python")
    p.add_argument("--max-docs", type=int, default=50000)
    p.add_argument("--device", default="auto")
    # model
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    # train
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args(argv)

    # Import torch-dependent pieces lazily.
    from pathlib import Path

    from .config import ModelConfig, TrainConfig
    from .tokenizer import CodeTokenizer
    from .train import train

    print(f"# streaming {args.max_docs} docs from {args.dataset}/{args.data_dir} ...")
    docs = stream_stack_python(args.dataset, args.data_dir, args.max_docs)
    print(f"# collected {len(docs)} documents")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Train the tokenizer on the streamed corpus and cache it.
    tok_path = out / "tokenizer.json"
    if not tok_path.exists():
        tok = CodeTokenizer.train(docs, vocab_size=args.vocab_size)
        tok.save(tok_path)

    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        eval_interval=1000,
        checkpoint_interval=2000,
    )
    # The bundled trainer reuses its own corpus collector; for scale-up we
    # monkeypatch the corpus by writing docs to a temp dir the trainer reads.
    # Simpler: persist docs to a file the trainer's extra-dir picks up.
    extra = out / "corpus"
    extra.mkdir(exist_ok=True)
    for i, doc in enumerate(docs):
        (extra / f"doc_{i}.py").write_text(doc, encoding="utf-8")

    metrics = train(
        args.out,
        model_cfg,
        train_cfg,
        device_name=args.device,
        extra_dir=str(extra),
        resume=True,
    )
    print("SCALE-UP DONE:", metrics)
    return metrics


if __name__ == "__main__":
    main()
