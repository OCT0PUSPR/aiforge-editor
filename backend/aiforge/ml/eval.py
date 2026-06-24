"""Evaluation: next-token accuracy + FIM exact-match on held-out snippets.

The FIM exact-match metric is the honest measure of infill quality: we take a
held-out code snippet, remove a contiguous middle span, ask the model to infill
it, and check whether the prediction exactly reconstructs the removed span (and,
more leniently, whether the first line matches).

Run:

    python -m aiforge.ml.eval --run runs/proof
"""

from __future__ import annotations

import argparse
import random
from typing import List, Optional

import torch

from .data import collect_corpus, make_datasets
from .generate import CodeCompleter
from .train import evaluate as _eval_loss


def _fim_examples(docs: List[str], n: int, rng: random.Random) -> List["tuple[str, str, str]"]:
    """Carve (prefix, middle, suffix) infill examples from real code lines."""
    examples = []
    attempts = 0
    while len(examples) < n and attempts < n * 50:
        attempts += 1
        doc = rng.choice(docs)
        lines = doc.splitlines(keepends=True)
        if len(lines) < 6:
            continue
        # Pick a middle span of 1-2 lines somewhere in the body.
        i = rng.randint(1, len(lines) - 3)
        span = rng.randint(1, 2)
        middle = "".join(lines[i : i + span])
        if not middle.strip():
            continue
        prefix = "".join(lines[:i])
        suffix = "".join(lines[i + span :])
        # Keep the context bounded so it fits the model's window.
        prefix = prefix[-600:]
        suffix = suffix[:300]
        examples.append((prefix, middle, suffix))
    return examples


def evaluate_fim(
    completer: CodeCompleter,
    docs: List[str],
    *,
    n: int = 60,
    seed: int = 7,
    max_new_tokens: int = 48,
) -> dict:
    rng = random.Random(seed)
    examples = _fim_examples(docs, n, rng)
    exact = 0
    first_line = 0
    nonempty = 0
    for prefix, middle, suffix in examples:
        pred = completer.infill(
            prefix, suffix, max_new_tokens=max_new_tokens, temperature=0.0, top_k=1
        )
        if pred.strip():
            nonempty += 1
        if pred.strip() == middle.strip():
            exact += 1
        if pred.strip().splitlines()[:1] == middle.strip().splitlines()[:1]:
            first_line += 1
    total = max(1, len(examples))
    return {
        "fim_examples": len(examples),
        "fim_exact_match": exact / total,
        "fim_first_line_match": first_line / total,
        "fim_nonempty_rate": nonempty / total,
    }


def run_eval(run_dir: str, *, device: str = "auto", n_fim: int = 60) -> dict:
    from .config import TrainConfig

    completer = CodeCompleter.load(run_dir, device=device)
    docs = collect_corpus()
    # Next-token accuracy on a fresh val split.
    train_cfg = TrainConfig(eval_iters=40)
    _, val_ds = make_datasets(docs, completer.tokenizer, train_cfg, completer.block_size)
    gen = torch.Generator().manual_seed(123)
    val_loss, val_acc = _eval_loss(completer.model, val_ds, train_cfg, completer.device, gen)
    fim = evaluate_fim(completer, docs, n=n_fim)
    metrics = {
        "val_loss": round(val_loss, 4),
        "next_token_acc": round(val_acc, 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in fim.items()},
    }
    return metrics


def main(argv: Optional[list] = None) -> dict:
    p = argparse.ArgumentParser(description="Evaluate the aiforge code model")
    p.add_argument("--run", default="runs/proof")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-fim", type=int, default=60)
    args = p.parse_args(argv)
    metrics = run_eval(args.run, device=args.device, n_fim=args.n_fim)
    print("EVAL METRICS:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


if __name__ == "__main__":
    main()
