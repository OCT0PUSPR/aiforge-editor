"""Training pipeline for the from-scratch code completion model.

Implements: AdamW with decoupled weight decay (decay only 2D params), a cosine
learning-rate schedule with linear warmup, gradient clipping, periodic
evaluation (val loss + next-token accuracy), checkpointing, and resume.

Run as a module:

    python -m aiforge.ml.train --out runs/proof --max-steps 1500

Device is auto-selected MPS > CUDA > CPU.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .config import ModelConfig, TrainConfig
from .data import collect_corpus, make_datasets
from .model import CodeTransformer, select_device
from .tokenizer import CodeTokenizer


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def configure_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to 2D (matrix) parameters."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def evaluate(model, dataset, cfg, device, generator) -> "tuple[float, float]":
    """Return ``(mean_val_loss, next_token_accuracy)`` over ``cfg.eval_iters`` batches."""
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    correct = 0
    total = 0
    for i in range(cfg.eval_iters):
        x, y = dataset.batch(cfg.batch_size, device, generator)
        logits, loss = model(x, y)
        losses[i] = loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    model.train()
    return float(losses.mean()), (correct / total if total else 0.0)


def save_checkpoint(path: Path, model, optimizer, step, model_cfg, train_cfg, best_val) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": model_cfg.to_dict(),
            "train_config": asdict(train_cfg),
            "best_val": best_val,
        },
        path,
    )


def train(
    out_dir: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    device_name: str = "auto",
    extra_dir: Optional[str] = None,
    resume: bool = False,
    log_file: Optional[str] = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    logf = open(log_file or (out / "train.log"), "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"# device={device} seed={train_cfg.seed}")

    # -- tokenizer ----------------------------------------------------------
    tok_path = out / "tokenizer.json"
    docs = collect_corpus(extra_dir)
    log(f"# corpus: {len(docs)} documents")
    if tok_path.exists():
        tokenizer = CodeTokenizer.load(tok_path)
        log(f"# loaded tokenizer (vocab={tokenizer.vocab_size})")
    else:
        log("# training BPE tokenizer...")
        tokenizer = CodeTokenizer.train(docs, vocab_size=model_cfg.vocab_size)
        tokenizer.save(tok_path)
        log(f"# trained tokenizer (vocab={tokenizer.vocab_size})")
    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.save(out / "model_config.json")

    # -- data ---------------------------------------------------------------
    train_ds, val_ds = make_datasets(docs, tokenizer, train_cfg, model_cfg.block_size)
    log(f"# tokens: train={len(train_ds.data)} val={len(val_ds.data)}")

    # -- model / optim ------------------------------------------------------
    model = CodeTransformer(model_cfg).to(device)
    optimizer = configure_optimizer(model, train_cfg)
    log(f"# model params: {model.num_params():,}")

    start_step = 0
    best_val = float("inf")
    ckpt_path = out / "ckpt.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", best_val)
        log(f"# resumed from step {start_step}")

    gen = torch.Generator().manual_seed(train_cfg.seed)
    model.train()
    t0 = time.time()
    running = 0.0
    final_metrics = {}

    for step in range(start_step, train_cfg.max_steps):
        lr = cosine_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = train_ds.batch(train_cfg.batch_size, device, gen)
            _, loss = model(x, y)
            loss = loss / train_cfg.grad_accum
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        running += loss_accum

        if step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            log(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | {dt:.1f}s")
            running = 0.0

        if step > 0 and step % train_cfg.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_ds, train_cfg, device, gen)
            log(f"  eval @ {step}: val_loss {val_loss:.4f} | next-token acc {val_acc:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    out / "best.pt", model, optimizer, step, model_cfg, train_cfg, best_val
                )

        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(ckpt_path, model, optimizer, step, model_cfg, train_cfg, best_val)

    # Final eval + checkpoint.
    val_loss, val_acc = evaluate(model, val_ds, train_cfg, device, gen)
    save_checkpoint(
        ckpt_path, model, optimizer, train_cfg.max_steps, model_cfg, train_cfg, best_val
    )
    if val_loss < best_val:
        save_checkpoint(
            out / "best.pt", model, optimizer, train_cfg.max_steps, model_cfg, train_cfg, val_loss
        )
    final_metrics = {
        "final_val_loss": val_loss,
        "final_next_token_acc": val_acc,
        "best_val_loss": min(best_val, val_loss),
        "steps": train_cfg.max_steps,
        "params": model.num_params(),
        "vocab_size": tokenizer.vocab_size,
        "elapsed_s": round(time.time() - t0, 1),
    }
    log(f"# DONE {final_metrics}")
    logf.close()
    return final_metrics


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the aiforge code completion model")
    p.add_argument("--out", default="runs/proof")
    p.add_argument("--device", default="auto")
    p.add_argument("--extra-dir", default=None)
    p.add_argument("--resume", action="store_true")
    # model
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    # train
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=50)
    return p


def main(argv: Optional[list] = None) -> dict:
    args = _build_argparser().parse_args(argv)
    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )
    return train(
        args.out,
        model_cfg,
        train_cfg,
        device_name=args.device,
        extra_dir=args.extra_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
