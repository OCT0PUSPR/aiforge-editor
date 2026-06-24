# aiforge.ml — a from-scratch code completion model

A decoder-only Transformer for code, implemented from scratch in PyTorch, with a
byte-level BPE tokenizer and **fill-in-the-middle (FIM)** training so it does
real infill (not just left-to-right continuation). No `transformers`/HF modeling
code — the attention, RoPE, RMSNorm, blocks, training loop, and generation are
all written here.

> **torch is import-guarded.** Importing `aiforge.ml` does not import torch; the
> backend API and CI run without it. Install training deps with
> `pip install -r requirements-train.txt`.

## What's implemented from scratch

| Component | File | Notes |
|---|---|---|
| RMSNorm | `model.py` | Root-mean-square norm, no mean subtraction, fp32 reduction |
| RoPE | `model.py` | Rotary position embeddings applied to Q/K (verified norm-preserving) |
| Causal self-attention | `model.py` | Multi-head, RoPE'd, SDPA causal kernel |
| SwiGLU MLP | `model.py` | Gated FFN (LLaMA-style) |
| Decoder block + model | `model.py` | Pre-norm residual blocks, tied embeddings, scaled init |
| Autoregressive generation | `model.py` | Temperature + top-k sampling, EOS stop |
| BPE tokenizer | `tokenizer.py` | Byte-level BPE (via `tokenizers`) + FIM special tokens |
| PSM/SPM FIM transform | `fim.py` | The Bavarian et al. (2022) document rearrangement |
| Corpus + FIM dataloader | `data.py` | stdlib + this repo; FIM-transformed, packed into blocks |
| Training pipeline | `train.py` | AdamW, cosine LR + warmup, grad clip, eval, checkpoint, resume |
| Evaluation | `eval.py` | Next-token accuracy + FIM exact-match on held-out snippets |
| Inference | `generate.py` | `CodeCompleter`: builds the PSM infill prompt, samples the middle |
| ONNX export | `export_onnx.py` | Portable serving graph + verification vs torch |
| Scale-up | `train_scaleup.py` | Same architecture, larger, on a streamed subset of The Stack (GPU) |

## Fill-in-the-Middle

Each training document is split at two random points into `(prefix, middle,
suffix)` and rearranged with sentinel tokens:

```
PSM:  <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle <|endoftext|>
SPM:  <|fim_prefix|> <|fim_suffix|> suffix <|fim_middle|> prefix middle <|endoftext|>
```

At inference, the editor sends the text before the cursor (`prefix`) and after
(`suffix`); we prompt with `<PRE> prefix <SUF> suffix <MID>` and the model
generates the code to insert, stopping at `<|endoftext|>`.

## Train the proof model (CPU/MPS/CUDA, ~25 min on Apple Silicon)

```bash
pip install -r requirements-train.txt
python -m aiforge.ml.train --out runs/proof --max-steps 2000
```

Device auto-selects **MPS > CUDA > CPU**. Checkpoints (`ckpt.pt`, `best.pt`),
the tokenizer (`tokenizer.json`), and config land in `runs/proof/`. Training is
resumable with `--resume`.

## Evaluate

```bash
python -m aiforge.ml.eval --run runs/proof
# -> next_token_acc, fim_exact_match, fim_first_line_match, fim_nonempty_rate
```

## Export to ONNX

```bash
python -m aiforge.ml.export_onnx --run runs/proof --verify
# verifies ONNX logits match the torch model (max abs diff ~1e-6)
```

## Serve it locally

Point the inline-completion backend at the trained model:

```bash
export AIFORGE_COMPLETE_BACKEND=local
export AIFORGE_LOCAL_MODEL_DIR=backend/runs/proof
```

`/api/.../ai/complete` (and the Monaco inline-completion provider) will now be
served by your own model. Chat and agentic edits keep their configured backend
(Anthropic in production), via the resilient failover chain. If no checkpoint is
present, the local backend degrades to the deterministic mock — nothing breaks.

## Scale up (bigger model, The Stack, GPU)

```bash
pip install -r requirements-train.txt datasets
python -m aiforge.ml.train_scaleup \
    --out runs/stack-small \
    --dataset bigcode/the-stack-dedup --data-dir data/python \
    --max-docs 50000 --vocab-size 32000 --block-size 1024 \
    --n-layer 12 --n-head 12 --n-embd 768 \
    --batch-size 16 --grad-accum 8 --max-steps 20000
```

Same architecture — only the corpus, tokenizer size, model width, and schedule
scale up.

## Honest scope

The bundled proof model is **small (≈17M params)** and trained briefly on a
**modest stdlib corpus** to demonstrate the full, correct pipeline end-to-end on
a laptop. It learns Python lexical/local structure; it is **not** competitive
with large code LLMs on coherent multi-line infill. For production-quality
completion, use `train_scaleup.py` on a GPU, or route completion to Anthropic /
HuggingFace (both are wired in). The architecture and training code are the
deliverable; the checkpoint is a proof of correctness.
```
