# Model plan

## Default: Qwen3-30B-A3B-Instruct-2507 — Q4_K_M (~18.6 GiB)

Why this model:

- **It cannot fit**: 18.6 GiB > 7 GiB runner RAM and > 14 GB runner disk. Inference is
  only possible by paging the weights from repovram and the KV from reporam. That is the
  provable claim.
- **MoE makes sparse paging meaningful**: 48 layers, 32 experts, top-8 routing. A token
  touches ~25% of expert bytes, so the page cache and the sparse bins do real work instead
  of streaming a dense model start-to-finish.
- **Capability**: current-generation MoE (2507 refresh), strong reasoning/chat quality at
  ~3B active parameters per token — a genuinely capable model, not a toy.

GGUF facts used by the engine (llama.cpp naming):

```
arch=qwen3moe  L=48  d=2048  heads=32  kv_heads=4  head_dim=128
experts=32  used=8  expert_ff=768  ctx=40960  rope_base=1e6  rms_eps=1e-6
token_embd, output_norm, output,
blk.N.attn_norm/attn_q/attn_q_norm/attn_k/attn_k_norm/attn_v/attn_output,
blk.N.ffn_norm/ffn_gate_inp, blk.N.ffn_{gate,up,down}_exps (3-D, per-expert contiguous)
```

Qwen3-specific details implemented and verified against `src/models/qwen3moe.cpp`:
per-head QK RMSNorm **before** NEOX rope (half-split, base 1e6), softmax top-k routing
with renormalization, SiLU-gated experts, residual order `x + attn; x + moe`.

## Supported quant types

`F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K` — dequant is numpy-vectorized and byte-exact
against scalar ports of `ggml-quants.c` (see `tests/test_dequant.py`). Q4_K_M uses Q4_K
for most tensors and Q6_K for attention/output; both are covered. IQ-quants are rejected
with a clear error.

## Capacity

| model | Q4_K_M size | fits 31.9 GiB tier | notes |
|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | ~18.6 GiB | yes, bins 0–9 | **default** |
| Qwen3-32B (dense) | ~19.8 GiB | yes | slower: every weight is touched per token |
| Qwen3-30B-A3B Q5_K_M | ~22 GiB | yes | better quality, higher bandwidth |
| Qwen3-30B-A3B **Q3_K_M** | ~13 GiB | yes | fits 16 GiB RAM / 14 GB SSD → fast mode, lower quality |
| Qwen3-30B-A3B Q2_K | ~12 GiB | yes | fastest fit, noticeably degraded |
| Qwen3-30B-A3B Q8_0 | ~32.5 GiB | no | exceeds 8159 pages |
| Qwen3-235B-A22B | >120 GiB | no | out of scope |

Swap the model with `03-load-model`'s `url` + `key` inputs (any single-file GGUF URL with
Content-Length and Range; split GGUFs are not supported). The engine reads the architecture
from the GGUF header; `qwen3` dense is also accepted.

## Performance expectations (2-core hosted runner)

These are estimates to set expectations; the load-model and boot-single workflows print
measured statistics (`STATS {...}`) from the runner.

| phase | cost |
|---|---|
| prewarm (headers + first layers) | 30–90 s, ~128 MiB |
| prefill 2k tokens (chunk 192) | 4–10 min, 16–32 GiB paged |
| decode token (warm experts) | 3–8 s |
| decode token (cold experts) | 6–12 s |
| 32-token shift | ~20 min incl. prefill amortization |

The 24/7 loop is deliberately shift-based: each runner job stays well under an hour, state
is durable in reporam, and the ring rotates. Quality is the real model's quality; latency
is the price of running a 30B MoE on repo-backed sparse memory.

## Chat format

ChatML (Qwen3-2507 non-thinking):

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

Sampling defaults: temp 0.7, top-k 20, top-p 0.8, repetition penalty 1.05.
