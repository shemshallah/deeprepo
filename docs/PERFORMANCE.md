# Measured performance (GitHub-hosted, public repo runner)

Model: Qwen3-30B-A3B-Instruct-2507 Q4_K_M (17.28 GiB, 4425 vram pages, 128 experts / 8 active).
Runner: ubuntu-latest, 4 vCPU / 15.6 GiB RAM / ~14 GB SSD. Weights page from repovram over
HTTPS Range; KV lives in reporam; Koyeb is not in the data path.

| phase | best measured | notes |
|---|---|---|
| import (one-time) | 40 min | 17.3 GiB streamed from HF + uploaded to release bins |
| prefill (67-token prompt) | **186 s** (best), 298–422 s observed | expert union covers ~all 128 experts/layer → ~18.7 GiB paged |
| decode | **21 s/token** (0.047 tok/s), 8-token shift | 0.022 tok/s before keep-alive/prefetch/caches |
| first token after prefill | ~1–3 s | prefill includes all experts |
| handoff state | KB–MB | KV rows already in reporam; no weight transfer |
| API quota after a full boot | 4944/5000 remaining | ETag/304 revalidation makes polling nearly free |

Decode is **latency-bound**, not bandwidth-bound: a 4 MiB page covers ~1.1 expert slices
(0.88–1.29 MiB), so a token's 384 expert reads become ~40–1,150 range requests depending on
cache hits. Measured improvements came from: persistent HTTPS connections, 32–64 MiB read
runs, parallel slice prefetch per layer, 8 GiB RAM page cache + 8 GiB SSD page cache, and
ETag/304 journal polling. 32 concurrent CDN connections made prefill worse; 8 is the sweet
spot.

## Why not "just download the model to the runner SSD"

The model is 17.28 GiB; a hosted runner has ~14 GB of SSD. It does not fit. An 8 GiB SSD
page cache is used instead, which is why decode hits ~86% of pages (measured 163 MB/token
network) — the remaining misses still cost a CDN round trip per 4 MiB page.

## Hardware tiers (what a sponsor buys)

| tier | hardware | what fits | expected decode |
|---|---|---|---|
| classic (now) | 4 vCPU / 16 GiB / 14 GB SSD | nothing: Q4 pages from CDN | **0.05 tok/s** |
| storage sponsor | + 50–100 GB SSD | Q4 on disk (mmap/page cache) | 0.2–0.8 tok/s |
| **RAM sponsor** | 32–64 GiB RAM | Q4/Q5/Q6 fully in RAM | **2–10 tok/s** (CPU-bound) |
| GPU sponsor | 24 GB VRAM | Q4-Q8 in VRAM | **30–100 tok/s** |

A RAM offer is worth more than storage; a GPU offer changes the demo from "impossible
paging" to "usable chat". Keep the hosted classic tier as the proof and run the gold tier
for daily use.

## Faster options (trade quality/claims)

| option | size | effect |
|---|---|---|
| Q3_K_M | ~13 GiB | fits runner SSD + 16 GiB RAM; prefill from disk, decode mostly RAM → est. 0.3–1 tok/s |
| Q2_K | ~12 GiB | same fit, lower quality |
| Q4_K_M (current) | 17.28 GiB | **does not fit RAM or disk**; proves the sparse repo-VRAM claim at 0.05 tok/s |
| prefix KV reuse | — | reuses the previous turn's KV instead of re-prefilling the whole chat history |
| 1 MiB pages | same model | 4 MiB pages over-fetch ~4x per expert slice; a 1 MiB page format needs a re-import but cuts decode bytes ~2–4x |
| hot-expert pinning | — | 1 GiB LFU cache, second-use admission; keeps reused experts, cuts repeat paging |
| 1 MiB decode reads | — | fine reader bypasses 4 MiB page overfetch for 0.88–1.29 MiB expert slices; no re-import |
| speculative decoding | +0.6B draft | propose K tokens with a small model, verify in one 30B forward; 2–3x, complex |

`03-load-model` accepts any single-file GGUF URL; switching to Q3_K_M is a re-import plus a
`vram/model` record update. Keep Q4 if the point is "it cannot fit anywhere".
