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

## Faster options (trade quality/claims)

| option | size | effect |
|---|---|---|
| Q3_K_M | ~13 GiB | fits runner SSD + 16 GiB RAM; prefill from disk, decode mostly RAM → est. 0.3–1 tok/s |
| Q2_K | ~12 GiB | same fit, lower quality |
| Q4_K_M (current) | 17.28 GiB | **does not fit RAM or disk**; proves the sparse repo-VRAM claim at 0.05 tok/s |
| prefix KV reuse | — | reuses the previous turn's KV instead of re-prefilling the whole chat history |

`03-load-model` accepts any single-file GGUF URL; switching to Q3_K_M is a re-import plus a
`vram/model` record update. Keep Q4 if the point is "it cannot fit anywhere".
