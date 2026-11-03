# Memory map — the 32/32 GiB sparse registers

## Geometry (one tier, identical for reporam and repovram)

| field | value |
|---|---|
| page | 4 MiB |
| pages per bin | 510 |
| bins (slots) | 16 |
| bin header | 4 MiB |
| bin asset | 4 MiB + 510 x 4 MiB = **2044 MiB** (< 2 GiB GitHub asset cap) |
| payload per bin | 2040 MiB |
| virtual size | 16 x 510 x 4 MiB = 32,640 MiB = **31.875 GiB** (~34.2 GB) |
| max visible assets | 16 bins = ~31.9 GiB (only bins that hold data exist) |

Two tiers, same geometry: **repovram 31.875 GiB + reporam 31.875 GiB** of sparse virtual
address space (the "32/32" registers). Empty space costs zero bytes and zero API calls.

Bin `b` holds pages `[b*510, (b+1)*510)`. Page 0 of each tier carries the DPRSB1
superblock; repovram weights start at page 1. An absent bin reads as zeros (a forged but
valid ram16 header keeps the slot verifier happy); a trimmed bin pads zeros past its
written extent.

## repovram layout

| region | pages | content |
|---|---|---|
| superblock | 0 | DPRSB1 JSON: geometry, machine id, role |
| weights | 1 .. 8159 | GGUF bytes, page `p` ↔ source offset `(p-1)*4 MiB` |

- Max model: 8159 pages = 31.87 GiB (4 MiB page padding at the end).
- Qwen3-30B-A3B-Instruct-2507 Q4_K_M (~18.6 GiB) occupies pages 1..4762 → bins 0..9;
  bins 10..15 never exist.
- Import is resumable per bin (`imp-<seq>.json` progress records on `repovram-j`).
- Bin 0's superblock page is copied over from the live asset during import so a fresh
  runner cannot wipe it.

## reporam layout (defaults)

| region | pages | bytes | role |
|---|---|---|---|
| superblock | 0 | 4 MiB | DPRSB1 |
| kv | 1 .. 1024 | 4 GiB | K/V rows for every layer |
| regs | 1025 .. 1280 | 1 GiB | RegionLog: ring lease, jobs, tokens, budgets, transcripts |
| scratch | 1281 .. 8159 | ~27 GiB | future use (activations, checkpoints) |

KV row = `n_kv_heads * head_dim * 2 bytes` (fp16) per side per layer.
For Qwen3-30B-A3B (48 layers, 4 kv heads, head_dim 128): 1024 B per row/side,
**2048 B per token per layer → 98,304 B/token**. With 4 GiB reserved and the current
per-layer stride (K+V halves), usable context ≈ **21.8k tokens**; the physical ceiling of
the region is ~43.7k tokens if the stride is repacked.

Only written pages exist. 64 decode tokens touch ~6.3 MiB of KV (≈2 pages per layer at
most), and the journal compresses long zero runs, so a session costs a few hundred KB to
a few MB of assets before compaction.

## L0 (runner-local, strictly bounded)

| cache | default | purpose |
|---|---|---|
| page LRU | 384 pages = 1.5 GiB | raw 4 MiB tier pages |
| dequant f16 cache | 1856 MiB | attention/norm/router tensors as f16 |
| output head | ~0.62 GiB f16 | loaded once, f32 chunks for logits (no 1.2 GiB transient) |
| KV hot window | inside the page LRU | recent rows are page-cache hits |
| engine workspace | ~1 GiB | activations, expert dequant scratch |

`DEEPREPO_DISK_CACHE=<dir>` adds the job-local SSD page cache (default in the runner
workflow: `$RUNNER_TEMP/deeprepo-pages`, 6 GiB). It is read-through, page-aligned, and
never stores deltas — only immutable bin pages, keyed by asset name. It dies with the VM.

`--strict` disables the f16 cache and computes everything from freshly paged f32 dequant
(slower, strongest proof, lower RAM). The f16 head is skipped in strict mode.

## Budget ledger (defaults, per UTC day, shared by all writers)

| kind | cap | rationale |
|---|---|---|
| down | 96 GiB | paged reads from release CDN |
| up | 12 GiB | journal deltas + compaction uploads |
| api | 288k calls | well under 5k/h PAT core limit; token buckets in `meter.py` |
| tokens | 24k generated | compute guard; the real limit is wall-clock |

The ledger is persisted in registers (`budget/today`); every writer adopts it. Caps fail
closed with `BudgetExceeded`, never silently overrun. Caps can be lowered by writing
`budget/caps` to the registers.

## Accounting examples

- One cold decode token (8/32 experts): ~8 x 48 x 2.65 MB ≈ **1.0 GiB** of expert bytes
  worst case; page-cache hits make 300–600 MB typical.
- 2k-token prefill: union of experts over the chunk per layer ≈ **16–32 GiB** streamed
  once, i.e. a large share of one day's down budget. This is why prompts are short and
  the generator is budgeted.
- Import: 18.6 GiB read from HF + ~19 GiB uploaded once, outside the daily runtime caps
  (maintenance window), then ~zero for the life of the model.
