# Architecture

deeprepo runs one large MoE model across GitHub-hosted runners that individually **cannot
hold it**: Qwen3-30B-A3B Q4_K_M is ~18.6 GiB; a standard runner has 7 GiB RAM and 14 GB
disk. The model does not fit anywhere, by design. It lives in two sparse, repo-backed
memory tiers and is paged on demand.

```
                       ┌─────────────────────────────────────────────┐
   issue comments ───▶ │ deeprepo coordinator (hosted runner)        │
   (deeprepo-chat)     │  tokenizer · job specs · ring · streaming  │
                       └───────┬───────────────────────┬─────────────┘
                               │ workflow_dispatch(PAT)│ journal tail (HTTP Range)
                 ┌─────────────▼──────────┐   ┌────────▼─────────────────────┐
                 │ deeprepo-cpu-1..5      │   │ reporam  (32 GiB virtual)    │
                 │  prewarm · run · drain │◀─▶│  KV cache · registers · jobs │
                 │  bounded page cache    │   │  page-delta journal + gens   │
                 └─────────────┬──────────┘   └──────────────────────────────┘
                               │ paged weight reads (HTTP Range)
                 ┌─────────────▼──────────┐
                 │ repovram (32 GiB virt) │
                 │  GGUF weights, ~18.6GB │
                 │  16 x 1.99 GiB bins    │
                 └────────────────────────┘
```

## Tiers

| tier | medium | contents | mutability |
|---|---|---|---|
| L0 physram | runner RAM | bounded page LRU + f16 dequant cache | ephemeral, wiped with the VM |
| L1 reporam | release bins + journal on `reporam` | KV rows, hierarchical registers, budgets, job/token records | append-only deltas, generation compaction |
| L2 repovram | release bins on `repovram` | GGUF weights, tiled page-by-page | immutable after import |
| L0.5 SSD | runner SSD (`$RUNNER_TEMP`) | read-through page cache (6 GiB, LRU) + at most one <=2 GiB import bin | ephemeral, created by the runner |

Per tier: **16 bins x 510 pages x 4 MiB ≈ 31.9 GiB virtual**, bin header 4 MiB, each bin
asset 2044 MiB (< GitHub's 2 GiB asset cap). Unwritten pages and absent bins read as zeros:
visible bytes equal written bytes, never 32 GiB.

## Why releases, not repo-root blocks

- GitHub caps a single file in git at 100 MB and flags repos that approach a few GB.
  Git also stores a full blob per changed block per commit: rewriting a 1.99 GB register
  on every KV flush would balloon history quadratically.
- Release assets allow 2 GiB per file, are served by CDN with HTTP Range, and are free of
  git-history cost. They are immutable, which the write path handles with a **journal +
  generation rotation**:
  - small writes: compressed `d-<seq>.dlt` page deltas, each a new asset;
  - compaction: fold deltas into a fresh `mem-XX.gN.bin`, publish `m-<gen>.json`, delete
    old bins and consumed deltas. No asset is ever modified in place.
- repovram never changes after import, so immutability costs nothing there.

## Components (`deepcore/`)

| module | role |
|---|---|
| `geom.py` | tier geometry, DPRSB1 superblock, ram16 slot headers, layouts |
| `store.py` | `SparseAsset` (zeros for absent/trimmed bins), `TierStore`, HF ranged reader |
| `tiers.py` | bootstrap, resumable GGUF import (streamed from HF), trimmed compaction, registers |
| `gguf.py` | paged GGUF parse, numpy dequant (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K), writer for tests |
| `engine.py` | Qwen3-MoE numpy engine: GQA+NEOX rope+QK-norm, top-k renorm MoE, KV through reporam |
| `tok.py` | byte-level BPE straight from GGUF vocab, ChatML |
| `ring.py` | lease ring, handoff packets, dispatch/await helpers |
| `worker.py` | runner side: `prewarm`, `run` (shift), `selftest` |
| `coordinator.py` | chat intake, prewarm-before-handoff pipeline, streaming, budget guard |
| `cli.py`, `runtime.py`, `meter.py` | operator entry points, wiring, budget ledger |

Reused unchanged from the earlier RAM16 work: `ram16.py` supplies the ranged-asset
abstraction, the byte-addressable register over 16 slots, the delta journal, generation
compaction primitives, RegionLog, and the GitHub-issue chat queue.

## A chat turn

1. **Intake**: coordinator (05-serve) sees a new comment on a `deeprepo-chat` issue.
2. **Tokenize**: ChatML prompt rendered with the vocab stored in the GGUF header.
3. **Job spec** → registers (`job/<id>`), committed as a journal delta in reporam.
4. **Prewarm** next ring slot: dispatch `runner.yml` op=`prewarm`; the runner verifies both
   tiers, checks the model record, warms the first layers into its page cache, and reports
   success. **No work is handed off before this succeeds.**
5. **Lease**: coordinator grants `ring/lease` (epoch, holder, expiry) in registers.
6. **Run**: the runner asserts the lease, prefills the prompt in chunks (KV written to
   reporam), decodes up to 32 tokens or 20 minutes, publishes token chunks as journal
   deltas, writes a handoff packet, and frees the lease.
7. **Stream**: coordinator tails deltas every 6 s and updates the issue comment.
8. **Handoff**: on shift end the next slot is prewarmed, granted the lease, and continues
   from the KV state already in reporam — reproduction is exact (tested).
9. **Close**: EOS or max tokens → final comment, transcript record, budget persisted.

## Proof this is really using repo memory

- Weights: every model byte enters a runner through `SparseAsset.read` over repovram bins;
  only a bounded page cache + f16 cache is local. The tiny-model test asserts logits equal
  a dense reference through this exact path (`tests/test_engine.py`).
- KV: written to reporam pages, flushed as journal deltas, read back through the same
  pager; a fresh process continues generation from those pages (`test_handoff_and_compaction`).
- Sparsity: the tiny model materializes **1 of 16** vram bins (`test_sparse_bins_are_absent`).
- Compaction: trimmed bins + manifest rehydration verified (`test_handoff_and_compaction`).

## What this is not

- Not realtime: one decode token costs a network+paging+compute path measured in seconds.
- Not a replacement for local inference; it is a provable out-of-core, repo-backed memory
  system with a real frontier-family MoE on top.
- Not a Git clone of the model: nothing is ever checked out into a repo working tree.
