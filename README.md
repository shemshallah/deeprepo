# deeprepo — a 30B MoE running on repo-backed sparse RAM/VRAM

deeprepo runs **Qwen3-30B-A3B-Instruct-2507 (Q4_K_M, ~18.6 GiB)** on a ring of
GitHub-hosted runners that individually cannot hold it: a standard runner has **7 GiB RAM**
and **14 GB disk**. The model deliberately does not fit. Its weights live in **repovram**
and its KV cache in **reporam** — two sparse virtual memory tiers backed by GitHub release
bins, read and written page-by-page (4 MiB) over HTTP Range.

```
repovram  16 bins x 1.99 GiB = 31.9 GiB sparse virtual   weights (immutable after import)
reporam   16 bins x 1.99 GiB = 31.9 GiB sparse virtual   KV + registers (journal + gens)
L0        bounded runner caches (~1.5 GiB pages + ~1.9 GiB f16 tensors)
L0.5      job-local SSD page cache (6 GiB, in-place files, dies with the VM)
```

Unwritten pages are implicit zeros. The 18.6 GiB model materializes **10 of 16** vram bins;
a KV session materializes a few MB. Nothing is ever cloned or downloaded to your phone:
imports and inference run entirely on GitHub runners.

## The ring

| repo | role |
|---|---|
| `deeprepo` | coordinator: chat intake, tokenizer, ring leases, streaming, workflows |
| `deeprepo-cpu-1..5` | disposable CPU slots: prewarm/hydrate → decode shift → handoff |
| `reporam` | sparse RAM tier: KV rows, hierarchical registers, jobs, budgets |
| `repovram` | sparse VRAM tier: tiled GGUF weights |

A chat turn is tokenized in `deeprepo`, written as a job spec into reporam registers, and
handed to the **next prewarmed** slot under a lease. The slot prefills in chunks (KV written
to reporam), decodes up to 32 tokens / 20 minutes, publishes token deltas, writes a handoff
packet, and frees the lease. The next slot is hydrated **before** it is granted the lease
and continues from the KV already in reporam. The coordinator tails the journal and streams
the reply to the issue.

## Where the memory claim is proven

`tests/` runs offline, with a tiny random Qwen3-MoE GGUF and file-backed tiers that use the
production pager:

- **dequant is byte-exact vs ggml**: scalar ports of `dequantize_row_q{4,5,6}_K/q8_0` from
  `ggml-quants.c` compared bit-for-bit (NaN-safe).
- **engine == dense reference**: logits match a fully in-memory implementation, with every
  weight read through sparse repovram bins and every KV row through reporam (strict f32
  path: atol 1e-4; fp16-KV decode path: argmax-identical).
- **handoff is exact**: generation split across two writer instances (simulated ring
  rotation) produces the same tokens as a continuous run.
- **compaction is safe**: deltas fold into new trimmed generation bins + manifest, old bins
  deleted, fresh readers rehydrate the same KV.
- **sparsity is real**: the tiny model materializes 1 of 16 vram bins; untouched pages and
  absent bins read as zeros.
- **import is safe + resumable**: the production importer tiles a streamed GGUF, preserves
  the bootstrap superblock on a fresh runner, and re-running skips completed bins.

```bash
pip install -r requirements-core.txt
python tests/run_all.py
```

## Quick start

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)**. In short:

1. create the 8 public repos (skeletons are stamped as siblings: `deeprepo-cpu-1..5`,
   `reporam`, `repovram`) and push them;
2. add secret **`DEEPREPO_PAT`** to `deeprepo` and all five cpu repos;
3. run workflows in order: **02 test → 00 bootstrap → 01 setup ring → 03 load model →
   04 boot single → 05 serve**;
4. label an issue **`deeprepo-chat`** and talk to the model.

## Optional middleman (hosted control plane / hot tier)

Hosted runners cannot accept inbound connections, so if you want low-latency mutable state
shared between runners — or a 24/7 orchestrator that survives GitHub disabling schedules —
run the optional middleman on any host (Koyeb/Fly/Render/VPS) or use R2 behind a Worker:

```bash
python -m deepcore.middle serve --root /data --port 8080   # token auth, ranged read/write
python -m deepcore keeper --loop --interval 600            # PAT control plane
```

Set `DEEPREPO_RAM_BACKEND=middle` to move only the mutable RAM tier there, keep the weights
on the free release CDN, and read [docs/MIDDLEMAN.md](docs/MIDDLEMAN.md) for the hosting
matrix, token scopes, and the weight/release lifecycle (never rewrite weights on handoff).

## Docs

- [Architecture](docs/ARCHITECTURE.md) — tiers, journal/generation writes, data flow.
- [Memory map](docs/MEMORY_MAP.md) — exact geometry, layouts, KV arithmetic, budgets.
- [Model plan](docs/MODEL.md) — why this Qwen, tensor map, quant support, expectations.
- [Runbook](docs/RUNBOOK.md) — setup, secrets, operations, troubleshooting.
- [ToS safety](docs/TOS_SAFETY.md) — budgets, concurrency, retention, what we avoid.
- [Middleman](docs/MIDDLEMAN.md) — optional hosted control plane + mutable hot tier.
- [Performance](docs/PERFORMANCE.md) — measured prefill/decode, tok/s, and why it is slow.

## Honest performance

This is out-of-core inference over a CDN, not local inference. Rough expectations on a
2-core hosted runner: prefill 2k tokens in 4–10 min, 3–12 s per decoded token depending on
expert-cache luck, ~32 tokens per 20-minute shift. Quality is the real Qwen3-30B-A3B's;
throughput is the price of running it on memory that doesn't exist locally.

Default daily budget (shared by all writers, persisted in registers): 96 GiB down,
12 GiB up, 288k API calls, 24k tokens. Caps fail closed.

## Persistent local tiers

On a machine that persists between runs (self-hosted runner, workstation) the bins can live
as real files inside the repo tree, modified in place — `DEEPREPO_TIER_DIR=./tiers` switches
both tiers to local files, no GitHub needed for the memory:

```bash
export DEEPREPO_TIER_DIR="$PWD/tiers"
python -m deepcore bootstrap
python -m deepcore load-model --url <gguf-url>
python -m deepcore local --prompt "hello"
```

Hosted runners stay on release tiers because their disks are wiped after every job; see
[docs/RUNBOOK.md](docs/RUNBOOK.md#6-persistent-local-tiers-self-hosted--single-machine).

## Legacy

The earlier RAM16/Backblaze-B2 experiment (`ram16.py`, `ram16_b2.py`, `native/`,
`.github/workflows/ram16-b2.yml`) is kept intact for reference and is unrelated to the
ring above. The new system reuses ram16's ranged-asset pager, delta journal, compaction
primitives, RegionLog, and issue queue.
