# ToS and abuse safety

The system is designed to stay a normal, documented use of GitHub: public repos, standard
hosted runners, release assets served by GitHub's own CDN, and hard budgets. It is not a
way to hide unrelated compute or to move bulk data through git history.

## What we deliberately do

- **Public repositories + standard `ubuntu-latest` runners.** Linux runner minutes are
  free for public repositories; that is the documented program, not a loophole.
- **Release assets for large binaries.** GitHub provides releases (2 GiB/file cap) and CDN
  downloads for exactly this kind of artifact. We never put model bytes in git.
- **Append-only writes + generation GC.** No release asset is modified in place; old bins,
  manifests, and consumed deltas are deleted after compaction, so visible storage tracks
  live data instead of growing forever.
- **Hard daily budgets** (defaults: 96 GiB down, 12 GiB up, 288k API calls, 24k tokens),
  persisted in registers so every runner and the coordinator share one ledger. Caps fail
  closed.
- **Bounded concurrency**: one shift per slot (`concurrency: runner`), one serve window
  (`concurrency: serve`), one bootstrap (`concurrency: bootstrap`).
- **Bounded job length**: runner jobs ≤ 90 min, serve windows ≤ 350 min, keeper ≤ 20 min.
  No job sits idle for six hours.
- **No polling storms**: runners are dispatch-driven (no cron); the keeper runs every
  20 min with a handful of API calls; the coordinator tails one assets listing per ~6 s
  during a shift.
- **Coalesced reads**: 4 MiB pages, contiguous runs merged into single Range requests,
  bounded LRU + request coalescing in `ram16` — we minimize requests per byte.
- **Least-privilege PAT** kept only in Actions secrets; no token ever appears in logs,
  code, or registers.
- **Model licensing respected**: Qwen3 is Apache-2.0; the model is downloaded once from the
  official Hugging Face repo.

## What we avoid

- No scraping, crawling, or mass API enumeration.
- No attempt to evade rate limits: token buckets + `Retry-After`/`x-ratelimit-*` handling
  and exponential backoff in `meter.py`/`ram16.Asset`.
- No private-repo abuse: cross-repo access uses the owner's PAT with explicit repo list.
- No crypto mining, no unrelated batch compute; all cycles are model inference documented
  in this repo.
- No method to bypass GitHub billing: private repos would bill minutes; we recommend public
  repos and the workflows use standard runners only.
- No "infinite" retention: the keeper/compaction deletes superseded generations.

## Guardrails you can tighten

| knob | default | effect |
|---|---|---|
| `budget/caps` register | 96/12 GiB, 288k api, 24k tokens | lower for a quieter footprint |
| `inputs.minutes` on 05-serve | 300 | shorter windows |
| `loop` input on 05-serve | true | false = one window per manual dispatch |
| runner `--shift-tokens` / `--shift-minutes` | 32 / 20 | shorter shifts, more handoffs |
| keeper cron | `*/20` | slow it down or disable when idle |
| `--strict` | off | fewer cached copies in runner RAM |

## Honest limits

- Still a lot of network traffic: a cold decode token can move ~1 GiB from the release CDN.
  The budget is the enforcement mechanism. Expect a few prompts per day at default caps.
- GitHub Actions is not a realtime inference platform; shifts and handoffs exist partly to
  keep jobs short and well-behaved.
- If GitHub changes release/CDN policies, the tier design must adapt — the `ram16` store
  abstraction keeps that change contained (`sparse` bins ↔ any ranged object store).
