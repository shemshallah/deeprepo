# deeprepo TODO / scratchcard (session 6 update on top)

## SESSION 6 STATE — repo-backed sparse RAM/VRAM ring (built, offline-verified)
- Direction changed from B2 to GitHub release tiers per user's session-6 call; the B2 code
  stays as legacy (`ram16_b2.py`, `native/`, `ram16-b2.yml`), untouched.
- **Built:** `deepcore/` (geom, store, tiers, gguf+dequant, engine, tok, ring, worker,
  coordinator, meter, runtime, cli), 7 deeprepo workflows (00 bootstrap, 01 setup-ring,
  02 test, 03 load-model, 04 boot-single, 05 serve, keeper), runner template + stamped
  `deeprepo-cpu-1..5`, `reporam`, `repovram` skeletons, docs.
- **Model:** Qwen3-30B-A3B-Instruct-2507 Q4_K_M (~18.6 GiB) — cannot fit in 7 GiB RAM /
  14 GB disk; MoE top-8 makes sparse paging meaningful.
- **Geometry:** 16 bins x 510 pages x 4 MiB = 31.875 GiB virtual per tier (bin asset
  2044 MiB < 2 GiB cap). Only written bins/pages exist; absent bins read as zeros.
- **Writes:** reporam is append-only journal deltas (`d-<seq>.dlt`) + generation compaction
  (`mem-XX.gN.bin`, manifest, delete old). No asset modified in place.
- **KV:** lives in reporam (f16 rows), addressed per layer/position; handoff reuses it.
- **Ring:** lease in registers; coordinator prewarms the NEXT slot before granting a shift;
  worker drains → handoff packet → lease free.
- **Verified offline (`python tests/run_all.py`, 4/4 files):**
  - dequant byte-exact vs scalar ports of ggml `dequantize_row_q{4,5,6}_K/q8_0`;
  - tiny Qwen3-MoE engine logits == dense reference through the real pager (strict);
  - KV visible to a fresh reader; untouched KV pages are zeros;
  - handoff across writer instances reproduces the continuous run token-for-token;
  - trimmed compaction + manifest rehydrate; 1/16 vram bins materialized;
  - production import tiles a streamed GGUF, preserves the bin-0 superblock on a fresh
    runner, and resume skips completed bins.
- **Fixed during build:** GGUF v3 u64 counts, writer KV/info padding, Q5_K 176 B blocks +
  per-half high bit, Q6_K per-l scale indexing, tensor addr rebase onto wbase, manifest
  adoption via store (not explicit slots), compaction header/superblock preservation,
  import bin-0 superblock preservation, f16 output head streamed in chunks.
- **Not yet run live:** needs the 8 repos + `DEEPREPO_PAT` + model import on real runners.
  Expect the first failures around PAT scopes, release upload limits, or hosted-runner
  timing — 02/01 are the fast diagnostics.

## NEXT STEPS
1. Push the stamped repos + deeprepo, add `DEEPREPO_PAT` (deeprepo + 5 cpu repos).
2. 00 bootstrap -> 01 setup-ring(selftest) -> 03 load-model (resume if timed out).
3. 04 boot-single: expect 5-15 min prefill for a short prompt; check `STATS` output.
4. 05 serve with a `deeprepo-chat` issue; watch lease/handoff in the coordinator logs.
5. Tune: `--cache-pages`, f16 cache, shift tokens, day caps. Consider `--strict` boot.
6. Optional follow-ups: split-file GGUF support, speculative batch across 5 slots, per-runner
   journal tags (drop the single-writer lease), SSD hydration cache for self-hosted runners.
# deeprepo TODO / scratchcard (session 5 update on top)

## SESSION 5 HONEST STATE
- Code now exists. Stopped by user mid-debug. Tested only in the sandbox (1 vCPU, root, no HF, no GitHub write token).
- DONE + TESTED (python3 -m unittest tests.test_rr, 5 tests pass):
  - rr/rrbin.py: RRBIN v1 segment format (raw/zlib/zero codecs, per-1MiB sha256 in header, header checksum), corruption detection.
  - rr/manifest.py: manifest with placement + copy-on-write generations + sha pin.
  - rr/pack.py: stream -> bins, striped across repos (bin i -> repos[i % R]), resumable skip of existing bins.
  - rr/reader.py: verified ranged reader (fetches 1 MiB sub-blocks, header pinned to manifest hash). Test shows ranged reads fetch far less than total.
  - rr/meter.py: token buckets + Retry-After / x-ratelimit handling (unit-tested lightly).
  - rr/store.py: DirStore (tested). GitHubReleaseStore is WRITTEN BUT UNTESTED (needs a real token/runner).
- BUILT, PARTLY WORKING: native/rrvm/rrvm.c + rrvm.h + test_rrvm.c (build: gcc -O2 -pthread -o test_rrvm test_rrvm.c rrvm.c; needs root or vm.unprivileged_userfaultfd=1).
  - Works in sandbox: userfaultfd MISSING + write-protect dirty tracking, 4 worker threads, strict budget (16 MiB cap on a 192 MiB x 2 regions): weights random reads content-verified (14.6k faults, peak 16 MiB, over_budget_fills=0); state region writes from 4 threads with zero-fill on first touch, ~11k WP faults, ~11k dirty writebacks, data verified after eviction round trips (bad=0), stalls=0.
  - KNOWN FAILURE: final check FAILED. After rrvm_drain() resident_bytes was 1 MiB (4 pages) instead of 0. Likely 4 = nworkers: a leaked `used` count in some path (suspects: handle_missing early-return path after reserve, EEXIST path, or drain racing a worker still handling a stale fault). Everything else in the test passed. Not yet debugged.
  - Eviction is FIFO, not CLOCK (userfaultfd cannot see reads of resident pages). Say FIFO in README.
  - The test's printed rss=210 MiB is a test artifact (the calloc'd 192 MiB backing store), not rrvm. Needs a separate RSS measurement that excludes the store (e.g. cgroup or file-backed store).
- NOT STARTED: rrfetchd.py + unix socket protocol + C socket backend; rrshim.c (weights path + allocator path for KV); stub GGUF generator; llama.cpp build/test; cgroup launcher; all workflows (bootstrap, ring-setup, weights-import, boot-single, ring-start, weights-update); lease CAS; handoff packet; issue chat; stats; README with measured numbers. The older ram16*.py / B2 files in the zip are from an earlier design and are unchanged.
- NOT CHECKED: GitHub terms. You suggested I skim them; I did not get to it. The ToS/abuse-limit concern from session 4 still stands and is unresolved.

## NEXT STEPS (in order)
1. Fix the rrvm_drain leak; add the check to test_rrvm.c as an assertion; add a file-backed store so RSS can be measured honestly.
2. rrfetchd.py (threaded, unix socket, uses rr.reader.Reader, bounded disk cache, overlay writes) + C socket backend + integration test against DirStore.
3. rrshim.c: weights (stub GGUF fd -> rrvm WEIGHTS region), allocator size-window for KV -> rrvm STATE region; test with a tiny generated GGUF and pinned llama.cpp built from GitHub.
4. Workflows in the order listed in session 4; boot-single stability gate before ring-start.
5. On a real runner: verify uffd (sudo), uffd-wp, cgroup v2, HEAD sizes for Qwen3-235B-A22B vs Qwen3-30B-A3B GGUF, Range throughput.

---
# deeprepo TODO / scratchcard (session 4 update on top)

## SESSION 4 HONEST STATE
- STILL NO PROJECT CODE WRITTEN. This session only ran sandbox probes and rewrote this file. Nothing below has been built or tested except the probe results listed here.
- Sandbox probe (not a runner): 1 vCPU, ~4 GB RAM, kernel 6.18, gcc 13.3, python 3.12, /dev/fuse present, running as root.
  - userfaultfd works AND reports UFFD_FEATURE_PAGEFAULT_FLAG_WP, so write-protect dirty tracking is available here. vm.unprivileged_userfaultfd=0, so on a runner the launcher must use sudo (or `sysctl vm.unprivileged_userfaultfd=1`) before starting. Verify on a real runner in Phase 0.
  - Network from sandbox: github.com reachable (llama.cpp can be cloned and built); Hugging Face is NOT in the allow list, so the model import and Qwen tests cannot be run here. Model work happens on a runner via the import workflow.
  - No libcurl headers: the C side will not speak HTTP.

## SESSION 4 DIRECTIVES (user, verbatim intent)
- Optimise for most impressive AND working properly; go all out on code, less planning.
- KV cache MUST live in repoVRAM space, with a custom, working allocator shim.
- One repo zip creates the `reporam` repo; `repovram` is created and its blocks formatted properly.
- Runner populates the model weights AFTER deeprepo-cpu-1..5 are set up.
- Deploy from a single runner first; start the full ring only once that runner is stable. This is its own workflow, in addition to the model setup and boot workflows.
- Biggest Qwen we can fit. Bin count stays a parameter; pick the model by measured GGUF size vs. repo capacity.
- The "bounded window, not literal zero" reading of the RAM rule is treated as accepted (user proceeded without objecting). Strict mode + cgroup cap + boot check stay as designed.

## SESSION 4 DESIGN DECISIONS (new)
- Fetching: C core never does HTTP. rrvm talks over a unix socket to `rrfetchd.py` (threaded Python daemon) that owns the Meter, signed-URL cache, Range reads, local disk cache cap, and overlay writes.
- Verification: each 64 MiB segment header holds a sha256 per 1 MiB sub-block (64 x 32 B = 2 KiB, fits the 4 KiB header), so a ranged 1 MiB fetch is verified without downloading the segment. Header hash is pinned in the manifest.
- rrvm page = 256 KiB (RR_PAGE_KB), resolved with one UFFDIO_COPY; fetch unit = 1 MiB sub-block.
- State space (KV): MISSING + WP. Never-written pages are zero-filled with no fetch. Dirty pages are written back on eviction as delta journal entries (epoch-stamped, zstd or zlib), uploaded as repoVRAM release assets in batches. Clean pages are dropped without I/O.
- Allocator shim (LD_PRELOAD): intercepts posix_memalign/aligned_alloc/malloc/calloc/realloc/free for sizes in [RR_STATE_MIN, RR_STATE_MAX] and anonymous mmap above RR_STATE_MIN, routing them to rrvm state regions; also intercepts open/mmap/pread on the stub GGUF fd for weights. Risk: it cannot tell KV from other big buffers by itself, so the size window is a knob and the shim logs every routed allocation. Must be checked against the pinned llama.cpp (ggml_aligned_malloc path).
- Strict budget: stall instead of over-budget fill; leases capped ~50% of window; per-thread reserve; cgroup v2 memory.max as backstop; boot check fails on resident_bytes > cap or over_budget_fills > 0.

## SESSION 4 WORKFLOWS (planned, in the main repo zip)
1. `bootstrap` : create `reporam` and `repovram` repos, create releases, format empty bins (RRBIN v1; repoVRAM bins ~4 KiB each when empty), write manifest + ring.json.
2. `ring-setup` : create/verify deeprepo-cpu-1..5 from the same zip, set RING_SLOT variables and DEEPREPO_PAT secret, smoke-test each (checkout, sudo, uffd, cgroup v2).
3. `weights-import` : runs on a runner AFTER ring-setup; streams the chosen Qwen GGUF from Hugging Face, cuts into 64 MiB segments, packs into bins, uploads to reporam/repovram with striping, resumable, then publishes manifest generation 1.
4. `boot-single` : one runner boots the model (shim + rrvm + cgroup), serves issue chat, writes stats. Must pass the stability gate: N minutes, zero over-budget fills, KV round-trip through repoVRAM, no rate-limit trips.
5. `ring-start` : only runs if the stability gate marker is present; starts the lease ring (WARM depth, handoff packet, CAS lease on ring-state ref).
6. `weights-update` : copy-on-write generations as described in session 3.

## SESSION 4 MODEL TARGET
- Biggest Qwen we can: candidates Qwen3-235B-A22B (MoE; GGUF size at Q2/Q3/Q4 must be measured, expect ~85-140 GB, i.e. roughly 40-70 bins) then Qwen3-30B-A3B as the fallback that definitely fits. Decide after a HEAD request for exact GGUF sizes on a runner. Per-token expert traffic for 22B active params is large; expect low tok/s and say so in measured stats.
- Egress/ToS caution (restating, not new): re-pulling tens of GB per ring lap, and running a perpetual inference ring on Actions, may violate GitHub terms or trip abuse limits. Use long active windows and demand paging; keep RING_ACTIVE_MIN high for big models; your call on risk.

## SESSION 4 BUILD ORDER (code starts after this zip)
1. `rr/` Python: rrbin formatter/verifier, manifest + generations, Meter, DirStore + GitHub release store, tests.
2. `native/rrvm/`: rrvm.c (MISSING, strict budget, WP, CLOCK eviction) + stress tests with budget invariants.
3. `rrfetchd.py` + unix-socket protocol + integration test against DirStore.
4. `rrshim.c` weights path + allocator path; stub GGUF generator; build pinned llama.cpp in sandbox and test with a tiny generated GGUF (no HF access here).
5. Workflows above, lease CAS, handoff packet, issue chat server, stats.
6. README with only measured numbers.

---
# deeprepo TODO / scratchcard (session 3 update on top)

## SESSION 3 HONEST STATE
- STILL NO CODE BUILT OR TESTED. Only this file changed. Everything below is design; numbers are projections until Phase 0 measures a real runner.
- I read "deepram" as RepoRAM. Tell me if you meant something else.

## SESSION 3 ANSWERS (and what they supersede)

### Q1. Can runners avoid ever downloading the whole model after initial setup?
- Yes, with demand paging over sharded weights, with one caveat. Runners are fresh VMs every job, so nothing persists on them. "Initial setup" is the one-time format/upload of the model into the repos (resumable, from HF, done once). After that a runner fetches only the segments it touches: for a MoE model that is the hot expert set (warmed from the predecessor's heat map), not the whole model. A dense model touches every weight each token, so it would effectively stream the whole model per token; dense is the wrong demo.
- SUPERSEDES "hydrate whole model to local disk at boot". The local disk becomes a bounded cache (RR_DISK_BUDGET_GB) of hot segments, never a full copy. Boot fetches the heat-ordered hot set only, in 16-64 MiB ranges.
- Sharding: weights are cut by (layer, expert) byte ranges into 64 MiB segments and striped across bins, so concurrent ranged reads hit different assets. Optional later: erasure coding so one slow shard never stalls a fetch.

### Q2. Can weights be spread over the reporam AND repovram repos?
- Yes. Make the segment the only unit and let placement be policy, not tier identity. The manifest maps every virtual segment to a location {repo, release, asset, slot}. RepoRAM repo = immutable base bins. repoVRAM repo = mutable overlay (KV, state, weight updates) AND it may also hold base weight bins as spill/stripe capacity.
- Striping policy: bin i -> repo (i mod R). Benefit is parallel fetch and failure isolation, NOT extra API quota (primary limits are per token; public downloads are not API-metered anyway).
- Capacity at 4+4 bins: ~16 GB real across both repos. With the runner window hard-capped well below that, a ~14-15 GB model cannot be resident in runner RAM, so even 4+4 supports a real "lives only in the repos" demo. Candidate: a Q3-class Qwen3-30B-A3B (verify the actual GGUF size first, it may exceed 15 GB). Larger demos (Qwen3-30B-A3B Q4 ~18-19 GB, gpt-oss-120b ~60-65 GB) need ~10 or ~33 bins; bin count stays a parameter.

### Q3. Updatable weights
- Release assets are immutable, and a 64 MiB change must not force re-uploading a 2 GB bin. So two object classes:
  1. Base bins: packed 31-segment RRBIN v1 bins, immutable, content-hashed.
  2. Overlay segments: individual assets named by content hash, one per changed 64 MiB segment (plus 256 KiB-page deltas in repoVRAM for fine-grained patches). Compaction repacks overlays into new base bins occasionally.
- Generations: each update writes new segments, then a new manifest generation that references unchanged segments by hash (copy-on-write), then advances the manifest pointer in the lease ref with the same git-ref CAS used for the lease. Unchanged segments are never re-uploaded or re-downloaded.
- Update kinds: full model swap, expert/layer replacement, LoRA kept as a separate small adapter (base stays immutable), merged weights as overlay segments.
- Adoption: runners adopt generation N+1 only at handoff or at a token boundary, never mid-forward-pass. Predecessor stays on N until it exits; the successor warms on N+1. Epoch fencing already rejects stale writers. A `weights-update` workflow in the main repo does the upload and pointer advance, metered like everything else. GC keeps the last N generations.

### Q4. "Model RAM/VRAM absolutely forbidden from using runner RAM"
- Literal zero is impossible: the CPU can only compute on bytes that are in runner RAM (or CPU cache). A remote byte has to pass through RAM to be used. What CAN be guaranteed, and will be enforced and tested:
  1. Hard window cap: weights and KV state may occupy at most RR_RAM_BUDGET_MB of runner RAM as transient resident pages. Never the whole model, never the whole KV.
  2. Strict mode (new, supersedes the soft budget): no over-budget fills. If no victim page exists, the faulting thread stalls until one frees. To avoid deadlock, leases are capped at ~50% of the window and each compute thread has a reserved minimum.
  3. cgroup v2 memory.max (via sudo systemd-run or a cgroup written directly) = non-model budget (OS ~2 GB + scratch/activations ~2 GB) + window. The kernel kills/throttles anything that exceeds it. rr_stats exports resident_bytes and the CI/boot check fails if resident_bytes > budget or over_budget_fills > 0.
  4. No full-model page cache or disk copy (O_DIRECT / fadvise DONTNEED, bounded disk cache). No mlock. No weight repacking that copies tensors into heap.
  5. KV/state: llama.cpp allocates KV in ordinary heap buffers. To put it in the repoVRAM space the shim must also route large ggml CPU buffer allocations into an rrvm space (intercept the big aligned allocations), or patch the ggml buffer allocator. Riskier than the weights shim; verify against the pinned llama.cpp source. Fallback: KV stays in the heap but inside the cgroup cap, with a hard context limit.
  6. Activations, scratch and the window are runner RAM by design (CPU only), per your earlier rule.
- If you want compute where the bytes live instead (no remote bytes crossing into one runner's RAM), the alternative is pipeline/expert-parallel inference across the ring: each runner holds its own shard in its own RAM. That uses runner RAM, just spread over runners, and needs a relay for runner-to-runner traffic. Parked as a stretch, not the plan.

## Updated architecture notes (supersede earlier wording where they conflict)
- Address map: one virtual map, spaces = weights (read-mostly, base + overlay) and state (read/write). Tier names are roles; placement is the manifest.
- Weights are read through rrvm demand paging; cold tier = Releases via public URL + Range; local disk = bounded cache; runner RAM = hard-capped window.
- Handoff packet (unchanged, small): epoch, manifest generation, KV journal pointer, page heat map, router transition table, session cursor.
- Metering, lease CAS, ring lifecycle, bin format (RRBIN v1) from the session 2 section below still apply.

## Updated build order
0. Phase 0 runner measurements (fio, memory bandwidth, Range throughput to Releases and HF, df/free/nproc, uffd + uffd-wp, cgroup v2 availability and memory.max enforcement, signed-URL expiry, 304 accounting).
1. rrbin formatter + verifier, manifest with placement + generations, Meter, ring_store.py with overlay-aware reads, DirStore tests. `hydrate.py` becomes `warm.py` (heat-ordered hot-set fetch with disk budget), not a full download.
2. rrvm core in C: MISSING mode, strict budget mode, then WP; stress tests including budget-invariant checks.
3. Shim + stub GGUF + cgroup launcher + Qwen3-30B-A3B end to end.
4. repoVRAM: KV space, delta journal, lease CAS, weight-update workflow (CoW generations), ring workflow, handoff drill.
5. Bigger model run, issue stats, README with measured numbers.
6. Stretch: peer handoff, zstd pages, erasure coding, zero-copy minor faults, own decode loop, pipeline-parallel ring.

## Open questions
- Confirm the "bounded transient window, not literal zero" reading of rule 4.
- RING_WARM_DEPTH (2 vs 4) and RING_ACTIVE_MIN (66).
- Demo model for 4+4 bins: a Q3-class 30B-A3B if it fits, otherwise scale bins to ~10.
- Weights origin: one-time import from Hugging Face into the repos, then repos only.

---
# Session 2 section and earlier (kept as written; session 3 above supersedes where they conflict)

# deeprepo TODO / scratchcard (session 2 update on top)

## SESSION 2 HONEST STATE
- STILL NO CODE BUILT OR TESTED. Only this file changed in this zip. The deploy zips, ring_store.py, hydrate.py, rrvm and the bin formatter do not exist yet.
- Sandbox probe (not a runner): kernel 6.18, 1 CPU, 3 GB RAM, userfaultfd syscall returned a working fd (with the USER_MODE_ONLY flag, which does NOT cover kernel-mode access such as read() into the region), /dev/fuse present, gcc 13.3, no libzstd/libcurl headers. uffd-wp, eviction and handler races were NOT exercised.
- All numbers below are projections until Phase 0 measures a real runner.

## SESSION 2 DECISIONS AND DESIGN DELTAS

### Memory roles (your directive)
- Runner RAM = CPU only: activations, scratch, a bounded hot window. Every byte the CPU reads must still pass through runner RAM, so RepoRAM acts as a capacity tier behind a budgeted window, not as RAM the CPU executes from.
- Avoid double caching: backend reads use O_DIRECT (or posix_fadvise DONTNEED) so the kernel page cache does not duplicate rrvm resident pages. Mount/hydrate dirs with that in mind. Cap the window by env (RR_RAM_BUDGET_MB).
- RepoRAM (node 0, read-mostly, weights) and repoVRAM (node 1, read/write, KV/state) are two rrvm spaces in one address map. Compute only ever sees plain pointers.

### Bin format RRBIN v1 (both tiers)
- One bin = 31 segments. Segment = 4 KiB header + 64 MiB payload. Bin size = 31 x (64 MiB + 4 KiB) = 2,080,501,760 bytes, under the 2 GiB release-asset cap with ~63 MiB headroom. Names: rrram-0..3.bin, rrvram-0..3.bin.
- Segment header (self-describing, so a ranged fetch verifies without the manifest): magic RRSEG1, space id, bin id, segment index, virtual offset, page class, codec (0 raw, 1 zstd, 2 zero/hole), payload length, sha256, generation/epoch.
- RepoRAM bins: codec raw, fixed slots, direct Range reads, no decode (quantized weights do not compress).
- repoVRAM bins: logical capacity 1.99 GB each, stored as zstd/zero frames with a footer index; a never-written segment costs 0 bytes, so an empty formatted bin is ~4 KiB and uploading "4 bins" is trivial. Writes never touch a release asset: they go to delta assets (epoch-named), compaction writes a new generation bin.
- Manifest (rr-manifest.json + .sha256): spaces, block table, segment sha256, heat order, generation. Page classes: PIN, STREAM, CACHE.
- Block count is a parameter. 4+4 matches your spec, but 8 GB real cannot exceed 16 GB runner RAM, so the "only possible in sparse virtual" demo needs ~10 blocks (Qwen3-30B-A3B ~18-19 GB) or ~33 blocks (gpt-oss-120b MXFP4 ~60-65 GB). A release holds up to 1000 assets, so this is configuration, not redesign. Verify model sizes/availability first.

### Repo layout
- deeprepo (main): releases reporam-v1 / repovram-v1 (bins + manifest), issues = chat, artifacts, ring.json, ring-state branch.
- deeprepo-cpu-1..5: identical zip. Slot comes from ring.json plus a RING_SLOT variable (1-5); no code differences. Needs DEEPREPO_PAT secret (fine-grained: actions write on the 5 repos, issues write and contents on main).

### Ring lifecycle (hot swap)
- States: COLD -> HYDRATE -> WARM (reader, tails deltas, prefaults hot set) -> ACTIVE (writer) -> DRAIN (flush, write handoff packet) -> EXIT. The slot is re-dispatched on a later lap.
- Knobs: RING_ACTIVE_MIN (default 66), RING_WARM_DEPTH. Depth 4 = all 5 alive, each lives ~330 min (your original design). Proposed default depth 2: lead time covers hydrate+warm and one boot failure is survivable, with fewer concurrent runners (ToS/cost dial). Your call.
- Handoff packet (small): epoch, KV journal pointer, page heat map, router transition table, session/queue cursor. Successor prefetches by heat order, then takes the lease. Predecessor dispatches the slot that has just become depth+1 from now, then drains and exits.
- Lease with a real CAS: a git ref (ring-state branch) updated through the Git Data API with force=false; a non-fast-forward push is rejected, so only one contender wins. Commit count = fencing epoch. All repoVRAM deltas carry the epoch; readers ignore stale epochs. Heartbeat ~30 s, TTL ~90 s (about 120 requests/hr).

### Metering (nothing blows a limit)
- One Meter class in front of every GitHub call, token buckets per class: core REST (cap ~80% of 5000/hr per token; GITHUB_TOKEN is lower, ~1000/hr per repo), content-creating writes (comments, edits, commits: secondary limits, keep well under ~80/min and ~500/hr), dispatches, release uploads. It reads x-ratelimit-* headers and backs off on 403/429 with Retry-After.
- Chat polling: conditional requests with ETag/If-None-Match (304 responses should not count against the primary limit; verify in Phase 0), adaptive 5 s active / 60 s idle. Optional issue_comment workflow as a doorbell that wakes the ring if no heartbeat.
- Replies: stream by editing ONE comment, throttled (>= 2 s between edits), final edit with the stats table.
- Reads: public browser_download_url + Range in 16-64 MiB chunks, redirect target cached until its signed-URL expiry, keep-alive. Never list assets in a hot loop; cache the listing, refresh on manifest change.
- Egress sanity: a 60 GB model re-hydrated every 66 min is ~1.3 TB/day, which looks abusive. For big models use a longer active window (RING_ACTIVE_MIN 150+), demand-page only the hot set, and pull from a peer later. The 8 GB tier at 66 min is ~176 GB/day.

### llama.cpp integration without forking
- Preferred: LD_PRELOAD shim. The model file is a stub GGUF: real header/tensor-info bytes on disk, the rest sparse. The shim intercepts mmap/madvise/munmap for that fd and returns an rrvm region instead. Verify against the pinned llama.cpp source before relying on it; fall back to patching llama_mmap.
- Required flags: keep mmap on, no --no-mmap, no --mlock, disable CPU weight repacking (--no-repack or build flag; it copies every tensor into RAM and defeats paging). Check the flag exists in the pinned version.
- GGUF stacks all experts of a layer in one tensor, so expert slices cannot be page-padded without changing ggml. Skip the repacker for now; compute (tensor, expert) byte ranges from the tensor info and prefetch those ranges at 256 KiB-1 MiB page granularity. rrgguf becomes a manifest/segment-order generator.
- Hints via cb_eval (router top-k observed), speculative layer l+1 gate, transition table fallback, as in the rrvm design below.

### Demos
- Headline: MoE model larger than runner RAM, running only because of the sparse virtual space (Qwen3-30B-A3B first, gpt-oss-120b second). Stats table per reply: tok/s, faults, prefetch hit rate, MB/s per tier, over-budget fills.
- repoVRAM: KV persisted across handoff (kill the writer, successor resumes without re-prefill).
- Stretch only: long-context page-retrieval attention over a huge sparse KV space. Needs a custom decode loop (llama.cpp reads the full KV), and prefill of very long context on 4 vCPU is hours, so persisted KV is what makes it feasible. Do not promise before measuring.
- Do not quote speed numbers until Phase 0/3 measure them.

### Updated build order
0. Phase 0 runner measurements (fio, memory bandwidth, Range throughput to Releases and HF, df/free/nproc, uffd + uffd-wp, signed-URL expiry, 304 accounting).
1. rrbin format tool + verifier (Python), Meter, ring_store.py, hydrate.py (offline tests with DirStore).
2. rrvm core in C (MISSING mode first, then WP) + stress tests.
3. Shim + stub GGUF + Qwen3-30B-A3B end to end.
4. repoVRAM KV space, delta journal, lease CAS, ring workflow, handoff drill.
5. gpt-oss-120b run, issue stats, README with measured numbers.
6. Stretch: peer handoff, zstd pages, zero-copy minor faults, own decode loop.

### Open questions
- RING_WARM_DEPTH default (2 vs 4) and RING_ACTIVE_MIN.
- Keep 4+4 bins as the shipped default, with ~10/33 as the demo scale-up?
- Weights origin: Releases (true RepoRAM) vs Hugging Face with Releases for derived data only?
- Confirm the repoVRAM interpretation (second address space, KV/state, delta-journaled).

---
# Previous TODO (kept as written; items above supersede where they conflict)

# deeprepo TODO / design plan (RepoRAM + repoVRAM, custom pager edition)

## HONEST STATE
- NOTHING NEW IS BUILT OR TESTED. This file is the only change to the package. The 6-repo deploy zips, ring_store.py, hydrate.py and the C pager do not exist yet.
- Checked in the authoring sandbox only: kernel 6.18, the userfaultfd syscall returns a working fd, /dev/fuse exists, gcc 13 present; no libcurl/zstd/cmake headers there. uffd write-protect mode was NOT exercised. A first attempt at the C header failed (shell error) and was discarded.
- Everything below is design. Numbers are projections, to be replaced by measurements in Phase 0.

## Goal
Make the model treat RepoRAM (capacity tier) and repoVRAM (read/write tier) as real, higher-latency RAM, with a completely custom memory layer, and ship one demo that only works because of the sparse virtual space: a model larger than the runner's 16 GB RAM, running on a free CPU runner.

## Terms (my interpretation, correct me if wrong)
- Runner RAM = CPU working memory only: activations, scratch, the hot resident window. It is the L1 the CPU computes on, not a storage tier.
- RepoRAM = node 0, read-mostly. Immutable weights. Big pages (1-4 MiB). Backed by hydrated local disk, cold tier = Releases (public download URLs).
- repoVRAM = node 1, read/write. KV cache, recurrent/SSM state, activation spill. Small pages (256 KiB), 4 KiB dirty tracking, write-back of dirty runs only, journal in a delta repo. GitHub CPU runners have no GPU, so this is a second address space with the VRAM role. If a Colab/Kaggle GPU joins later, its device memory becomes this node's L1 behind the same interface.

## Architecture (4 layers)
- L0 rrvm (C11, native/rrvm/): user-space pager on Linux userfaultfd. Spaces, page table, tiers, prefetch leases, heat-based eviction, uffd-wp dirty tracking, write-back. Transparent to ordinary loads/stores and SIMD kernels.
- L1 rrgguf (Python): repacks a GGUF so every (layer, expert) matrix slice is page-aligned and contiguous, groups pages into 16-64 MiB segments in layer order, writes a manifest (segment sha256, heat order, page sizes) plus rr.* KV metadata in the header.
- L2 engine seam: llama.cpp stays the compute engine. rrvm replaces its model mapping (patch llama_mmap to call rr_map). Hints come from the public cb_eval hook, no ggml fork.
- L3 control: ring_store, hydrate, lease, handoff, issue chat (existing ram16 queue).
- Stretch (only if ggml leaves speed on the table): own C decode loop with AVX2/AVX512 Q4_K and MXFP4 kernels and an explicit expert pipeline. Months of work; not on the critical path.

## rrvm design
- Reserve each space with mmap(NORESERVE), register with uffd (MISSING, plus WP for writable spaces). Per-space page table: state byte (ABSENT, FETCHING, RESIDENT, EVICTING, FLUSHING), class, heat, lease count. 4 KiB dirty bitmap for writable spaces. Tiny: 64 GiB / 256 KiB = 262k entries.
- Backend vtable: read(off,dst,len) -> OK | HOLE, write(off,src,len), close. Stock backends: file (SEEK_DATA/HOLE gives true sparse holes), throttle (latency + bandwidth shaper for tests), readthrough (local file + remote; pulls whole 16-64 MiB segments once, coalesces callers, persists a have-bitmap for resumable boots).
- Fault flow: handler thread reads uffd events. ABSENT -> queue urgent fetch. FETCHING -> nothing (the worker's UFFDIO_COPY wakes the faulter). RESIDENT (stale event) -> UFFDIO_WAKE. Workers: CAS ABSENT->FETCHING, backend read into staging, make room, UFFDIO_COPY (UFFDIO_ZEROPAGE for holes), state RESIDENT, broadcast. In-flight dedupe is the state machine itself.
- Eviction: CLOCK with decaying heat. Skip PIN class and leased pages. STREAM pages start at heat 0 (evict-behind), CACHE pages gain heat from rr_hint (router choices). uffd gives no access bit, so heat is engine-hinted, not sampled. Evict = madvise(DONTNEED). Budget is soft: if no victim exists, fill anyway and count over_budget_fills (never deadlock).
- API: rr_prefetch(off,len,prio) takes a lease and queues absent pages; rr_wait (explicit sync so compute threads do not each fault); rr_release(heat_bump); rr_hint; rr_set_class; rr_flush; rr_stats.
- Dirty tracking (repoVRAM): fill with UFFDIO_COPY_MODE_WP. A write faults once per 4 KiB subpage: mark bit, unprotect that subpage, wake. Flush protocol: CAS RESIDENT->FLUSHING, re-protect the whole page, snapshot+clear bitmap, write only dirty runs, back to RESIDENT. WP events arriving during FLUSHING/EVICTING are parked under one mutex and drained after (wake only on eviction, mark+unprotect on flush). Evict of a writable page = protect, flush, DONTNEED, wake.
- Known sharp edges to test early: WRITEPROTECT on non-resident ranges, parked-event lost-wakeup race, kernel-mode access (read()/write() into the region) is delivered by plain uffd but not by UFFD_USER_MODE_ONLY, unprivileged use needs sudo sysctl -w vm.unprivileged_userfaultfd=1 (runner has sudo).
- Optimization later: memfd + minor-fault mode (UFFDIO_CONTINUE) removes the staging copy; zstd pages; io_uring.

## Capacity: the 8 GB real decision (needs your call)
- 8 GB real cannot hold a model that does not already fit in 16 GB RAM, so it cannot support the "only possible in sparse virtual RAM" demo. Real backing must exceed runner RAM.
- Make block count a parameter. Demo A (safe): Qwen3-30B-A3B Q4_K_M, ~18-19 GB, about 10 blocks. Demo B (headline): gpt-oss-120b MXFP4, ~60-65 GB, about 33 blocks, 5.1B active params; needs virtual space raised from 32 GB to 128 GB (page table cost is trivial) and fits the ~93 GB local disk. Verify exact GGUF sizes and current model availability before committing; my knowledge ends June 2026.
- Weights are immutable and already public on Hugging Face (Range + CDN). The pager's cold backend is pluggable: Releases (true RepoRAM) or HF as origin. Releases then hold only derived data: repacked layout, hot-expert snapshots, KV journals. Peer pull from the previous ring runner beats re-downloading 60 GB every boot.

## RAM budget on a 16 GB runner (verify with free/df at boot)
- OS + runner agent ~2 GB, compute scratch + activations ~2 GB, hot resident window (rrvm RepoRAM budget) ~8-9 GB, repoVRAM resident budget ~1-2 GB. All as env knobs. Disk: hydrated segments, then rrvm pages the hot set in.

## Why MoE is the demo
- Sparse activation matches sparse memory: only routed experts need to be resident. Per-token touched bytes are about the active params (~2.7 GB for gpt-oss-120b, ~1.9 GB for Qwen3-30B-A3B), not the full model.
- Expert-aware prefetch: after the router top-k is computed (cb_eval observes it) prefetch the chosen experts' gate/up/down slices in parallel and rr_wait before the matmul. Speculative router: run layer l+1's cheap gate (n_embd x n_expert) on layer l's output to prefetch one layer ahead. Learned cross-layer transition table as a fallback. Heat hints from router hits make hot experts sticky; first-token cold misses are the cost, steady state is the win.
- Honest expectations (projections): Qwen3-30B-A3B mostly resident after warm-up, plausibly single-digit tok/s. gpt-oss-120b bounded by cold-miss bytes over local-disk bandwidth: roughly 0.3-1 tok/s cold, better as the hot set warms. Unknown until the Phase 0 fio/bandwidth measurements. Do not quote numbers publicly before measuring.

## repoVRAM uses
- KV cache spill past the RAM budget (long context), SSM/recurrent state, activation spill.
- Headline demo 2: persistent KV across ring handoff. Prefill on CPU is expensive; the new runner maps the old runner's flushed KV journal and resumes in seconds instead of re-prefilling. Kill the writer mid-conversation and show the reader take over.

## Demo script
1. Boot step prints free -h (16 GB), model size (60+ GB), rrvm status (128 GiB virtual, resident, hit rates by tier).
2. Chat through an issue comment; each reply posts a stats table: tokens/s, faults, prefetch hit rate, MB/s per tier, over-budget fills.
3. Handoff drill: kill the writer, next ring runner resumes with KV intact.

## Phases (supersedes the old build order; old items kept below)
0. Measure on a real runner: fio on local disk, memory bandwidth, curl Range throughput to Releases CDN and to HF, nproc, free, df, uffd and uffd-wp availability. Workflow step only.
1. rrvm core + tests: sparse zero pages, random access vs oracle, eviction under budget, lease protection, wp dirty/flush/evict races, multithread stress, kill-mid-flush recovery. CI workflow on ubuntu-latest with the sysctl. Synthetic MoE benchmark (Zipf routing, cross-layer correlation, injected latency) comparing demand-only vs transition-table prefetch vs oracle lookahead, labeled synthetic.
2. Backends: readthrough/hydrate (segments, resumable, sha-verified), HTTP Range backend (libcurl, retries, keep-alive), ring_store.py and hydrate.py from the old build order.
3. rrgguf repacker + manifest; llama_mmap patch; cb_eval expert prefetch; run Qwen3-30B-A3B end to end.
4. repoVRAM: KV in a writable space, delta journal repo, handoff resume. Lease + PAT dispatch ring (old items 4-6).
5. gpt-oss-120b headline run, stats reporting to issues, README with measured numbers.
6. Stretch: speculative router tuning, peer page handoff, zstd, minor-fault zero-copy, own decode loop.

## Open questions for you
- Confirm the repoVRAM interpretation above.
- Real backing size: OK to scale RepoRAM to ~33 blocks, or use HF as weights origin and keep Releases for derived data only?
- Demo model: Qwen3-30B-A3B first, then gpt-oss-120b?

## Risks
- Actions ToS restricts workloads unrelated to the repo's software project; bulk model data and a 24/7 ring are a gray area. One account, no limit-dodging, keep an off-GitHub copy.
- Cold pages are 50-200 ms remotely; RAM-like only for the hot set. Unflushed writes (~30 s) are lost on crash.
- 4 vCPU decode is memory-bandwidth bound; there is no GPU on these runners.
- uffd-wp parking/flush races are the highest-risk code; test first.

---
# Carried over from the previous TODO (unchanged)

## HONEST STATE
- NO source files were modified or added this session. Only this TODO changed. I read ram16.py (full), README, workflow and the old TODO; ram16_b2.py, main.cpp and the tests were not re-read.
- Everything below is design for the next session. Nothing here is built or tested.

## Topology (6 repos: 5 CPU ring + 1 RepoRAM)
- ring-0..ring-4: identical runner repos, each with the serve workflow. Same code, differing only by SLOT index.
- reporam: Releases-only data repo. Public, so downloads use browser_download_url (CDN, no API quota).
- Stagger: runner life ~330 min. Start slot k at offset k*66 min via a cron or dispatch chain. At any time 5 are alive in a rolling window, so there is always a warm reader ready to promote, with ~4.5 h of total overlap. Handoff = dispatch ring[(k+1)%5] from a PAT (GITHUB_TOKEN cannot dispatch across repos).
- One writer at a time: lease holder. Others are readers tailing deltas.
- Optional 7th (later): vram repo for KV-cache and activations spill. Only worth it once a GPU source (Colab/Kaggle) exists.

## RepoRAM geometry
- 8 GB real = 4 blocks of 1.99 GB (under the 2 GiB asset cap): reporam-blk-0..3. Virtual space 32 GB, sparse; unmapped = zero page, costs nothing.
- Hierarchy: block (1.99 GB asset) > segment (64 MiB, the unit of Range fetch and of rate-limit accounting) > page (4 MiB weights / 256 KiB KV+state) > subpage (4 KiB, RAM only, dirty bitmap).
- Radix page table: 32 GB / 4 MiB = 8192 entries (tiny). Manifest holds block table + per-segment sha256 + dedupe refs.
- Spread blocks across repos so each repo has separate API limits: bin i -> ring repo i mod N is fine, but keep reads on browser_download_url.

## Rate-limit budget (the thing that killed B2)
- Authenticated REST: 5000 req/hr per token. Asset-API URLs and list calls count; browser_download_url on public repos does not hit the REST quota.
- Rules: never list assets in a hot loop (cache the listing, refresh on manifest change only); fetch in 16-64 MiB Range chunks, not 256 KiB; batch delta publish every 30-60 s and only when dirty; compaction at ~300 deltas (release cap ~1000 assets).
- Hydrate-to-local-disk at boot (93 GB free) then mmap. Remote is cold tier only, so steady-state remote requests approach zero.
- Hydrate order: superblock, embeddings, layer 0..N, trace-ordered after first run. Fetch only changed segments after the first boot (manifest diff).
- Peer handoff: new runner pulls hot segments from the old one over a Cloudflare Tunnel/relay instead of Releases (cuts the 150+ GB/day).
- Budget math to verify against current docs: GitHub states no bandwidth metering on release downloads, but that is not guaranteed. Keep an off-GitHub copy.

## Build order (carry over from old TODO, still valid)
1. ring_store.py: RingStore routing bins to repos, same interface as GitHubStore (list/asset/get/put/put_file/delete), DirStore tests.
2. hydrate.py: parallel ranged download to local disk, hot regions first, resumable, sha-verified, mmap-backed Ram16 with remote fallback.
3. Patch serve(): wrap mem.put / mem.commit / kv.commit in try/except so a failed write cannot kill the loop.
4. Lease + reader/writer roles + overlapping handoff (PAT dispatch). Serve deadline 335 min vs 350 min job timeout.
5. Cloudflare Durable Object lease and live-write relay.
6. Peer-to-peer hot page handoff.
7. Perf: trace prefetch, page zstd + dedupe, hedged reads, zram/zswap warm tier, userfaultfd.
8. Big model: weights immutable in bins; only KV/state writable. Layer-aligned 16-64 MiB segments. Runners are CPU-only, 16 GB RAM, so a 30B-class Q4 (~18 GB) needs mmap from local disk, and expect low tokens/s. GPU needs Colab/Kaggle.
9. Repo series scaffolding: ring-store core, hydrate/runner, lease worker, model forge, one per architecture. Confirm architecture list with the user first.
10. Tests per module.

## Brainstorm bank (ideas, unranked)
- zram on runner: 16 GB RAM can hold ~30-40 GB compressible pages (warm tier).
- Erasure coding k-of-n across repos so one throttled repo never stalls a fetch.
- Hedged/speculative Range reads for the next N segments.
- Content-addressed 64 KiB pages + zstd: zero and repeated pages collapse.
- Write log -> local NVMe journal -> batched delta assets; single-writer lease, last-writer-wins per page.
- Tier by heat: RAM > zram > local disk > peer runner > Releases.
- Native FUSE mount (main.cpp) is likely moot once hydrate-to-disk exists; keep only if disk is too small.
- Model switch: KV cache size scales with context; spill KV to RepoRAM only past ~RAM budget.
- Failure drills: kill writer mid-flush, throttled repo, 404 after compaction, promote reader.

## Risks
- Actions ToS restricts workloads unrelated to the repo's software project; a 24/7 ring plus bulk data in Releases is a gray area. One account only, no limit-dodging accounts, keep an off-GitHub copy.
- Cold pages stay at 50-200 ms. RAM-like only for the hot set.
- Unflushed writes lost on crash (~30 s).

## Carried over (B2 era, still open)
- Real-runner FUSE test, authenticated model sources in format-url, page-fault benchmarks, end-to-end tests with a small GGUF, stronger writer authority.
