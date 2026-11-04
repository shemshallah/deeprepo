# Batch ring (throughput mode)

The 24/7 ring is **latency/persistence oriented**: one write lease, KV in reporam, handoffs.
Batch mode is the opposite trade for **throughput**: up to five independent conversations run
at once, one per CPU slot, each with **local KV** (`MemKV`) and no reporam/journal writes.

```
07-batch (dispatch) ──> cpu-1 issue 8
                    ├─> cpu-2 issue 9
                    ├─> cpu-3 issue 10
                    ├─> cpu-4 issue 11
                    └─> cpu-5 issue 12
each runner: fetch issue -> tokenize -> page weights from repovram -> answer -> comment
```

- No lease, no shared writer, no handoff: jobs cannot corrupt each other.
- KV lives in the runner's RAM for that turn only (not in repoRAM). The main ring keeps the
  repo-backed KV claim; batch trades that for 5× aggregate throughput.
- Weights still come from repovram (Q3 or Q4 via the model registry / `CHAT_MODEL`).
- Replies are posted when generation finishes (updates every ~8 s once decoding starts);
  there is no ring handoff.
- Each dispatched issue gets the `deeprepo-busy` label until its runner finishes; `pending` skips busy issues so a turn is never double-assigned. A crashed runner can leave the label behind — remove it manually.
- **Don't run 05-serve and 07-batch on the same issues simultaneously** — both may answer.
  Pause 05 (cancel the serve window) while batching.
- Per-runner memory: page cache 8 GiB + f16 cache 1.8 GiB + head 0.6 GiB + MemKV (~50 MB/turn)
  + expert cache 1 GiB. With Q3 (~13 GiB), the 8 GiB RAM + 8 GiB SSD caches cover the model
  almost entirely, so batch decode is substantially faster than Q4 paging.

Run: Actions → **07 batch ring** (limit 5, model empty = `vars.CHAT_MODEL`).

`python -m deepcore batch --limit 5` (with `GH_TOKEN`, tier env) dispatches; the runner uses
`deepcore.worker chat --issue N --model KEY`.
