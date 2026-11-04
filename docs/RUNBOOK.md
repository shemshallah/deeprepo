# Runbook

Everything runs on GitHub-hosted runners. **Your phone never downloads or stores model
data** — the only local artifacts are source files, docs, and the stamped repo skeletons.

## 0. Repositories

You need eight repositories under one owner:

| repo | role |
|---|---|
| `deeprepo` | coordinator, workflows, engine source |
| `deeprepo-cpu-1` … `deeprepo-cpu-5` | ring slots (disposable runners) |
| `reporam` | sparse RAM tier (KV + registers) |
| `repovram` | sparse VRAM tier (weights) |

Make them **public**: standard Linux runner minutes are free for public repos, and GitHub's
release CDN serves the ranged page reads between repos efficiently. (Private works, but
Actions minutes are billed and cross-repo assets need the PAT everywhere.)

The seven sibling directories are already stamped next to `deeprepo/`:

```bash
ls ~/deeprepo-cpu-* ~/reporam ~/repovram
```

Push each one (the user does the commits):

```bash
for d in deeprepo-cpu-1 deeprepo-cpu-2 deeprepo-cpu-3 deeprepo-cpu-4 deeprepo-cpu-5 reporam repovram; do
  cd ~/$d && git init -q && git add -A && git commit -qm "init $d" && \
  git remote add origin git@github.com:OWNER/$d.git && git push -u origin main || true
done
```

`deeprepo` itself: commit and push as usual (its `.git` already exists).

## 1. One secret: `DEEPREPO_PAT`

Create a **classic PAT** with scopes `repo` + `workflow` (simplest; it covers all eight
repos), or a fine-grained token with:

| repo | permissions |
|---|---|
| `deeprepo` | Contents RW, Actions RW, Issues RW |
| `reporam`, `repovram` | Contents RW (release assets) |
| `deeprepo-cpu-1..5` | Actions RW (dispatch/read runs), Contents R |

Add it as **Actions secret `DEEPREPO_PAT`** in `deeprepo` **and in all five
`deeprepo-cpu-*` repos** (six copies of the same value). Optional repository variables if
you used different names: `REPORAM_REPO`, `REPOVRAM_REPO`, `DEEPREPO_MAIN`.

Never commit the token. Everything reads it from secrets.

## 2. Bring-up order

Run these from **Actions** in `deeprepo`:

1. **02 test** — offline correctness gate (dequant vs ggml, engine vs dense, handoff,
   compaction). No secrets needed. Run it after every code change.
2. **00 bootstrap tiers** — creates the `reporam`/`repovram` releases and bin-0
   superblocks; writes the budget caps; records geometry. Idempotent.
3. **01 setup ring** — verifies all five cpu repos + `runner.yml`, dispatches a `selftest`
   to each (no model needed), records `ring/health`. Fix anything red before continuing.
4. **03 load model** — streams the GGUF from Hugging Face and tiles it into repovram,
   one <=2 GiB bin at a time, resumable. Default model:
   `Qwen/Qwen3-30B-A3B-Instruct-2507-GGUF` Q4_K_M (~18.6 GiB). Expect 20–60 min. If the
   job times out, re-run the same workflow: completed bins are skipped.
5. **04 boot single** — one runner boots the model alone (`deeprepo-cpu-1` by default),
   answers a prompt through prefill + decode, and writes the `gate/single` record. Do not
   start the ring until this is green.
6. **05 serve** — issue chat loop; self-re-dispatches while the budget allows. The
   `keeper` cron (every 20 min) restarts it if it ever stops, but never while one is
   already running and never over budget.

## 3. Chat

- Create an issue titled with your first question, body optional.
- Add the label **`deeprepo-chat`**.
- Subsequent comments are turns in the same session.
- The bot replies in-place and streams partial text (updates every ~8 s).
- `/remember <text>`-style long-term facts are not part of this build; registers keep
  transcripts (`sess/<issue>/<job>`) for continuity.

One-shot (no issue): run **04 boot single** with a custom prompt; output lands in the
workflow summary.

## 4. Operations

| need | action |
|---|---|
| status | run `python -m deepcore status` locally/anywhere with `GH_TOKEN` and env vars, or read the last workflow summary |
| budget | registers `budget/today`; lower caps by writing `budget/caps` |
| pause | cancel `05 serve` / `keeper`; free the lease by writing `{"holder": null}` to `ring/lease` (or just wait for expiry, ≤35 min) |
| compaction | automatic between shifts when ≥48 deltas exist; also `python -m deepcore keeper` |
| swap model | run `03 load model` with a new `url`/`key`; old bins are overwritten bin-by-bin, new `vram/model` record replaces the old |
| re-import same model | leave `fresh` off (resume); turn it on only to force a full re-upload |
- long replies: `05 serve` takes `max_tokens` (default 128 ≈ 40–60 min on Q3; one resident writer does 4×32-token shifts).
| reduce bandwidth | lower `budget` caps, shorten prompts, increase `--cache-pages` (fewer refetches), or run `--strict` (no f16 cache → more page hits are re-dequantized, not refetched) |

## 5. Troubleshooting

| symptom | cause / fix |
|---|---|
| `Missing secret DEEPREPO_PAT` | add the secret to **each** repo (deeprepo + all cpu slots) |
| `repo .../reporam not found` | create the repos, or set `REPORAM_REPO`/`REPOVRAM_REPO` variables |
| prewarm fails with 401/403 | PAT lacks Contents R on reporam/repovram or Actions W on the cpu repo |
| `has no mem-00.bin` | run **00 bootstrap** first |
| `no model loaded` | run **03 load model**; check its summary for the failing bin |
| boot fails on prewarm | the runner checked out an old `deeprepo`; ensure `main` has `deepcore/` |
| model download 404 | paste a direct `…/resolve/main/….gguf` URL; split GGUFs are unsupported |
| `budget exhausted` | daily cap hit; keeper resumes after 00:00 UTC or raise caps |
| reader 404 after compaction | transient; the reader refreshes the manifest and retries once — rerun the step if it persists |
| runner disk full | each import bin is ≤2.1 GB; if you changed geometry, clear `/tmp` |
| serve starts but no reply | issue needs a **user** comment after the bot's last reply; check `ring/health` |

## 5b. Why not bins committed in the repo?

A hosted runner **can** create/modify files of any size in its working tree — we do exactly
that for the SSD page cache (`DEEPREPO_DISK_CACHE`, 6 GiB in `$RUNNER_TEMP`) and for the
temporary import bin. But:

- the VM is discarded at job end, so those files cannot be the shared register;
- `git push` rejects files over **100 MB**, so a 1.99 GiB bin can never be committed;
- 100 MB chunks are pushable but each change rewrites a whole blob into history: a 32 GiB
  tier means a 32 GiB repo, every KV flush adds MBs of history, and every clone/checkout
  pulls it — storage abuse territory;
- Git LFS is the only large-file-in-repo path and is worse here: billed storage and
  bandwidth, content-addressed immutable objects (modification = new object), no Range
  reads for a 4 MiB page.
- The **web editor is a commit editor**: "upload file" caps at 25 MB per file in the
  browser, every save appends a new blob to history, and readers still get whole files, not
  4 MiB pages. Automating the web UI is the same commit backend with extra fragility, not a
  mutable file service. There is no server-side mutable multi-GB file on GitHub; the
  append-only delta journal is the sanctioned way to make a persistent register mutable.

Durable large-file media on GitHub are **release assets** (2 GiB/file, CDN, HTTP Range),
which is why reporam/repovram use them with the delta-journal write path.

## 6. Persistent local tiers (self-hosted / single machine)

Hosted runner VMs are wiped after every job, so bins cannot live in the checkout there —
release tiers are the durable medium. On a machine that **persists** (self-hosted runner,
workstation, phone with enough storage) you can keep the bins as real files inside the repo
tree and modify them in place:

```bash
export DEEPREPO_TIER_DIR="$PWD/tiers"      # gitignored, 32/32 GiB sparse virtual address space
python -m deepcore bootstrap               # forge bin-0 superblocks (8 MiB each, instant)
python -m deepcore load-model --url <gguf-url> --key local-model
python -m deepcore status
python -m deepcore local --prompt "hello" --max-tokens 64
```

- Files created: `tiers/vram/mem-00.bin` (+ generation bins), `tiers/ram/mem-00.bin`,
  `tiers/ram-j/*.dlt` (journal), `tiers/vram-j/*.json` (import progress).
- Layout is identical to the release tiers; the same engine and pager run unchanged.
- `tiers/` is in `.gitignore`. Never commit bins: >100 MB files are rejected by GitHub and
  git would store a full new blob per modification.
- This mode is for one machine (or a shared volume). The hosted ring uses release tiers so
  every slot can read the same memory.
- The runner workflow already creates real files under
  `$GITHUB_WORKSPACE/tiers/l0.5` (the SSD page cache). On hosted runners that whole tree is
  discarded with the VM; on a self-hosted runner with `actions/checkout` set to
  `clean: false`, it survives job to job and can be promoted to a full local tier with
  `DEEPREPO_TIER_DIR=$GITHUB_WORKSPACE/tiers`.
- `local` requires the model to fit the machine's disk once during import (bins are written
  one at a time, then kept in place).

## 7. Changing geometry

Defaults live in `deepcore/__init__.py` (`PAGE`, `PPS`, `BINS`) and in the KV/regions of
`00-bootstrap` (`kv_pages`, `reg_pages`). Changing page/bins after data exists requires a
full re-bootstrap and re-import — do it before loading the model.
