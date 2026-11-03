#!/usr/bin/env python3
"""Stamp the sibling repos locally: deeprepo-cpu-1..5, reporam, repovram.

You create/push the repos yourself; this only writes files next to deeprepo/.
  python tools/stamp_repos.py [--parent DIR] [--git-init]
"""
import argparse
import os
import stat

RUNNER_YML = """name: CPU runner
# Dispatched by deeprepo (PAT) for one of: prewarm (hydrate+verify), run (a compute
# shift), selftest (no model needed). Everything big streams through the repo tiers;
# this VM only ever holds a bounded page cache and at most one <=2 GiB import bin.
on:
  workflow_dispatch:
    inputs:
      op:
        description: prewarm | run | selftest
        type: choice
        options: [prewarm, run, selftest]
        default: prewarm
      job:
        description: job id for op=run
        type: string
        default: ""
      minutes:
        description: shift wall-clock budget
        type: string
        default: "20"

permissions:
  contents: read

concurrency:
  group: runner
  cancel-in-progress: false

env:
  GH_TOKEN: ${{ secrets.DEEPREPO_PAT }}
  DEEPREPO_RAM: ${{ vars.REPORAM_REPO || format('{0}/reporam', github.repository_owner) }}
  DEEPREPO_VRAM: ${{ vars.REPOVRAM_REPO || format('{0}/repovram', github.repository_owner) }}
  DEEPREPO_MAIN: ${{ vars.DEEPREPO_MAIN || format('{0}/deeprepo', github.repository_owner) }}
  DEEPREPO_RAM_BACKEND: ${{ vars.RAM_BACKEND || 'github' }}
  DEEPREPO_VRAM_BACKEND: ${{ vars.VRAM_BACKEND || 'github' }}
  DEEPREPO_MIDDLE_URL: ${{ vars.MIDDLE_URL }}
  DEEPREPO_MIDDLE_TOKEN: ${{ secrets.MIDDLE_TOKEN }}
  RING_SLOT: ${{ github.repository }}
  # L0.5: real files created inside the workspace. On hosted runners this is scratch that
  # dies with the VM; on a self-hosted runner with `clean: false` it survives across jobs.
  DEEPREPO_DISK_CACHE: ${{ github.workspace }}/tiers/l0.5
  DEEPREPO_DISK_CACHE_GB: "8"

jobs:
  runner:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      - name: Preflight
        run: |
          if [ -z "$GH_TOKEN" ]; then
            echo "::error::Add repository secret DEEPREPO_PAT (contents RW on $DEEPREPO_RAM/$DEEPREPO_VRAM, read on $DEEPREPO_MAIN)."
            exit 1
          fi
          echo "slot=$RING_SLOT ram=$DEEPREPO_RAM vram=$DEEPREPO_VRAM main=$DEEPREPO_MAIN"
          df -h / | tail -1

      - name: Checkout deeprepo core
        uses: actions/checkout@v4
        with:
          repository: ${{ env.DEEPREPO_MAIN }}
          path: core
          token: ${{ secrets.DEEPREPO_PAT || github.token }}

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: core/requirements-core.txt

      - run: pip install -r core/requirements-core.txt

      - name: Execute
        working-directory: core
        run: |
          set -euo pipefail
          python -m deepcore.worker "${{ inputs.op }}" --job "${{ inputs.job }}" --shift-minutes "${{ inputs.minutes }}" --shift-tokens 32 --max-shifts 4 --cache-pages 1024 --kv-cache-pages 192
"""

CPU_README = """# {name}

CPU ring slot for **deeprepo**. This repository holds no model weights and no data:
it is a disposable runner. deeprepo dispatches `runner.yml` here; the job

1. hydrates against `repovram` (weights, sparse release bins) and `reporam` (KV + registers),
2. takes the write lease, runs a bounded decode/prefill shift on the shared model,
3. publishes tokens + handoff state and releases the lease.

## Required setup (once)

- Secret `DEEPREPO_PAT`: fine-grained token with
  - `Contents: read/write` on `reporam` and `repovram`
  - `Contents: read` on `deeprepo` (and this repo)
  - `Actions: read/write` on this repo (for reports), `Issues: write` on `deeprepo`
- The ring topology is resolved automatically from the repo owner; override with repository
  variables `REPORAM_REPO`, `REPOVRAM_REPO`, `DEEPREPO_MAIN` if you named things differently.

Nothing runs on a schedule here; only dispatched shifts. See deeprepo `docs/RUNBOOK.md`.
"""

TIER_README = {
    "reporam": """# reporam — sparse virtual RAM tier

Backing store for deeprepo's **RAM tier**: KV cache, hierarchical registers (ring lease,
job specs, token stream, budgets, transcripts), and scratch.

- 16 release bins x 1.99 GiB = **~31.9 GiB sparse virtual address space**, 4 MiB pages.
- **Sparse**: only written pages exist. A few tokens of KV occupy a few KB of assets.
- **Writes are append-only**: each flush uploads a compressed page-delta asset
  (`d-<seq>.dlt`); compaction folds deltas into a new generation of trimmed bins
  (`mem-XX.gN.bin`), publishes a manifest, then deletes old bins and consumed deltas.
  No release asset is ever modified in place.
- Readers use HTTP Range + a bounded local page cache. Do not clone this repo.
- Managed by `deeprepo` workflows (`00-bootstrap`, `05-serve`, `keeper`). Do not push data here.
""",
    "repovram": """# repovram — sparse virtual VRAM tier

Backing store for deeprepo's **weights**: a GGUF (default Qwen3-30B-A3B-Instruct-2507
Q4_K_M, ~18.6 GiB) tiled into 16 release bins of 1.99 GiB (~31.9 GiB sparse virtual).

- Read-only after import. Each bin holds 510 x 4 MiB pages; page p of the model lives at
  vram page 1+p.
- **Never cloned, never downloaded whole**: the engine reads 4 MiB pages over HTTP Range
  and verifies them through ram16's page cache.
- Import is resumable and runs entirely on a GitHub runner (`03-load-model`): it streams
  from Hugging Face, builds one <=2 GiB bin at a time, uploads, deletes locally.
- Managed by `deeprepo` workflows (`00-bootstrap`, `03-load-model`). Do not push data here.
""",
}


def write(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--git-init", action="store_true", help="git init each stamped repo (no commit)")
    a = ap.parse_args()
    names = [f"deeprepo-cpu-{i}" for i in range(1, 6)] + ["reporam", "repovram"]
    for n in names:
        root = os.path.join(a.parent, n)
        os.makedirs(root, exist_ok=True)
        if n.startswith("deeprepo-cpu"):
            write(os.path.join(root, ".github", "workflows", "runner.yml"), RUNNER_YML)
            write(os.path.join(root, "README.md"), CPU_README.format(name=n))
        else:
            write(os.path.join(root, "README.md"), TIER_README[n])
            write(os.path.join(root, ".gitignore"), "*\n!.gitignore\n!README.md\n!docs/\n")
            write(os.path.join(root, "docs", "REGISTERS.md" if n == "reporam" else "LAYOUT.md"),
                  "Managed by deeprepo bootstrap. See deeprepo docs/MEMORY_MAP.md.\n")
        if a.git_init and not os.path.isdir(os.path.join(root, ".git")):
            os.system(f'cd "{root}" && git init -q')
        print(f"stamped {root}")
    print("\nNext: create the 7 repos on GitHub, push each directory, then in deeprepo:")
    print("  1) add secret DEEPREPO_PAT (see docs/RUNBOOK.md)")
    print("  2) run 00-bootstrap -> 01-setup-ring -> 02-test -> 03-load-model -> 04-boot-single -> 05-serve")


if __name__ == "__main__":
    main()