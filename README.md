# RAM16 GGUF runner + sparse 32-GiB Backblaze B2 memory

This package keeps the original `ram16.py` model/GGUF parser, `Machine`, KV and persistent `RegionLog` memory, GitHub Issues queue, and inference loop. `ram16_b2.py` supplies the B2-backed byte-addressable store.

## Architecture

- **GitHub Actions:** disposable CPU runner and the single writer coordinator.
- **Backblaze B2:** persistent compressed page objects, immutable generation manifests, and the current-generation register.
- **GitHub Issues:** chat/session queue (`ram16-chat` label by default).
- **Virtual address space:** 32 GiB, 4 MiB pages (8,192 pages). Unwritten pages are implicit zeroes. Nonzero pages are zlib-compressed and content-addressed.
- **Original RAM16 layout:** page 0 superblock; GGUF weights; KV spill region; persistent memory region, calculated by `ram16.plan_layout()`.

This is **sparse virtual storage**, not 32 GiB of physical RAM or VRAM. A 32-GiB incompressible image cannot fit in a less-than-10-GB object-storage allowance. The default safety cap is 9,500,000,000 bytes for visible objects under `ram16/pages/`; it does not account for unrelated bucket data. Use a dedicated bucket/prefix and keep quota headroom.

## Configure GitHub

Create these repository secrets (never commit credentials):

- `B2_KEY_ID` — Backblaze application key ID
- `B2_APPLICATION_KEY` — application key secret
- `B2_BUCKET` — bucket name
- `B2_ENDPOINT` — region-specific S3 endpoint, e.g. `https://s3.us-west-004.backblazeb2.com`

The main application key needs read/list/write permissions for normal operation. For tighter isolation, optionally create `B2_READ_KEY_ID` and `B2_READ_APPLICATION_KEY` secrets for a bucket-restricted, read-only key used only by the native GGUF mount. The mount falls back to the main key if these are absent. To run garbage collection with `--apply`, the main key also needs permission to delete objects and list object versions.

The workflow automatically sets `RAM16_COORDINATOR_REPO` to `${{ github.repository }}`. Your repository named `deeprepo` is therefore the canonical coordinator; no manually configured coordinator variable is needed. All B2-mutating workflow runs should be dispatched from this repository. GitHub Actions concurrency groups are repository-scoped, so this repository serializes cooperating writers. Other repositories should dispatch this repository's workflow rather than writing to B2 directly. This is cooperative coordination, not a storage-provider atomic compare-and-swap; do not give the same B2 write key to untrusted or independently operated writers.

`GITHUB_TOKEN` is provided by Actions; the workflow permissions allow issue comments and optional workflow handoff.

## First-time setup

1. Add the package files to your GitHub repository named `deeprepo`, preserving `.github/workflows/ram16-b2.yml`.
2. Add the four B2 secrets. No `RAM16_COORDINATOR_REPO` variable is required; the workflow derives the canonical `owner/deeprepo` identity from GitHub.
3. Open **Actions → RAM16 GGUF runner on Backblaze B2 → Run workflow**.
4. Run `init` to create the empty sparse address space.
5. Run `format-url`. The `model_url` input is prefilled with the Qwen3-4B-Instruct-2507 Q4_K_M GGUF URL tested for byte-range support. Keep `kv_mb=1024`, `batch_pages=8`, and `restart_import=false` for the first import. The URL server must provide `Content-Length` and honor HTTP byte ranges; a stable ETag or Last-Modified value is preferred.
6. Apply the `ram16-chat` label to an issue, then run `serve`. Comment in the issue to converse with the model. `/remember <text>` stores a persistent fact in the original `RegionLog` memory region.

### Resumable imports

`format-url` commits each batch to B2 and then advances `ram16/IMPORT.json`. If the runner stops, run the same `format-url` command again with the same URL and layout. It verifies source identity, resumes from the last committed offset using HTTP Range, and reconstructs SHA-256 by rereading the already imported prefix from B2. If you intentionally want to abandon the partial import, select `restart_import=true` in the workflow; this clears the import checkpoint and starts again. Do not run two imports concurrently.

The final superblock is marked ready only after the full byte count and SHA-256 have been verified. The normal import path computes a SHA-256 over the complete source GGUF before marking the superblock ready. Each stored page is independently SHA-256 verified when the native reader fetches it.


### Native adapter manual build

```bash
sudo apt-get install cmake pkg-config libfuse3-dev libcurl4-openssl-dev libssl-dev zlib1g-dev libboost-dev
cmake -S native/ram16-gguf-fuse -B native/ram16-gguf-fuse/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/ram16-gguf-fuse/build --parallel 2
```

The mount binary reads `B2_BUCKET`, `B2_ENDPOINT`, optional `B2_REGION` / `RAM16_B2_PREFIX`, and prefers `B2_READ_KEY_ID` / `B2_READ_APPLICATION_KEY` over the main write credentials. The workflow starts it automatically during `serve` and unmounts it on exit.

## Commands

```bash
python -m pip install -r requirements-b2.txt
python ram16_b2.py init
python ram16_b2.py status
python ram16_b2.py format --gguf /path/to/model.gguf
python ram16_b2.py format-url https://example.org/model.gguf
python ram16_b2.py serve --minutes 330 --label ram16-chat --handoff
python ram16_b2.py commit
python ram16_b2.py gc                         # dry-run only
python ram16_b2.py gc --min-age-days 7 --retain-generations 5 --apply
```

For local administration, writer commands are disabled by default. Set `RAM16_ALLOW_LOCAL_WRITES=1` explicitly in a controlled environment if you intentionally want to bypass the GitHub coordinator check. Never run a local writer at the same time as a workflow writer.

## Garbage collection safety

`gc` is **dry-run by default**. It validates visible manifests, retains the newest N generations and every generation younger than the age threshold, and only considers page objects not referenced by those retained manifests. `--apply` deletes old object versions, not just current names, because B2 buckets are versioned by default. The minimum age is at least one day; keep a generous retention window and run a dry-run first. The operation is serialized by the canonical repository's workflow concurrency group. Do not run GC while a non-cooperating external writer/importer is active.

## Runtime limits

### Native remote GGUF I/O (no full local model copy)

The `serve` workflow now builds and starts `native/ram16-gguf-fuse/ram16-gguf-mount`, a native C++17 FUSE3 filesystem. It exposes the imported GGUF as a normal read-only file at `/tmp/ram16-b2-mount/model.gguf`. The file's logical size is the GGUF size; its bytes are read directly from the RAM16 B2 page map.

The data path is:

1. llama.cpp/ggml opens the mounted file through the existing `llama-cpp-python` model path.
2. `use_mmap=True` lets the existing loader use normal memory mapping; no llama.cpp fork or local full-file materialization is required.
3. Linux page faults/read requests enter the FUSE read callback.
4. The C++ adapter signs S3-compatible GET requests using AWS Signature V4, fetches the referenced compressed 4-MiB B2 page object, decompresses it, verifies its SHA-256, and returns the requested byte range.
5. A bounded native LRU cache (8 pages / 32 MiB by default) and the kernel page cache reduce repeated object reads.

The adapter validates the `CURRENT` pointer against the manifest SHA-256, checks the generation and virtual geometry, validates page references, and verifies each fetched page before returning data. It mounts read-only. The existing `OutOfCoreGGUF.to_file()` path remains available for local use when `RAM16_GGUF_MOUNT_PATH` is not set.

**Build dependencies on the GitHub runner:** `libfuse3-dev`, `libcurl4-openssl-dev`, `libssl-dev`, `zlib1g-dev`, `libboost-dev`, CMake and pkg-config. The workflow installs these and builds the C++ adapter automatically for `serve`. The runner must expose `/dev/fuse`; if not, the job fails explicitly rather than silently downloading the entire model.

**Performance/limits:** B2 is still object storage, not RAM or VRAM. The first access to each 4-MiB page incurs a remote object GET and decompression. A 10-GiB model can require roughly 2,560 page-object fetches during a complete scan. mmap removes the complete local GGUF copy, but does not remove llama.cpp's need to allocate tensor buffers/working memory, nor does it make remote page faults low-latency. The runner must still have enough RAM/CPU/GPU memory for the model's actual inference allocations. The mount is read-only; model writes, training checkpoints and KV spill continue through the separate B2 register/memory path.

This is a native mmap-compatible GGUF I/O adapter at the filesystem boundary, not a new ggml compute backend or a modified GGML tensor kernel. It deliberately preserves upstream llama.cpp compatibility instead of maintaining a brittle private fork.

GitHub-hosted runners are temporary and time-limited. `--handoff` starts another workflow run; it does not preserve process RAM between jobs.

## Verification status

Local verification passed for Python compilation, workflow YAML parsing, native C++ syntax checking, and a native integration harness backed by a local S3-compatible HTTP fixture. The harness loaded the B2 `CURRENT` register and manifest, verified page hashes, decompressed page objects, and performed random-offset reads; the C++ AWS Signature V4 authorization header matched botocore's reference signer. This environment did not have libfuse3 development headers or `/dev/fuse`, so a real FUSE mount/build and actual GitHub-hosted-runner inference remain unverified until the workflow is run. No real B2 credentials or model were available here.

## Current engineering boundaries

- Coordinator serialization applies to cooperating clients that use the canonical repository workflow. It is not an atomic cross-provider CAS and cannot stop a client holding the B2 key from bypassing this program.
- The B2 safety cap measures current visible page objects, not unrelated bucket contents or historical versions hidden by delete markers. Run GC to reclaim old versions; keep quota headroom.
- End-to-end tests against your actual B2 bucket and your selected GGUF still require your credentials and account permissions; no credentials were supplied for this local verification.
