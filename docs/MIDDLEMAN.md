# Middleman: control plane and mutable hot tier

Hosted runner VMs cannot accept inbound connections and their disks are wiped after every
job. That makes two things impossible from inside Actions alone:

1. a *shared, mutable, low-latency* page register that every runner can read/write, and
2. a 24/7 orchestrator that is not GitHub's cron (scheduled workflows are disabled after
   60 days of repository inactivity).

A middleman you host solves both. It is **optional**; the default system is GitHub-only
(release tiers + keeper cron). Two distinct roles, deliberately separate:

## Role A — mutable hot tier (data plane)

`deepcore/middle.py` is an HTTP blob store with ranged reads and ranged writes, bearer-token
auth, and the same interfaces as the GitHub release tiers. `Ram16` does not know the
difference; the journal, compaction, manifests, and sparse bins all work unchanged.

```bash
MIDDLE_TOKEN=$(openssl rand -hex 32) python -m deepcore.middle serve --root /data --port 8080
```

Point one or both tiers at it:

```bash
DEEPREPO_MIDDLE_URL=https://middle.example.com
DEEPREPO_MIDDLE_TOKEN=<token>
DEEPREPO_RAM_BACKEND=middle      # mutable reporam here
DEEPREPO_VRAM_BACKEND=github     # weights stay on the free release CDN (recommended)
```

Recommended split:

- **reporam on the middleman** — it is small (KV + registers: MBs to a few hundred MBs),
  write-heavy, and latency-sensitive.
- **repovram on GitHub releases** — immutable, ~18.6 GiB, served free and fast by the
  release CDN. Proxying this through a middleman doubles egress, adds a single point of
  failure, and buys nothing.

If you put both tiers on the middleman, budget disk: 19 GiB for weights + KV growth, plus
snapshots. A 2–5 GiB volume is enough for reporam only.

## Role B — control plane (PAT lives here)

The same package runs the existing keeper as a long-lived process on any host with outbound
HTTPS and a PAT:

```bash
GH_TOKEN=<fine-grained PAT or GitHub App token> \
DEEPREPO_RAM=owner/reporam DEEPREPO_VRAM=owner/repovram \
DEEPREPO_MAIN=owner/deeprepo \
python -m deepcore keeper --loop --interval 600
```

Once per interval it:

1. reads the budget ledger and pauses when a cap is hit;
2. checks the stability gate and the model record;
3. checks the ring lease (never fights a live shift);
4. compacts reporam generation bins when the journal grows;
5. dispatches `05-serve` when the ring is idle.

**The middleman is never in the weight/KV data path.** Prefill and decode page weights
directly from GitHub's release CDN; a 512 MB free-tier service cannot proxy 18 GiB and
must not try. The Koyeb service runs only the keeper (dispatch, quota, status). Measured:
a full 30B boot moved 28.7 GB through the runner<->CDN path with **zero** page requests
through Koyeb.

This is the "machinery that uses a PAT to read/write the ephemeral RepoRAM and cycle the
runners" — but **without proxying bulk bytes**. Runners still talk directly to GitHub:

```
cpu runner ──443──▶ GitHub release CDN / API        (weights + KV pages, fast, no double hop)
     ▲
     │ workflow_dispatch (PAT)
middleman (control plane) ──▶ GitHub API (dispatch, run status, compaction, GC)
```

Disable the GitHub `keeper` workflow when you run the hosted keeper, or leave it on and rely
on the `serve` concurrency group (one pending run max) — both are safe, but duplicated
dispatch wastes API calls.

### Security

- Prefer a **GitHub App** installation token over a long-lived PAT: the middleman can mint
  short-lived tokens scoped to the eight repos. A PAT is fine for a single-user setup but is
  a high-value secret once it leaves Actions.
- Minimum scopes: Contents RW on `reporam`/`repovram`, Actions RW on `deeprepo` and
  `deeprepo-cpu-1..5`, Issues RW on `deeprepo`.
- Terminate TLS at the host; never expose the store over plain HTTP.
- The token is only in the host's environment/secret manager. Never in the repo.
- The middleman is a single trust boundary: back up its volume, and treat its data as
  session state, not the source of truth.

## Weight and release lifecycle (important)

The weights are **immutable** and should be deleted/re-uploaded only when the model itself
changes:

```
import once  -> bins on repovram (10 of 16 bins for Qwen3-30B-A3B Q4_K_M)
handoffs     -> KV moves through reporam only; weights are re-read from the CDN
model swap   -> write new bins -> update vram/model -> delete old bins (or keep a generation)
```

Deleting the weights on every handoff would:

- force every following runner to re-download 18.6 GiB (6+ minutes at typical runner speeds),
- invalidate the CDN/edge cache and the runner SSD cache,
- 404 readers mid-shift (asset deletion is immediate),
- burn the daily upload budget for zero benefit — the weights did not change.

What *is* ephemeral per session is reporam: deltas are compacted into generations and old
deltas/bins are deleted. You can also wipe a session's KV region explicitly after its
transcript is recorded.

## Hosting matrix

| host | can run the middleman? | notes |
|---|---|---|
| GitHub Pages | no | static only, no code, no writes |
| Cloudflare Workers + R2 | control: yes; store: needs a page-granular Worker | Workers can't run this Python; R2 has no ranged writes, so store each 4 MiB page as an object and rewrite the Worker API accordingly. Egress-free, 10 GB free. |
| PythonAnywhere (free) | no | 512 MB disk, web-app terms, no room for state |
| PythonAnywhere (paid) | control plane only | tiny keeper/coordinator, not a tier |
| Koyeb / Fly / Render / Railway | yes | run the Dockerfile with a volume; ~$0–5/mo for 2–5 GiB |
| VPS / Raspberry Pi + tunnel | yes | cheapest at scale, you own backups/TLS |
| self-hosted runner | yes | run it beside the runner; use `DEEPREPO_TIER_DIR` for the full local tier instead |

## Deploy on Koyeb (recommended control plane)

Koyeb is a good home for the **keeper** (stateless, needs a PAT, outbound HTTPS) and, if your
plan includes a persistent volume, for the reporam hot tier as well.

**Option A — build from GitHub (simplest):** Koyeb dashboard → Create Service → GitHub →
`shemshallah/deeprepo`, builder = Dockerfile, port `8080`. Environment:

```
MIDDLE_TOKEN   = <cat ~/.deeprepo-secrets/middle_token>
GH_TOKEN       = <your PAT>            # only for the keeper service
DEEPREPO_RAM   = shemshallah/reporam
DEEPREPO_VRAM  = shemshallah/repovram
DEEPREPO_MAIN  = shemshallah/deeprepo
```

For the **keeper**, override the run command:

```
python -m deepcore keeper --loop --interval 600
```

For the **store**, keep the default command (`python -m deepcore.middle serve --root /data --port 8080`)
and attach a volume at `/data` if your plan supports it. Without a volume the store is
ephemeral (fine for testing, not for reporam).

**Option B — image from GHCR:** run workflow `06 middle image`, then create a Koyeb service
from `ghcr.io/shemshallah/deeprepo-middle:latest`. Make the package public in GitHub →
Packages, or add a Koyeb registry credential (a PAT with `read:packages`).

After deploy:

1. Set `MIDDLE_URL=https://<service>.koyeb.app` as a repository **variable** in `deeprepo`
   and all five cpu repos.
2. To put reporam on it: set variable `RAM_BACKEND=middle` everywhere, then re-run
   `00-bootstrap` (fresh middleman store) and `05-serve`. Keep `VRAM_BACKEND=github`.
3. If the keeper runs on Koyeb, disable the GitHub `keeper` workflow schedule.

Note: Koyeb's free instance has no persistent volume; use it for the keeper, keep reporam on
GitHub releases (default), or move the tier to a volume-enabled instance.

## Deploy (Fly example)

```bash
fly launch --no-deploy
fly volumes create middle_data --size 3
fly secrets set MIDDLE_TOKEN=$(openssl rand -hex 32)
fly deploy
# optional hosted keeper (same image, different command):
fly secrets set GH_TOKEN=<pat> DEEPREPO_RAM=owner/reporam DEEPREPO_VRAM=owner/repovram \
  DEEPREPO_MAIN=owner/deeprepo
fly machine run . --app deeprepo-middle \
  --command "python -m deepcore keeper --loop --interval 600"
```

Koyeb/Render: deploy the Dockerfile, mount a volume at `/data`, set `MIDDLE_TOKEN`, expose
port 8080. Then set repository variables `MIDDLE_URL` and secret `MIDDLE_TOKEN` in
`deeprepo` and every cpu repo; select `RAM_BACKEND=middle` if you want the hot tier there.

## Failure modes

| failure | effect | mitigation |
|---|---|---|
| middleman down | reporam reads/writes fail; shifts stop | keep GitHub as the durable tier and journal; retry/backoff in `Asset`; restart the service |
| middleman disk full | writes fail with 5xx | cap KV/context, compact, prune sessions, alert on `/healthz` plus disk |
| token revoked | dispatch/compaction stop | rotate the secret; runners keep serving until their jobs end |
| duplicated keeper | two serve dispatches | concurrency group keeps one pending; still prefer one keeper |
| lost volume | session state lost | weights are on GitHub; re-import not required unless repovram was also there |
