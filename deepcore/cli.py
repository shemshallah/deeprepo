"""python -m deepcore <cmd> — operator entry points.

status | bootstrap | load-model | sweep | gate | serve | once | keeper | worker | local
"""
import json
import sys
import time


def cmd_status(env):
    from .geom import read_super
    sb_r, sb_v = read_super(env.ram), read_super(env.vram)
    mi = env.model_info() or {}
    b = env.budget.snapshot()
    print(f"reporam  {env.ram_repo}  gen {env.ram.gen} deltas {len(env.ram.journal.list()) if env.ram.journal else 0} "
          f"pages {env.ram.npages} ({env.ram.size/2**30:.1f} GiB virtual)")
    print(f"repovram {env.vram_repo}  gen {env.vram.gen} pages {env.vram.npages}")
    print("ram layout", json.dumps(sb_r["layout"]))
    print("model", json.dumps({k: mi.get(k) for k in ("key", "ready", "size", "weight_pages")}))
    print("budget", json.dumps(b))
    print("lease", json.dumps(env.regs.get("ring/lease")))
    print("gate", json.dumps(env.regs.get("gate/single")))
    health = env.regs.get("ring/health") or {}
    for r, h in sorted(health.items()):
        print(f"  {r}: {'OK' if h.get('ok') else 'DOWN'} ({h.get('mode', '?')}) @ {h.get('t')}")
    return 0


def cmd_bootstrap(argv):
    """Bootstrap does not open Env (reporam may not exist yet)."""
    import os
    import ram16
    from . import TAG_RAM, TAG_VRAM
    from .tiers import bootstrap
    arg = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    local = os.environ.get("DEEPREPO_TIER_DIR")
    kv_pages = int(arg("--kv-pages", 1024))
    reg_pages = int(arg("--reg-pages", 256))
    if local:
        from .store import local_bootstrap
        out = local_bootstrap(local, kv_pages=kv_pages, reg_pages=reg_pages,
                              force="--force" in argv)
        print(json.dumps(out))
        return 0
    ram_repo = os.environ.get("DEEPREPO_RAM", "reporam")
    vram_repo = os.environ.get("DEEPREPO_VRAM", "repovram")
    from .store import open_store
    from .tiers import ensure_journals
    murl, mtok = os.environ.get("DEEPREPO_MIDDLE_URL", ""), os.environ.get("DEEPREPO_MIDDLE_TOKEN", "")
    ram_backend = os.environ.get("DEEPREPO_RAM_BACKEND", "github")
    vram_backend = os.environ.get("DEEPREPO_VRAM_BACKEND", "github")
    ensure_journals(vram_repo, ram_repo, ram_backend, token, murl, mtok)
    ensure_journals(vram_repo, ram_repo, vram_backend, token, murl, mtok)
    out = bootstrap(open_store(vram_repo, TAG_VRAM, os.environ.get("DEEPREPO_VRAM_BACKEND", "github"), token, murl, mtok),
                    open_store(ram_repo, TAG_RAM, os.environ.get("DEEPREPO_RAM_BACKEND", "github"), token, murl, mtok),
                    kv_pages=kv_pages, reg_pages=reg_pages, force="--force" in argv)
    print(json.dumps(out))
    return 0


def cmd_load_model(env, url, model_key="qwen3-30b-a3b-q4_k_m", resume=True):
    from .store import HfFile
    from .tiers import import_model
    import ram16
    import os
    hf = HfFile(url)
    if getattr(env, "local_dir", None):
        from .store import LocalJournal
        import os as _os
        jstore = LocalJournal(_os.path.join(env.local_dir, "vram-j"))
    else:
        from .store import open_store
        jstore = open_store(env.vram_repo, "repovram-j", getattr(env, "vram_backend", "github"),
                            env.token, getattr(env, "middle_url", ""), getattr(env, "middle_token", ""))
    res = import_model(env.vram.tier_store, jstore, hf, budget=None, resume=resume)
    env.regs.put("vram/model", {"key": model_key, "ready": True, "size": res["size"],
                                "weight_pages": res["weight_pages"], "bins": res["bins"],
                                "url": url, "t": int(time.time())})
    env.regs.commit()
    print(json.dumps(res))
    return 0


def cmd_sweep(env, op="selftest"):
    from . import ring
    from .coordinator import Coordinator
    c = Coordinator(env)
    out = {}
    for r in ring.RING:
        try:
            out[r] = c.prewarm(r, op=op)
        except Exception as e:
            out[r] = False
            print(f"{r}: {e!r}")
    print(json.dumps(out))
    return 0 if out and all(out.values()) else 1


def cmd_gate(env, ok, note=""):
    env.regs.put("gate/single", {"ok": bool(ok), "note": note, "t": int(time.time())})
    env.regs.commit()
    print(json.dumps(env.regs.get("gate/single")))
    return 0 if ok else 1


def cmd_keeper(env, loop=False, interval=600):
    """24/7 guard: budget, compaction, lease-aware restart. Can run as a long-lived process
    on any host (Koyeb/Fly/Render/VPS/self-hosted runner) with a PAT, replacing GitHub cron.
    Data-plane traffic stays direct runner<->GitHub; this is control-plane only."""
    import time as _t
    while True:
        try:
            out = _keeper_tick(env)
        except Exception as e:
            out = {"t": int(_t.time()), "error": repr(e)}
            print(json.dumps(out))
        if not loop:
            return 0
        _t.sleep(max(60, int(interval)))


def _keeper_tick(env):
    from . import ring
    import os, time
    regs = env.regs
    out = {"t": int(time.time())}
    if env.budget.exhausted():
        out["action"] = "paused: budget exhausted"
        print(json.dumps(out))
        return out
    mi = env.model_info() or {}
    gate = regs.get("gate/single") or {}
    if not mi.get("ready"):
        out["action"] = "idle: no model loaded (run 03-load-model)"
    elif not gate.get("ok"):
        out["action"] = "idle: single-runner stability gate not passed (run 04-boot-single)"
    else:
        lease = regs.get("ring/lease") or {}
        active = bool(lease.get("holder")) and (lease.get("until", 0) > time.time())
        if active:
            out["action"] = f"busy: {lease['holder']} holds the lease"
        else:
            if len(env.ram.journal.list()) >= 32:
                from .tiers import compact_trimmed
                try:
                    out["compact_gen"] = compact_trimmed(env.ram)
                except Exception as e:
                    out["compact_error"] = repr(e)
            repo = os.environ.get("GITHUB_REPOSITORY", "")
            if repo and os.environ.get("KEEPER_DISPATCH", "1") == "1":
                try:
                    ring.dispatch_workflow(repo, "05-serve.yml", env.token, {"minutes": "300", "loop": "true"})
                    out["action"] = "dispatched 05-serve"
                except Exception as e:
                    out["action"] = f"dispatch failed: {e!r}"
            else:
                out["action"] = "idle: fresh + gate ok (dispatch disabled)"
    print(json.dumps(out, indent=2))
    return out


def cmd_local_run(env, prompt, max_tokens=64, runner="local"):
    """Fully local single-machine run over persistent tier files (DEEPREPO_TIER_DIR)."""
    from . import ring
    from .worker import op_run
    if not getattr(env, "local_dir", None):
        raise SystemExit("set DEEPREPO_TIER_DIR to a persistent tier directory first")
    tok = env.tokenizer()
    ids = tok.chat_ids([{"role": "user", "content": prompt}])
    job_id = f"local{int(time.time())}"
    env.regs.put(f"job/{job_id}", {"id": job_id, "issue": 0, "prompt_ids": ids,
                                   "gen": {"max_tokens": max_tokens, "temp": 0.7, "top_k": 20,
                                           "top_p": 0.8, "seed": 42, "f16_cache_mb": 1856, "chunk": 192},
                                   "t": int(time.time())})
    env.regs.commit()
    env.runner_name = runner
    ring.Lease.grant(env.regs, runner)
    op_run(env, job_id, shift_minutes=600, shift_tokens=max_tokens)
    out = []
    for k in env.regs.keys(f"run/{job_id}/tok/"):
        out += env.regs.get(k)["ids"]
    print(tok.decode(out))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    arg = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    if cmd == "bootstrap":
        return cmd_bootstrap(argv)
    if cmd == "worker":
        from . import worker
        return worker.main(argv[1:])
    if cmd in ("serve", "once"):
        from . import coordinator
        return coordinator.main(argv)
    if cmd == "keeper":
        from .runtime import Env
        return cmd_keeper(Env(writer=True), loop="--loop" in argv,
                          interval=int(arg("--interval", 600)))
    if cmd == "load-model":
        from .runtime import Env
        return cmd_load_model(Env(writer=True), arg("--url"), arg("--key", "qwen3-30b-a3b-q4_k_m"),
                              resume="--fresh" not in argv)
    if cmd == "gate":
        from .runtime import Env
        return cmd_gate(Env(writer=True), "--fail" not in argv, arg("--note", ""))
    if cmd == "sweep":
        from .runtime import Env
        return cmd_sweep(Env(writer=True), arg("--op", "selftest"))
    if cmd == "local":
        from .runtime import Env
        return cmd_local_run(Env(writer=True), arg("--prompt", "Hello"),
                             int(arg("--max-tokens", 64)))
    if cmd == "status":
        from .runtime import Env
        return cmd_status(Env(writer=False))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())