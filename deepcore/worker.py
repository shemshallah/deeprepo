"""CPU-runner side: python -m deepcore.worker prewarm|run|selftest|drain.

The runner owns the reporam write lease for its shift. It loads the job spec, hydrates
KV + handoff state through reporam, streams weights through repovram pages, writes
generated tokens back as journal deltas, and either completes or hands off.
"""
import json
import os
import sys
import time

from . import ring, PAGE


def _log(*a):
    print(*a, flush=True)


def op_prewarm(env, minutes_headstart=90):
    """Hydration gate: verify tiers + model, warm a small hot set, NO writes."""
    from .geom import read_super
    t0 = time.time()
    sb_r = read_super(env.ram)
    sb_v = read_super(env.vram)
    mi = env.model_info()
    assert mi and mi.get("ready"), "repovram has no ready model record"
    _log(f"prewarm: ram geom {env.ram.tier_geom} vram geom {env.vram.tier_geom}")
    _log(f"prewarm: model {mi.get('key')} {mi['size']/2**30:.2f} GiB, {mi['weight_pages']} pages, {len(mi.get('bins', []))} bins")
    # warm: header pages + first attention blocks of the first layers (bounded)
    from .gguf import GGUFOnRam
    g = GGUFOnRam(env.vram, 1, int(mi["size"]))
    warm_bytes = 0
    for name in ("token_embd.weight",):
        t = g.tensors[name]
        env.vram.prefetch(t.addr, min(t.nbytes, 32 << 20))
        warm_bytes += min(t.nbytes, 32 << 20)
    for l in range(min(4, int(g.meta["qwen3moe.block_count"]))):
        for nm in ("attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate_inp"):
            key = f"blk.{l}.{nm}.weight"
            if key in g.tensors:
                t = g.tensors[key]
                env.vram.prefetch(t.addr, t.nbytes)
                warm_bytes += t.nbytes
    total = time.time() - t0
    _log(f"prewarm: warmed ~{warm_bytes/2**20:.0f} MiB in {total:.0f}s; ready")
    env.ram.refresh()
    return {"ok": True, "warm_bytes": warm_bytes, "secs": round(total, 1)}


def op_run(env, job_id, shift_minutes=20, shift_tokens=64, max_shifts=1, log=_log):
    """Active shift: hold lease, load state through reporam, prefill/decode, publish tokens.

    Invariants:
      - KV rows for positions [0, len(prompt)+made) are already in reporam (idempotent rewrites OK).
      - `last_logits` = logits AFTER the most recently encoded position.
    """
    from . import ring
    import numpy as np
    regs = env.regs
    runner = env.runner_name
    try:
        return _run_shift(env, job_id, shift_minutes, shift_tokens, max_shifts, log)
    except BaseException:
        try:
            ring.Lease.free(regs, runner)
        except Exception:
            pass
        raise


def _run_shift(env, job_id, shift_minutes, shift_tokens, max_shifts, log):
    from . import ring
    import numpy as np
    regs = env.regs
    runner = env.runner_name
    lease = ring.Lease.assert_held(regs, runner)
    job = regs.get(f"job/{job_id}")
    if not job:
        raise IOError(f"job/{job_id} not found")
    gen = job.get("gen", {})
    rng = np.random.default_rng(gen.get("seed") or int(time.time()))
    model_key = job.get("model") or os.environ.get("CHAT_MODEL") or None
    tok = env.tokenizer(model_key)
    t0 = time.time()
    prev = []
    for key in regs.keys(f"run/{job_id}/tok/"):
        prev += regs.get(key)["ids"]
    prompt_ids = job["prompt_ids"]
    pos_prompt = len(prompt_ids)
    made = len(prev)
    made_start = made
    max_new = int(gen.get("max_tokens", 256))
    f16 = int(job.get("f16_cache_mb", 1856))
    tinfo = env.tune_vram_cache(model_key, f16)
    log(f"cache: {tinfo}")
    eng = env.engine(key=model_key, f16_cache_mb=f16, chunk=int(job.get("chunk", 256)))
    room = max(64, int(eng.ctx) - max_new - 8)
    if len(prompt_ids) > room:
        if made:
            raise IOError(f"job prompt {len(prompt_ids)} + {max_new} exceeds KV capacity {eng.ctx}")
        head = max(8, room // 4)
        prompt_ids = prompt_ids[:head] + prompt_ids[-(room - head):]
        job["prompt_ids"] = prompt_ids
        log(f"run: prompt truncated to {len(prompt_ids)} tokens to fit the KV region ({eng.ctx})")
    pos_prompt = len(prompt_ids)
    log(f"run: job {job_id} epoch {lease['epoch']} prompt {pos_prompt} prev {made} target {max_new}")
    t_prefill0 = time.time()
    if made == 0:
        # fresh job: chunked prefill, reusing the session's KV prefix when provided
        prefix = max(0, min(int(job.get("prefix_len", 0)), len(prompt_ids) - 1))
        if prefix:
            log(f"run: reusing KV prefix of {prefix}/{len(prompt_ids)} tokens")
        last_logits = None
        i = prefix
        while i < len(prompt_ids):
            chunk = prompt_ids[i:i + eng.chunk]
            last_logits = eng.forward(chunk, i)
            i += len(chunk)
            regs.commit()  # keeps KV + tails durable as prefill progresses
        if last_logits is None:  # no new suffix: re-encode the last token for logits
            last_logits = eng.forward([prompt_ids[-1]], len(prompt_ids) - 1)
    else:
        # continue: re-encode the last generated token (idempotent KV write) to get its logits
        last_logits = eng.forward([prev[-1]], pos_prompt + made - 1)

    t_prefill = time.time() - t_prefill0
    EOS = tok.special.get(tok.IM_END)
    # continue the chunk sequence across shifts: keys must be globally unique per job
    seq = len(regs.keys(f"run/{job_id}/tok/"))
    tokens, batch, done = [], [], False
    t_decode0 = time.time()
    t_first = None
    encoded = made  # how many generated tokens have KV + logits computed
    # WRITER RESIDENCY: one job keeps the lease across several internal shifts so its page
    # cache stays warm; only checkpoint deltas + informational handoff packets. The ring
    # rotates only when this job ends (lease freed) or its wall clock runs out.
    job_deadline = t0 + shift_minutes * 60 * max_shifts
    for shift_i in range(max(1, int(max_shifts))):
        if shift_i and not done:
            try:
                ring.Lease.renew(regs, runner, minutes=35)
            except PermissionError as e:
                log(f"run: lease lost, stopping shift loop: {e}")
                break
        shift_deadline = min(time.time() + shift_minutes * 60, job_deadline)
        made_shift = 0
        while made < max_new and time.time() < shift_deadline:
            nid = eng.sample(last_logits, temperature=gen.get("temp", 0.7), top_k=gen.get("top_k", 20),
                             top_p=gen.get("top_p", 0.8), penalty=gen.get("penalty", 1.05),
                             recent=tokens + prev, rng=rng)
            if nid == EOS:
                done = True
                break
            tokens.append(nid)
            made += 1
            made_shift += 1
            if t_first is None:
                t_first = time.time() - t_decode0
            batch.append(nid)
            if len(batch) >= 16:  # coalesce commits: fewer uploads under secondary limits
                regs.put(f"run/{job_id}/tok/{seq:04d}", {"ids": batch})
                seq += 1
                batch = []
                regs.commit()
            if made >= max_new:
                done = True
                break
            if made_shift >= shift_tokens:
                break  # internal checkpoint, same writer continues
            last_logits = eng.forward([nid], pos_prompt + made - 1)
            encoded = made
        if batch:
            regs.put(f"run/{job_id}/tok/{seq:04d}", {"ids": batch})
            seq += 1
            batch = []
            regs.commit()
        if done or time.time() >= job_deadline:
            break
        if made > encoded:  # shift ended before the last token's forward: do it now
            last_logits = eng.forward([tokens[-1]], pos_prompt + made - 1)
            encoded = made
        st_i = eng.stats()
        st_i["shift"] = shift_i + 1
        log(f"run: internal checkpoint {shift_i+1}/{max_shifts} at {made} tokens "
            f"(writer stays resident, lease held)")
    env.persist_budget()
    st = eng.stats()
    st["shift_secs"] = round(time.time() - t0, 1)
    st["prefill_secs"] = round(t_prefill, 1)
    st["decode_secs"] = round(time.time() - t_decode0, 1)
    st["first_token_secs"] = round(t_first, 1) if t_first is not None else None
    st["tok_per_s"] = round(max(0, made - made_start) / max(1e-6, st["decode_secs"]), 3)
    try:
        regs.put("stats/last", {"job": job_id, **st})
    except Exception:
        pass
    if done:
        regs.put(f"run/{job_id}/done", {"tokens": made, "stats": st})
        ring.Lease.free(regs, runner)
        log(f"run: job {job_id} complete: {made} tokens")
        log("STATS " + json.dumps(st))
    else:
        ring.handoff_packet(regs, job_id, runner, pos_prompt + made, last_tok=None, n_generated=made, stats=st)
        ring.Lease.free(regs, runner)
        log(f"run: shift end after {made} tokens; handoff packet published")
        log("STATS " + json.dumps(st))
    return {"made": made, "done": done}


def op_selftest(env):
    """No-network self test: fixture-store engine round trip (runs the shipped test suite subset)."""
    import subprocess
    r = subprocess.run([sys.executable, "tests/run_all.py", "--quick"], capture_output=True, text=True, timeout=1200)
    ok = r.returncode == 0
    _log(r.stdout[-4000:])
    if r.stderr:
        _log(r.stderr[-2000:])
    return {"ok": ok}


def op_chat(env, issue, model=None, max_tokens=48, log=_log):
    """Batch mode: answer one issue with local KV (no reporam/journal writes)."""
    from . import batch
    repo = env.main_repo or os.environ.get("GITHUB_REPOSITORY", "")
    owner = (repo.split("/")[0] if "/" in repo else os.environ.get("GITHUB_REPOSITORY_OWNER", ""))
    if not repo or "deeprepo-cpu-" in repo:      # batch jobs run in a CPU repo
        repo = f"{owner}/deeprepo" if owner else repo
    env.main_repo = repo
    log(f"chat: repo={repo} issue={issue} model={model or 'default'}")
    txt = batch.answer(env, issue, model_key=model or None, max_tokens=max_tokens, log=log)
    return {"ok": True, "chars": len(txt)}


OPS = {"prewarm": op_prewarm, "run": op_run, "selftest": op_selftest, "chat": op_chat}


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=list(OPS) + ["drain"])
    ap.add_argument("--job", default="")
    ap.add_argument("--shift-minutes", type=int, default=20)
    ap.add_argument("--shift-tokens", type=int, default=64)
    ap.add_argument("--max-shifts", type=int, default=1)
    ap.add_argument("--cache-pages", type=int, default=2048)
    ap.add_argument("--kv-cache-pages", type=int, default=256)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--issue", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args(argv)
    from .runtime import Env
    # clamp the page cache to what this VM can actually hold (public runners: 16 GiB)
    try:
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
        a.cache_pages = max(256, min(a.cache_pages, int((ram_gb - 5) * 256)))
        print(f"runner RAM {ram_gb:.1f} GiB -> page cache {a.cache_pages} pages ({a.cache_pages*4/1024:.1f} GiB)", flush=True)
    except Exception:
        pass
    env = Env(writer=(a.op == "run"), cache_pages=a.cache_pages, kv_cache_pages=a.kv_cache_pages)
    env.runner_name = __import__("os").environ.get("RING_SLOT") or "deeprepo-cpu-?"
    if a.op == "drain":
        env.ram.flush(max_pages=128)
        return 0
    if a.op == "chat":
        res = op_chat(env, a.issue, model=a.model or None, max_tokens=a.max_tokens)
    else:
        res = OPS[a.op](env, **({"job_id": a.job, "shift_minutes": a.shift_minutes,
                                  "shift_tokens": a.shift_tokens, "max_shifts": a.max_shifts}
                                 if a.op == "run" else {}))
    _log("RESULT", json.dumps(res)[:2000])
    return 0 if res.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))