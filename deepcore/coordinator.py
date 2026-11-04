"""Coordinator (deeprepo): chat intake, ring orchestration, streaming, budget guard.

The coordinator NEVER runs the model. It tokenizes, writes job specs, prewarms a runner
(hydration), grants the write lease, dispatches the run, tails token records out of
reporam, streams the reply, and rotates to the next runner at shift boundaries.
"""
import json
import os
import sys
import time

import ram16

from . import ring
from .tok import apply_no_think
from .meter import BudgetExceeded
from .status import StatusBoard

POLL = 6.0


class Coordinator:
    def __init__(s, env, log=print):
        s.env, s.log = env, log
        s.health = env.regs.get("ring/health") or {}
        s.ring = ring.ring_slots(env.main_repo)
        s.token = env.token
        s.status = StatusBoard(env.main_repo, env.token, log=log)

    # ---------------- ring plumbing ----------------

    def _dispatch(s, repo, op, job=None, minutes=20):
        ring.dispatch_workflow(repo, "runner.yml", s.token, inputs={
            "op": op, "job": job or "", "minutes": str(minutes)})

    def prewarm(s, repo, timeout=1500, op="prewarm"):
        s.log(f"{op} {repo} ...")
        try:
            base = ring.last_run(repo, "runner.yml", s.token)
            base_id = base["id"] if base else None
            s._dispatch(repo, op)
        except Exception as e:
            s.log(f"dispatch failed for {repo}: {e!r}")
            return False
        c = ring.await_run(repo, "runner.yml", s.token, timeout=timeout, poll=15, base_id=base_id)
        ok = c == "success"
        s.health[repo] = {"ok": ok, "t": int(time.time()), "conclusion": c, "mode": op}
        s.status.update(s.env, runner=repo, phase=f"{op} {'ok' if ok else 'failed'}")
        h = dict(s.env.regs.get("ring/health") or {})
        h[repo] = s.health[repo]
        s.env.regs.put("ring/health", h)
        if not (s.env.regs.get("ring/lease") or {}).get("holder"):
            s.env.regs.commit()
        s.log(f"{op} {repo}: {c}")
        return ok

    def pick(s, current=None, exclude=()):
        alive = [r for r in s.ring if (s.health.get(r) or {}).get("ok") and r not in exclude]
        if not alive:
            alive = [r for r in s.ring if r not in exclude]
        if not alive:
            alive = list(s.ring)
        if current in alive and len(alive) > 1:
            alive.remove(current)
        return alive[int(time.time()) // 60 % len(alive)]

    # ---------------- the job pipeline ----------------

    def run_job(s, job_id, deadline, first_runner=None, on_token=None, max_shift_minutes=25):
        """Drive a job across as many ring shifts as needed. Returns (done, text)."""
        env, regs = s.env, s.env.regs
        tok = env.tokenizer()
        seen, text, failed = set(), "", set()
        current = first_runner
        while time.time() < deadline:
            if s.status.enabled and s.status.quota_low():
                s.status.pause(s.env, minutes=int(os.environ.get("QUOTA_PAUSE_MINUTES", "30")))
            runner = current if (current and current not in failed) else s.pick(current, exclude=failed)
            if not s.prewarm(runner):
                failed.add(runner)
                current = None
                if len(failed) >= len(s.ring):
                    break
                continue
            ring.Lease.grant(regs, runner, minutes=max_shift_minutes + 10)  # renewed per internal shift
            s.status.update(s.env, runner=runner, phase=f"shift {job_id}")
            s._dispatch(runner, "run", job=job_id, minutes=max_shift_minutes)
            current = runner
            last_tick = 0.0
            while True:
                env.ram.refresh()   # tail the runner's journal deltas (writer connection)
                regs.refresh()
                added = False
                for key in regs.keys(f"run/{job_id}/tok/"):
                    if key in seen:
                        continue
                    seen.add(key)
                    chunk = regs.get(key)["ids"]
                    text += tok.decode(chunk)
                    env.budget.spend("tokens", len(chunk))
                    added = True
                if added and on_token and time.time() - last_tick >= 8.0:
                    on_token(text)
                    last_tick = time.time()
                if regs.get(f"run/{job_id}/done"):
                    if on_token:
                        on_token(text)
                    s.status.update(s.env, runner=runner, phase="idle — reply complete")
                    return True, text
                lease = regs.get("ring/lease") or {}
                expired = lease.get("until", 0) < time.time() - 60
                if (lease.get("holder") is None or expired) and time.time() - last_tick > 2.0 and (
                        seen or regs.get(f"ring/packet/{job_id}")):
                    s.status.update(s.env, runner=runner, phase="handoff")
                    break  # job ended (or the writer died); rotate
                if time.time() >= deadline:
                    return False, text
                time.sleep(POLL)
            current = None
        return False, text

    # ---------------- chat ----------------

    def serve_issue(s, req, minutes=300, max_tokens=None):
        env, regs = s.env, s.env.regs
        if env.budget.exhausted():
            raise BudgetExceeded("daily budget exhausted")
        model_key = os.environ.get("CHAT_MODEL") or None
        mi = env.model_info(model_key) or {}
        f16 = 8000 if 0 < int(mi.get("size", 1)) < (6 << 30) else 1856
        tok = env.tokenizer(model_key)
        if max_tokens is None:
            max_tokens = int(os.environ.get("MAX_REPLY_TOKENS", "48"))
        ids = tok.chat_ids(apply_no_think(req.messages, model_key))
        active = regs.get(f"sess/active/{req.id}")
        prefix_len = 0
        if not active:
            prev = (regs.get(f"sess/ctx/{req.id}") or {}).get("ids") or []
            m = min(len(prev), len(ids))
            while prefix_len < m and prev[prefix_len] == ids[prefix_len]:
                prefix_len += 1
            prefix_len = min(prefix_len, max(0, m - 1))
            if prefix_len < 16:
                prefix_len = 0
        job = regs.get(f"job/{active['job']}") if active else None
        if job and not regs.get(f"run/{active['job']}/done"):
            job_id, ids = active["job"], job.get("prompt_ids", ids)
            s.log(f"issue {req.id}: resuming job {job_id}")
        else:
            job_id = f"j{int(time.time())}"
            left = max(1, min(max_tokens, env.budget.caps["tokens"] - env.budget.today["tokens"]))
            regs.put(f"job/{job_id}", {
                "id": job_id, "issue": req.id, "prompt_ids": ids,
                "gen": {"max_tokens": left, "temp": 0.7, "top_k": 20, "top_p": 0.8,
                        "penalty": 1.05, "seed": int(time.time()) % 2**31,
                        "f16_cache_mb": f16, "chunk": 192},
                "model": model_key,
                "prefix_len": prefix_len,
                "t": int(time.time())})
            regs.put(f"sess/active/{req.id}", {"job": job_id, "t": int(time.time())})
            regs.commit()
        q = s.queue
        handle = q.begin(req)
        done, text = s.run_job(job_id, time.time() + minutes * 60,
                               on_token=lambda t: q.update(handle, t or "…"))
        if done:
            q.finish(handle, text or "(no output)")
            final = regs.get(f"run/{job_id}/done") or {}
            gen_ids = []
            for k in regs.keys(f"run/{job_id}/tok/"):
                gen_ids += regs.get(k)["ids"]
            regs.put(f"sess/ctx/{req.id}", {"ids": ids + gen_ids, "t": int(time.time())})
            regs.put(f"sess/{req.id}/{job_id}", {"q": (req.messages[-1]["content"] or "")[:2000],
                                                 "a": text[:8000], "stats": final.get("stats")})
            if (regs.get(f"sess/active/{req.id}") or {}).get("job") == job_id:
                regs.delete(f"sess/active/{req.id}")
            regs.commit()
        else:
            # keep the comment unfinished (▌) and the active job; the next dispatch resumes it
            q.update(handle, text + "\n\n⏳ continuing on the next dispatch…")
        env.persist_budget()
        env.safe_commit()
        return text

    def serve(s, minutes=300, compact_at=48):
        env = s.env
        q = s.queue = ram16.IssueQueue(env.main_repo, s.token, "deeprepo-chat")
        deadline = time.time() + minutes * 60
        env.regs.refresh()
        s.status.update(env, phase="serving — waiting for an issue")
        while time.time() < deadline - 60:
            if s.status.enabled and s.status.quota_low():
                s.status.pause(env, minutes=int(os.environ.get("QUOTA_PAUSE_MINUTES", "30")))
            if env.budget.exhausted():
                s.log("budget exhausted; pausing serves")
                break
            req = q.next()
            if req is None:
                time.sleep(20)
                continue
            t1 = time.time()
            try:
                s.serve_issue(req, minutes=min(240, max(30, int(deadline - time.time() - 60))))
            except BudgetExceeded as e:
                q.finish(req.handle or q.begin(req), f"⏸ budget cap reached ({e}); the keeper resumes me on the next window.")
            except Exception as e:
                try:
                    q.finish(req.handle or q.begin(req), f"⚠ error: {type(e).__name__}: {e}")
                except Exception:
                    pass
                s.log(f"issue {req.id} failed: {e!r}")
            if len(env.ram.journal.list()) >= compact_at and not (env.regs.get("ring/lease") or {}).get("holder"):
                try:
                    from .tiers import compact_trimmed
                    compact_trimmed(env.ram, log=s.log)
                except Exception as e:
                    s.log(f"compact failed: {e!r}")
            s.log(f"served issue {req.id} in {time.time()-t1:.1f}s")
        env.persist_budget()
        env.safe_commit()


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["serve", "once"])
    ap.add_argument("--minutes", type=int, default=300)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--runner", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args(argv)
    from .runtime import Env
    env = Env(writer=True)
    env.main_repo = env.main_repo or os.environ.get("GITHUB_REPOSITORY", "")
    c = Coordinator(env)
    try:
        if a.op == "serve":
            c.serve(minutes=a.minutes)
        else:
            model_key = os.environ.get("CHAT_MODEL") or None
            tok = env.tokenizer(model_key)
            ids = tok.chat_ids(apply_no_think([{"role": "user", "content": a.prompt or "Hello"}], model_key))
            job_id = f"j{int(time.time())}"
            env.regs.put(f"job/{job_id}", {"id": job_id, "issue": 0, "prompt_ids": ids,
                                           "model": model_key,
                                           "gen": {"max_tokens": a.max_tokens, "seed": int(time.time()) % 2**31,
                                                   "f16_cache_mb": 1856, "chunk": 192},
                                           "t": int(time.time())})
            env.regs.commit()
            done, text = c.run_job(job_id, time.time() + a.minutes * 60,
                                   first_runner=a.runner or None,
                                   on_token=lambda t: print(t, flush=True))
            print(("\n" if text else "") + ("DONE" if done else "TIMEOUT"), flush=True)
            env.persist_budget()
            if not done:
                sys.exit(3)
    finally:
        print("BUDGET_EXHAUSTED" if env.budget.exhausted() else "BUDGET_OK", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])