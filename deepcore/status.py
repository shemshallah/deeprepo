"""Pinned status issue: a living board the coordinator/keeper edits at every handoff.

Shows: active ring slot, phase, uptime, API quota remaining, budget, last checkpoint
(compaction generation + journal seq), model, gate. Pauses work when the PAT quota is
80% used (20% remaining) and says so.
"""
import datetime
import json
import os
import threading
import time
import urllib.error
import urllib.request

LABEL = "deeprepo-status"
TITLE = "deeprepo status"


def _fmt_dur(secs):
    secs = max(0, int(secs))
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    return (f"{d}d " if d else "") + f"{h}h {m:02d}m"


class StatusBoard:
    """Issue-backed status board. Disabled automatically without a repo/token."""

    def __init__(s, repo, token, log=print):
        s.repo, s.token, s.log = repo, token, log
        s.enabled = bool(repo and "/" in repo and token)
        s.number, s.node_id, s.pinned = None, None, False
        s._lk = threading.Lock()
        s._quota = (None, 0.0)  # (remaining, fetched_at)
        s._limit = 5000

    # ---------- GitHub helpers ----------

    def _api(s, method, path, body=None):
        h = {"Accept": "application/vnd.github+json", "User-Agent": "deeprepo",
             "Authorization": f"Bearer {s.token}"}
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            h["Content-Type"] = "application/json"
        rq = urllib.request.Request(f"https://api.github.com{path}", data=data, headers=h, method=method)
        with urllib.request.urlopen(rq, timeout=60) as r:
            b = r.read()
        return json.loads(b) if b else None

    def _graphql(s, query, variables):
        rq = urllib.request.Request("https://api.github.com/graphql",
                                    data=json.dumps({"query": query, "variables": variables}).encode(),
                                    headers={"Authorization": f"Bearer {s.token}", "User-Agent": "deeprepo",
                                             "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(rq, timeout=60) as r:
            return json.loads(r.read())

    def ensure(s):
        if not s.enabled:
            return False
        if s.number:
            return True
        try:
            found = s._api("GET", f"/repos/{s.repo}/issues?labels={LABEL}&state=open&per_page=5")
            if found:
                s.number = found[0]["number"]
                s.node_id = found[0].get("node_id")
            else:
                issue = s._api("POST", f"/repos/{s.repo}/issues",
                               {"title": TITLE, "body": "starting…", "labels": [LABEL]})
                s.number, s.node_id = issue["number"], issue.get("node_id")
            if not s.pinned and s.number:
                s._pin()
            return True
        except Exception as e:
            s.log(f"status board disabled: {e!r}")
            s.enabled = False
            return False

    def _pin(s):
        try:
            s._api("PUT", f"/repos/{s.repo}/issues/{s.number}/pin")
            s.pinned = True
            return
        except Exception:
            pass
        if not s.node_id:
            return
        try:
            r = s._graphql("mutation($id:ID!){ pinIssue(input:{issueId:$id}){ issue { number pinned } } }",
                           {"id": s.node_id})
            s.pinned = bool((r.get("data") or {}).get("pinIssue"))
        except Exception:
            pass

    # ---------- quota ----------

    def quota(s, ttl=60):
        """(remaining, limit). /rate_limit is free; cached to avoid extra calls."""
        rem, at = s._quota
        if rem is not None and time.time() - at < ttl:
            return rem, s._limit
        try:
            d = s._api("GET", "/rate_limit")["resources"]["core"]
            s._limit = int(d.get("limit") or 5000)
            s._quota = (int(d.get("remaining") or 0), time.time())
            return s._quota[0], s._limit
        except Exception:
            return None, None

    def quota_low(s, pct_used=80):
        rem, lim = s.quota()
        if rem is None:
            return False
        return rem <= lim * (100 - pct_used) / 100.0

    # ---------- render ----------

    def render(s, env, runner=None, phase="idle", extra=""):
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        started = env.regs.get("ring/started")
        if not started:
            started = int(time.time())
            env.regs.put("ring/started", started)
        uptime = _fmt_dur(time.time() - (started or time.time()))
        try:
            slots = len(__import__("deepcore.ring", fromlist=["RING"]).RING)
        except Exception:
            slots = 5
        health = env.regs.get("ring/health") or {}
        alive = sum(1 for v in health.values() if (v or {}).get("ok")) or slots
        rem, lim = s.quota()
        b = env.budget.snapshot()
        gen = getattr(env.ram, "gen", 0)
        seq = getattr(env.ram, "seq", 0)
        mi = env.model_info() or {}
        gate = (env.regs.get("gate/single") or {}).get("ok")
        lease = env.regs.get("ring/lease") or {}
        holder = lease.get("holder")
        lines = [
            f"**Phase** {phase} · **Runner** {runner or holder or '—'} ({alive}/{slots} healthy)",
            f"**Uptime** {uptime} · **Gate** {'pass' if gate else 'pending'} · "
            f"**Model** `{mi.get('key') or 'none'}`",
            f"**API quota** {rem if rem is not None else '?'} / {lim if lim is not None else '?'} remaining"
            + (" ⚠️ paused (≥80% used)" if (rem is not None and lim and rem <= lim * 0.2) else ""),
            f"**Budget today** down {b['down']/2**30:.1f}/{b['caps']['down']/2**30:.0f} GiB · "
            f"up {b['up']/2**30:.1f}/{b['caps']['up']/2**30:.0f} GiB · "
            f"tokens {b['tokens']}/{b['caps']['tokens']}",
            f"**Checkpoint** v-ckpt-{gen} · **Journal seq** {seq} · **Lease** "
            f"`{holder or 'free'}` epoch {lease.get('epoch', 0)}",
        ]
        last = env.regs.get("stats/last") or {}
        if last.get("tok_per_s") is not None:
            lines.append(f"**Throughput** {last['tok_per_s']:.3f} tok/s · "
                         f"prefill {last.get('prefill_secs')}s · decode {last.get('decode_secs')}s · "
                         f"first token {last.get('first_token_secs')}s · paged {last.get('vram_bytes',0)/2**30:.1f} GiB")
        if extra:
            lines.append(extra)
        lines.append(f"_updated {now}_")
        return "\n\n".join(lines)

    def update(s, env, **kw):
        if not s.enabled or not s.ensure():
            return
        with s._lk:
            body = s.render(env, **kw)
            try:
                s._api("PATCH", f"/repos/{s.repo}/issues/{s.number}", {"body": body})
            except Exception as e:
                s.log(f"status update failed: {e!r}")

    def pause(s, env, minutes=30, why="API quota ≥80% used"):
        s.update(env, phase=f"paused — {why}; resuming in {minutes}m",
                 extra=f"⏸ sleeping {minutes} minutes to respect the API budget")
        time.sleep(minutes * 60)
