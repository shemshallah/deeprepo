"""The 5-slot CPU ring: leases, prewarm/hydrate handoff, dispatch to deeprepo-cpu-N.

Single-writer discipline: reporam's journal accepts one writer at a time, guarded by a
lease record ("ring/lease"). Handoff order: runner drains -> publishes handoff packet +
frees lease -> coordinator prewarms (hydrates) the NEXT runner -> grants lease -> new
runner continues from the packet. Nobody is handed work before hydration succeeds.
"""
import json
import time
import urllib.request

from . import BINS, PAGE, PPS


class Lease:
    """Lease record stored in the reporam registers."""

    @staticmethod
    def grant(regs, holder, minutes=45):
        epoch = ((regs.get("ring/lease") or {}).get("epoch") or 0) + 1
        rec = {"holder": holder, "epoch": epoch, "until": time.time() + minutes * 60, "t": int(time.time())}
        regs.put("ring/lease", rec)
        regs.commit()
        return rec

    @staticmethod
    def free(regs, holder, epoch=None):
        cur = regs.get("ring/lease") or {}
        if cur.get("holder") == holder and (epoch is None or cur.get("epoch") == epoch):
            regs.put("ring/lease", {"holder": None, "epoch": cur.get("epoch", 0), "until": 0, "freed": int(time.time())})
            regs.commit()

    @staticmethod
    def renew(regs, holder, minutes=30):
        cur = regs.get("ring/lease") or {}
        if cur.get("holder") != holder:
            raise PermissionError(f"cannot renew: holder is {cur.get('holder')}")
        cur["until"] = time.time() + minutes * 60
        cur["renewed"] = int(time.time())
        regs.put("ring/lease", cur)
        regs.commit()
        return cur

    @staticmethod
    def assert_held(regs, holder, slack=30.0):
        cur = regs.get("ring/lease") or {}
        if cur.get("holder") != holder or cur.get("until", 0) < time.time() - slack:
            raise PermissionError(f"lease not held: {holder} vs {cur}")
        return cur


def handoff_packet(regs, job_id, runner, pos, last_tok, n_generated, stats=None):
    """What a finishing runner leaves for the next one. KV itself is already in reporam."""
    pkt = {"job": job_id, "from": runner, "pos": pos, "last_tok": last_tok,
           "n_generated": n_generated, "t": int(time.time()), "stats": stats or {}}
    regs.put(f"ring/packet/{job_id}", pkt)
    return pkt


RING = [f"deeprepo-cpu-{i}" for i in range(1, 6)]


def next_runner(regs, current=None):
    """Round-robin over ring slots that reported healthy in the last hydration sweep."""
    health = regs.get("ring/health") or {}
    alive = [r for r in RING if (health.get(r) or {}).get("ok")]
    if not alive:
        alive = list(RING)
    if current in alive and len(alive) > 1:
        alive.remove(current)
    return alive[int(time.time()) % len(alive)]


def dispatch_workflow(repo, workflow, token, inputs=None, ref="main"):
    """workflow_dispatch with a PAT (actions:write). Used for prewarm/run/selftest."""
    rq = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
        json.dumps({"ref": ref, "inputs": inputs or {}}).encode(),
        {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
         "User-Agent": "deeprepo"}, method="POST")
    urllib.request.urlopen(rq, timeout=30).read()


def last_run(repo, workflow, token):
    rq = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?per_page=1",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "User-Agent": "deeprepo"})
    with urllib.request.urlopen(rq, timeout=30) as r:
        js = json.loads(r.read())
    return (js.get("workflow_runs") or [None])[0]


def await_run(repo, workflow, token, timeout=1500, poll=15, log=print):
    """Wait for the newest run of a workflow to finish; returns conclusion or None."""
    t0 = time.time()
    base = last_run(repo, workflow, token)
    base_id = base["id"] if base else None
    while time.time() - t0 < timeout:
        time.sleep(poll)
        run = last_run(repo, workflow, token)
        if not run:
            continue
        if base_id is not None and run["id"] == base_id:
            continue  # the previous run, not the one we dispatched
        if run.get("conclusion"):
            return run["conclusion"]
    return None