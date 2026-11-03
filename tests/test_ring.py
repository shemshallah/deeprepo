"""Ring protocol integration test (offline): coordinator job -> lease -> worker shift ->
handoff -> next runner -> completion. Compares token ids against a single continuous shift.

Uses the production Coordinator.run_job / worker.op_run / Lease / Registers code over
file-backed tiers; only the GitHub dispatch is replaced by an in-process call.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

from deepcore import PAGE
from deepcore.geom import plan_ram_layout, read_super
from deepcore.meter import Budget, ApiMeter
from deepcore.tiers import Registers, forge_bin0
from deepcore import worker, ring
from deepcore.runtime import Env

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import TINY, make_ram, make_tiny_gguf, tile_model


class FixtureEnv(Env):
    """Env wired to local fixture tiers instead of GitHub. Reuses all production logic."""

    def __init__(s, tmp, writer, name, size, wp):
        s.token = ""
        s.writer = writer
        s.ram_repo, s.vram_repo, s.main_repo = "reporam", "repovram", "deeprepo"
        s.runner_name = name
        s.vram = make_ram(os.path.join(tmp, "vram"), cache_pages=16)
        s.ram = make_ram(os.path.join(tmp, "ram"), journal_dir=os.path.join(tmp, "j"),
                         writer=writer, cache_pages=64)
        lay = read_super(s.ram)["layout"]
        s.regs = Registers(s.ram, lay["regs"])
        s.budget = Budget()
        s.api = ApiMeter()
        s._model = {"ready": True, "size": size, "weight_pages": wp, "key": "tiny"}

    def model_info(s):
        return s._model


def setup(tmp):
    gguf = os.path.join(tmp, "tiny.gguf")
    make_tiny_gguf(gguf, seed=21)
    size = os.path.getsize(gguf)
    vdir = os.path.join(tmp, "vram")
    os.makedirs(vdir, exist_ok=True)
    forge_bin0(os.path.join(vdir, "mem-00.bin"), "vram", {"weights": [1, 16 * 510 - 1]})
    wp = tile_model(gguf, vdir, None)
    rdir = os.path.join(tmp, "ram")
    os.makedirs(rdir, exist_ok=True)
    forge_bin0(os.path.join(rdir, "mem-00.bin"), "ram", plan_ram_layout(16 * 510, kv_pages=64, reg_pages=8))
    return size, wp


def make_job(env, prompt_ids, max_tokens=6):
    job_id = "jtest"
    env.regs.put(f"job/{job_id}", {"id": job_id, "issue": 0, "prompt_ids": prompt_ids,
                                   "gen": {"max_tokens": max_tokens, "temp": 0.0, "seed": 7,
                                           "f16_cache_mb": 4, "chunk": 8},
                                   "t": 0})
    env.regs.commit()
    return job_id


def run_single(tmp, size, wp, prompt, max_tokens):
    """One continuous shift (baseline)."""
    env = FixtureEnv(tmp, writer=True, name="deeprepo-cpu-1", size=size, wp=wp)
    job_id = make_job(env, prompt, max_tokens)
    ring.Lease.grant(env.regs, "deeprepo-cpu-1")
    worker.op_run(env, job_id, shift_minutes=5, shift_tokens=max_tokens)
    out = []
    for k in env.regs.keys(f"run/{job_id}/tok/"):
        out += env.regs.get(k)["ids"]
    return out


def run_ring(tmp, size, wp, prompt, max_tokens, shift=3):
    """Coordinator drives two shifts; the second runner continues from reporam state."""
    coord_env = FixtureEnv(tmp, writer=True, name="coordinator", size=size, wp=wp)
    from deepcore.coordinator import Coordinator
    c = Coordinator(coord_env)
    c.prewarm = lambda repo, timeout=0, op="prewarm": True   # no dispatch; hydration trivially ok
    calls = []

    def fake_dispatch(repo, op, job=None, minutes=20):
        calls.append((repo, op, job))
        if op == "run":
            runner_env = FixtureEnv(tmp, writer=True, name=repo, size=size, wp=wp)
            worker.op_run(runner_env, job, shift_minutes=5, shift_tokens=shift)

    c._dispatch = fake_dispatch
    job_id = make_job(coord_env, prompt, max_tokens)
    seen = []

    def on_tok(t):
        seen.append(t)

    done, text = c.run_job(job_id, deadline=__import__("time").time() + 60,
                           on_token=on_tok, max_shift_minutes=5)
    ids = []
    for k in coord_env.regs.keys(f"run/{job_id}/tok/"):
        ids += coord_env.regs.get(k)["ids"]
    packet = coord_env.regs.get(f"ring/packet/{job_id}") or {}
    return done, ids, calls, packet


def test_ring_handoff_matches_single_shift():
    tmp = tempfile.mkdtemp()
    try:
        size, wp = setup(tmp)
        prompt = [7, 12, 40]
        # baseline: one continuous shift of 6 tokens
        t2 = tempfile.mkdtemp()
        size2, wp2 = setup(t2)
        base = run_single(t2, size2, wp2, prompt, 6)
        shutil.rmtree(t2)
        # ring: 2 shifts of 3 tokens each, separate writer instances
        done, ids, calls, packet = run_ring(tmp, size, wp, prompt, 6, shift=3)
        runs = [c for c in calls if c[1] == "run"]
        assert done, "ring did not complete"
        assert len(runs) >= 2, f"expected a handoff, got calls {calls}"
        assert ids == base, f"ring tokens {ids} != continuous {base}"
        print(f"  ring: {len(runs)} shifts, handoff packet {packet.get('n_generated')} tokens, ids match continuous run {ids}")
    finally:
        shutil.rmtree(tmp)


def test_writer_residency_one_dispatch():
    """max_shifts>1: a single dispatched job runs several internal shifts, keeps the lease,
    and still reproduces the continuous token stream (cold-start amortization)."""
    tmp = tempfile.mkdtemp()
    try:
        size, wp = setup(tmp)
        prompt = [7, 12, 40]
        env = FixtureEnv(tmp, writer=True, name="deeprepo-cpu-1", size=size, wp=wp)
        job_id = make_job(env, prompt, 6)
        ring.Lease.grant(env.regs, "deeprepo-cpu-1")
        calls_before = len(env.regs.keys(f"run/{job_id}/tok/"))
        worker.op_run(env, job_id, shift_minutes=5, shift_tokens=2, max_shifts=3)
        ids = []
        for k in env.regs.keys(f"run/{job_id}/tok/"):
            ids += env.regs.get(k)["ids"]
        # baseline via one full shift on a separate copy
        t2 = tempfile.mkdtemp()
        size2, wp2 = setup(t2)
        base = run_single(t2, size2, wp2, prompt, 6)
        shutil.rmtree(t2)
        assert ids == base, f"resident tokens {ids} != baseline {base}"
        packets = env.regs.keys(f"ring/packet/{job_id}")
        assert not packets, "internal checkpoints must not look like a finished handoff"
        assert env.regs.get(f"run/{job_id}/done"), "job should complete inside one job"
        assert not (env.regs.get("ring/lease") or {}).get("holder"), "lease must be freed at job end"
        print(f"  residency: one dispatch, 3 internal shifts, ids match continuous run {ids}")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_ring_handoff_matches_single_shift()
    test_writer_residency_one_dispatch()
    print("ring protocol test passed")