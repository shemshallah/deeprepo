#!/usr/bin/env python3
"""Offline test runner (no network, no credentials): dequant vs ggml, GGUF, tokenizer,
meter, sparse tiers, engine-vs-dense, handoff, compaction.  Usage: python tests/run_all.py [--quick]"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODS = ["test_dequant.py", "test_units.py", "test_import.py", "test_engine.py", "test_ring.py", "test_local_tiers.py", "test_disk_cache.py", "test_middle.py", "test_status.py"]


def main(argv):
    quick = "--quick" in argv
    env = dict(os.environ, PYTHONPATH=ROOT)
    fails, passed = [], []
    for m in MODS:
        print(f"== {m} ==", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, m)],
                          cwd=ROOT, env=env, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-4000:])
            fails.append(m)
        else:
            passed.append(m)
    print(f"\n{len(passed)}/{len(MODS)} test files passed" + (f"; FAILED: {fails}" if fails else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))