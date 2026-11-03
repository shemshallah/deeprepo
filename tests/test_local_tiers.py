"""Persistent local tiers: bins live as files in the repo tree (gitignored) and are modified
in place. This is the mode for a self-hosted runner or workstation — the "memory register on
the fly, in-repo" variant — as opposed to hosted runners, whose VMs are wiped after each job.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

from deepcore import PAGE
from deepcore.geom import plan_ram_layout, read_super
from deepcore.store import connect_local_tier, local_bootstrap
from deepcore.engine import Qwen3MoE
from deepcore.tiers import compact_trimmed

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import TINY, dense_from_gguf, make_tiny_gguf, tile_model
from test_engine import dense_forward


def test_local_tiers_engine_and_inplace_writes():
    root = tempfile.mkdtemp()
    try:
        out = local_bootstrap(root, kv_pages=64, reg_pages=8)
        assert out["ram"] == "created" and out["vram"] == "created"
        # bins are real files, instantly sparse-ish (8 MiB bin-0 each)
        for p in (os.path.join(root, "vram", "mem-00.bin"), os.path.join(root, "ram", "mem-00.bin")):
            assert os.path.exists(p) and os.path.getsize(p) == 8 << 20
        gguf = os.path.join(root, "tiny.gguf")
        make_tiny_gguf(gguf, seed=31)
        size = os.path.getsize(gguf)
        tile_model(gguf, os.path.join(root, "vram"), None)
        vram = connect_local_tier(os.path.join(root, "vram"), cache_pages=16)
        ram = connect_local_tier(os.path.join(root, "ram"), journal_dir=os.path.join(root, "ram-j"),
                                 writer=True, cache_pages=32)
        lay = read_super(ram)["layout"]
        kv = {"base": lay["kv"][0] * PAGE, "bytes": lay["kv"][1] * PAGE}
        eng = Qwen3MoE(vram, 1, size, ramkv=ram, kv_layout=kv, f16_cache_mb=4, strict=True)
        meta, w = dense_from_gguf(gguf)
        ids = [6, 30, 41]
        got = eng.forward(ids, 0)
        assert np.allclose(got, dense_forward(w, dict(TINY), ids, 0), atol=1e-4, rtol=1e-5)
        # KV write-back is an actual file in the tier tree
        ram.flush()
        deltas = [f for f in os.listdir(os.path.join(root, "ram-j")) if f.startswith("d-")]
        assert deltas, "expected a journal delta file"
        # compaction rewrites bins in place; a fresh reader still sees the KV
        compact_trimmed(ram, work=root)
        reader = connect_local_tier(os.path.join(root, "ram"), journal_dir=os.path.join(root, "ram-j"))
        assert reader.read(kv["base"], 128) == ram.read(kv["base"], 128)
        gens = [f for f in os.listdir(os.path.join(root, "ram")) if ".g" in f]
        assert gens, f"expected a compacted generation bin, got {os.listdir(os.path.join(root,'ram'))}"
        print(f"  local tiers: bins in-tree, delta {deltas[0]}, compacted {gens[0]}")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_local_tiers_engine_and_inplace_writes()
    print("local tier tests passed")