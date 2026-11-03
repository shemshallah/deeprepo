"""Exercise the production import path (deepcore.tiers.import_model) offline:
fake HF reader -> tiled bins -> engine logits. Also checks resume does not re-upload.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import LocalBins, LocalJournal, TINY, dense_from_gguf, make_ram, make_tiny_gguf
from deepcore import PAGE
from deepcore.geom import plan_vram_layout
from deepcore.tiers import forge_bin0, import_model


class FakeHf:
    def __init__(s, data, etag='"abc"'):
        s.data, s.size, s.etag = data, len(data), etag
        s.reads = 0

    def read(s, off, n):
        s.reads += 1
        return s.data[off:off + n].ljust(n, b"\0")


def test_import_tiles_and_engine_reads():
    tmp = tempfile.mkdtemp()
    try:
        gguf = os.path.join(tmp, "tiny.gguf")
        make_tiny_gguf(gguf, seed=13)
        data = open(gguf, "rb").read()
        vdir = os.path.join(tmp, "vram")
        os.makedirs(vdir)
        forge_bin0(os.path.join(vdir, "mem-00.bin"), "vram", plan_vram_layout(16 * 510))
        store = LocalBins(vdir)
        store.tag = "vram"
        jdir = os.path.join(tmp, "j")
        jstore = LocalJournal(jdir)
        puts = []
        orig = store.put_file

        def counted(name, path):
            puts.append(name)
            return orig(name, path)

        store.put_file = counted
        res = import_model(store, jstore, FakeHf(data), resume=False, work=tmp)
        assert res["size"] == len(data)
        assert res["bins"] == [0], res  # tiny model fits in one bin
        assert puts == ["mem-00.bin"]
        # resume with the recorded progress must not re-upload
        res2 = import_model(store, jstore, FakeHf(data), resume=True, work=tmp)
        assert puts == ["mem-00.bin"], f"resume re-uploaded: {puts}"
        raw = open(os.path.join(vdir, "mem-00.bin"), "rb").read()
        assert raw[4 << 20:(4 << 20) + 6] == b"DPRSB1", "import must preserve/regenerate the superblock"
        # engine over the imported tier matches the dense reference
        from deepcore.engine import Qwen3MoE
        ram_j = os.path.join(tmp, "journal")
        ram_dir = os.path.join(tmp, "ram")
        os.makedirs(ram_dir)
        from deepcore.geom import plan_ram_layout
        forge_bin0(os.path.join(ram_dir, "mem-00.bin"), "ram", plan_ram_layout(16 * 510, kv_pages=64, reg_pages=8))
        vram = make_ram(vdir, cache_pages=16)
        ram = make_ram(ram_dir, journal_dir=ram_j, writer=True, cache_pages=32)
        from deepcore.geom import read_super
        lay = read_super(ram)["layout"]
        kv = {"base": lay["kv"][0] * PAGE, "bytes": lay["kv"][1] * PAGE}
        meta, w = dense_from_gguf(gguf)
        eng = Qwen3MoE(vram, 1, len(data), ramkv=ram, kv_layout=kv, f16_cache_mb=4, strict=True)
        ids = [3, 14, 15, 92]
        got = eng.forward(ids, 0)
        # dense reference via the test's independent implementation
        from test_engine import dense_forward
        ref = dense_forward(w, dict(TINY), ids, 0)
        assert np.allclose(got, ref, atol=1e-4, rtol=1e-5)
        print(f"  import -> {res['bins']} bin(s), resume idempotent, engine matches dense")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_import_tiles_and_engine_reads()
    print("import tests passed")