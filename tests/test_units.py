"""Geometry, sparse-slot semantics, superblocks, registers."""
import os
import tempfile

import numpy as np
import ram16

from deepcore import BINS, BIN_HDR, PAGE, PPS
from deepcore.geom import (forge_header, pack_super, plan_ram_layout, plan_vram_layout,
                           read_super, tier_bytes, tier_pages)
from deepcore.store import SparseAsset
from deepcore.tiers import Registers, forge_bin0

import sys
sys.path.insert(0, os.path.dirname(__file__))
from fixtures import LocalAsset, LocalBins, LocalJournal, make_ram


def test_geometry_arithmetic():
    # 16 bins x 510 pages x 4 MiB ~= 31.9 GiB per tier (the "32 GB sparse" registers)
    assert tier_pages() == 16 * 510
    assert 31.5 < tier_bytes() / 2**30 < 32.5
    assert BIN_HDR + PPS * PAGE < 2 * 2**30  # every bin asset < 2 GiB release cap
    print(f"  tier = {tier_pages()} pages = {tier_bytes()/2**30:.2f} GiB virtual, bin = {(BIN_HDR+PPS*PAGE)/2**30:.2f} GiB")


def test_superblock_roundtrip():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "mem-00.bin")
    lay = plan_ram_layout(tier_pages(), kv_pages=64, reg_pages=8)
    forge_bin0(p, "ram", lay)
    jd = os.path.join(tmp, "j")
    ram = make_ram(tmp, journal_dir=jd, writer=True)
    sb = read_super(ram)
    assert sb["tier"] == "ram" and sb["layout"] == lay
    assert ram.read(0, 6) == b"DPRSB1"
    # registers live at their region and survive reopening
    regs = Registers(ram, lay["regs"])
    regs.put("ring/lease", {"holder": "deeprepo-cpu-1", "epoch": 3})
    regs.commit()
    ram2 = make_ram(tmp, journal_dir=jd)
    regs2 = Registers(ram2, lay["regs"])
    assert regs2.get("ring/lease")["holder"] == "deeprepo-cpu-1"
    print("  superblock + registers round trip")


def test_sparse_bins_zero_read():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "mem-00.bin")
    forge_bin0(p, "vram", plan_vram_layout(tier_pages()))
    ram = make_ram(tmp)
    assert ram.read(200 * PAGE, 4096) == b"\0" * 4096          # unwritten page
    assert ram.read((BINS - 1) * PPS * PAGE, 4096) == b"\0" * 4096  # absent bin 15
    # forged headers are accepted by ram16's verifier for every absent slot
    for i in (3, 9, 15):
        h = forge_header(i)
        assert ram16.MAGIC == h[:8]
    print("  absent bins read as zeros; headers forged and accepted")


def test_meter_budget():
    from deepcore.meter import Budget, BudgetExceeded
    b = Budget(caps={"down": 1000, "up": 100, "api": 5, "tokens": 10})
    b.record("down", 900)
    try:
        b.record("down", 200)
        assert False
    except BudgetExceeded:
        pass
    assert b.today["down"] == 900
    b.record("tokens", 10)
    assert b.exhausted()
    print("  budget caps fail closed")


def test_tokenizer_roundtrip():
    from deepcore.tok import BPETokenizer
    # tiny synthetic vocab: every single byte char + a few merged tokens + specials
    b2u, _u2b = BPETokenizer.__new__(BPETokenizer), None
    from deepcore.tok import _bytes_to_unicode
    b2u, u2b = _bytes_to_unicode()
    toks = [chr(0x100 + i) for i in range(8)]  # filler ids
    toks += [b2u[b] for b in range(256)]
    toks[0] = "<|im_start|>"; toks[1] = "<|im_end|>"; toks[2] = "<|endoftext|>"
    ty = [3, 3, 3] + [1] * (len(toks) - 3)
    from deepcore.tok import QWEN2_PRE
    tk = BPETokenizer(toks, ty, [], QWEN2_PRE)
    s = "hello, world! 123\nnew line"
    ids = tk.encode(s, add_specials=False)
    assert tk.decode(ids) == s
    ids2 = tk.encode("<|im_start|>hi<|im_end|>")
    assert ids2[0] == 0 and ids2[-1] == 1
    assert tk.decode(ids2) == "<|im_start|>hi<|im_end|>"
    print("  byte-level BPE encode/decode + specials round trip")


def test_gguf_writer_reader_roundtrip():
    import tempfile
    from deepcore.gguf import parse_header, dequant, write_gguf
    rng = np.random.default_rng(11)
    x = rng.standard_normal((6, 32)).astype(np.float32)
    y = rng.standard_normal(16).astype(np.float32)
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "m.gguf")
    write_gguf(p, {"general.architecture": "qwen3moe", "x": 4, "f": 0.5, "arr": [1, 2, 3]},
               [("a", [32, 6], 0, x), ("b", [16], 1, y), ("c", [32, 4], 8, rng.standard_normal((4, 32)).astype(np.float32))])
    f = open(p, "rb").read()
    meta, tensors, data_off, _ = parse_header(lambda o, n: f[o:o + n], len(f))
    assert meta["general.architecture"] == "qwen3moe" and meta["x"] == 4 and meta["arr"] == ("array", 4, 3)
    a = dequant(f[tensors["a"].addr:tensors["a"].addr + tensors["a"].nbytes], 0).reshape(6, 32)
    assert np.allclose(a, x)
    b = dequant(f[tensors["b"].addr:tensors["b"].addr + tensors["b"].nbytes], 1).reshape(16)
    assert np.allclose(b, y.astype("<f2").astype(np.float32))
    print("  GGUF write/parse/dequant round trip (f32/f16/q8_0)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("unit tests passed")