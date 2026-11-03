"""Job-local SSD page cache: in-place files created by the runner during a shift.
Proves the cache is read-through, byte-correct across page boundaries, survives source
loss, and is LRU-trimmed to its cap. This is L0.5 (ephemeral); hosted VMs discard it.
"""
import os
import shutil
import struct
import sys
import tempfile

from deepcore import BIN_HDR, BINS, PAGE, PPS
from deepcore.geom import forge_header
from deepcore.store import CachedAsset, DiskCache, LocalAsset, LocalTierStore, SparseAsset


def make_bin(path, pages=4, seed=b"x"):
    with open(path, "wb") as f:
        f.truncate(BIN_HDR + pages * PAGE)
        f.seek(0)
        f.write(forge_header(0))
        f.seek(BIN_HDR)
        rng = __import__("numpy").random.default_rng(7)
        f.write(rng.integers(0, 256, pages * PAGE, dtype="uint8").tobytes())
    return os.path.getsize(path)


def test_read_through_and_page_split():
    root = tempfile.mkdtemp()
    try:
        p = os.path.join(root, "mem-00.bin")
        size = make_bin(p)
        cache = DiskCache(os.path.join(root, "cache"), cap_bytes=64 << 20)
        a = CachedAsset(LocalAsset(p), "mem-00.bin", cache, size)
        full = LocalAsset(p)
        # unaligned, multi-page read
        off, n = BIN_HDR + 12345, 3 * PAGE + 777
        got = a.read(off, n)
        assert got == full.read(off, n)
        st = cache.stats()
        assert st["files"] > 0 and st["bytes"] <= st["cap"]
        # remove the source: the cached pages must still satisfy a repeated read
        os.rename(p, p + ".gone")
        assert a.read(off, n) == got
        # tiny/header reads bypass and hit the source (now missing -> error expected)
        try:
            a.read(0, 8)
            assert False
        except OSError:
            pass
        print(f"  read-through ok: {st['files']} page files, {st['bytes']/2**20:.1f} MiB cached, survives source loss")
    finally:
        shutil.rmtree(root)


def test_lru_trim():
    root = tempfile.mkdtemp()
    try:
        p = os.path.join(root, "mem-00.bin")
        size = make_bin(p, pages=6)
        cache = DiskCache(os.path.join(root, "cache"), cap_bytes=2 * PAGE)
        a = CachedAsset(LocalAsset(p), "x", cache, size)
        full = LocalAsset(p)
        for k in range(6):
            off = BIN_HDR + k * PAGE
            assert a.read(off, PAGE) == full.read(off, PAGE)
        st = cache.stats()
        assert st["bytes"] <= 2 * PAGE, st
        assert st["files"] <= 2
        print(f"  LRU trim ok: {st['files']} files / {st['bytes']/2**20:.1f} MiB <= cap")
    finally:
        shutil.rmtree(root)


def test_sparse_padding_still_works():
    root = tempfile.mkdtemp()
    try:
        p = os.path.join(root, "mem-00.bin")
        size = make_bin(p, pages=2)
        cache = DiskCache(os.path.join(root, "cache"), cap_bytes=64 << 20)
        h = forge_header(0)
        sa = SparseAsset(0, h, real=CachedAsset(LocalAsset(p), "m", cache, size), real_size=size)
        # read straddling the materialized extent -> zeros past it
        off = size - 100
        got = sa.read(off, 400)
        assert got[:100] == LocalAsset(p).read(off, 100)
        assert got[100:] == b"\0" * 300
        # absent bin still zeros
        sz = SparseAsset(3, forge_header(3))
        assert sz.read(BIN_HDR, 128) == b"\0" * 128
        print("  sparse padding + absent-bin zeros preserved under the cache")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_read_through_and_page_split()
    test_lru_trim()
    test_sparse_padding_still_works()
    print("disk cache tests passed")