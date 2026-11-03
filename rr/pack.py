"""Pack a source (file/stream) into RRBIN bins, stripe across repos, upload via a store, publish manifest. Resumable."""
import argparse, hashlib, os, sys, tempfile
from . import SEG_PAYLOAD, HDR, SEGS_PER_BIN
from .rrbin import pack_segment, parse_header, SPACES
from .manifest import Manifest


def read_full(f, n):
    buf = bytearray()
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c: break
        buf += c
    return bytes(buf)


def pack_stream(f, total, store, repos, rel, space="ram", prefix="rrram", manifest=None, pclass=1, gen=1,
                segs_per_bin=SEGS_PER_BIN, workdir=None, log=print):
    """Stream `total` bytes from f. Bin i goes to repos[i % len(repos)]. Returns manifest."""
    m = manifest or Manifest()
    if space not in m.d["spaces"]: m.add_space(space, total)
    nseg = -(-total // SEG_PAYLOAD); nbins = -(-nseg // segs_per_bin)
    workdir = workdir or tempfile.mkdtemp(prefix="rrpack")
    vseg = 0
    for b in range(nbins):
        name = f"{prefix}-{b}.bin"; repo = repos[b % len(repos)]
        path = os.path.join(workdir, name); off = 0
        with open(path, "wb") as out:
            for i in range(segs_per_bin):
                if vseg >= nseg: break
                chunk = read_full(f, min(SEG_PAYLOAD, total - vseg * SEG_PAYLOAD))
                hdr, stored = pack_segment(chunk, space, b, i, vseg * SEG_PAYLOAD, pclass, gen)
                out.write(hdr); out.write(stored)
                m.set_seg(space, vseg, name, off, len(stored), parse_header(hdr)['codec'], hashlib.sha256(hdr).digest())
                off += HDR + len(stored); vseg += 1
        size = os.path.getsize(path)
        m.add_bin(name, repo, rel, size)
        if store.has(repo, rel, name) and store.size(repo, rel, name) == size: log(f"skip {name} (exists)")
        else: store.put(repo, rel, name, path); log(f"uploaded {name} {size}")
        os.remove(path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--store-dir", required=True)
    ap.add_argument("--repos", default="owner/reporam,owner/repovram"); ap.add_argument("--release", default="rrram-v1")
    a = ap.parse_args()
    from .store import DirStore
    st = DirStore(a.store_dir); total = os.path.getsize(a.src)
    with open(a.src, "rb") as f: m = pack_stream(f, total, st, a.repos.split(","), a.release)
    print(m.digest())

if __name__ == "__main__": main()
