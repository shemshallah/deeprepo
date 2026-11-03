#!/usr/bin/env python3
"""RAM16 slot forge (Termux).  pkg install python [gh]; termux-setup-storage
  python ramslot.py [--gguf SRC]   menu: 1-8 forge ONE bin, then offer upload + delete (phone never holds >1 bin)
  python ramslot.py 3              forge slot 3 -> mem-02.bin
  python ramslot.py 3 --upload owner/repo[:tag] --rm   forge, upload, delete local copy (needs gh auth login)
  python ramslot.py all --upload owner/repo            slots 1-8 strictly one at a time: forge, upload, delete
  python ramslot.py verify
  --gguf PATH_OR_URL   bake a GGUF into the weights region (URL is streamed by Range: the phone never holds the source)
  --kv-mb N            KV-spill region size (default 1024). Use identical --gguf/--kv-mb for all 8 slots.
Storage budget RAM16_BUDGET_GB (default 3): refuses/offers to delete other bins before forging.

BIN FORMAT (hierarchical + sparse):  slot -> 476 extents (4 MiB) -> 1024 blocks (4 KiB)
  [4 KiB header][top table 476 x 16 B][extents]   extent kinds: ZERO (stored as nothing) | RAW (dense, contiguous: weights)
  | MAPPED (8 KiB leaf table + only the non-zero blocks, zlib where it helps).
  Logical slot = 1,996,488,704 B; physical size = whatever is populated (an empty slot is ~12 KB). Cap < 2 GiB always holds.
"""
import os, sys, struct, json, time, uuid, shutil, subprocess, urllib.request
from ram16 import SlotBuilder, SlotMap, FileAsset, plan_layout, pack_super

BLOCK = int(os.environ.get("RAM16_BLOCK", 4096))
EXT = int(os.environ.get("RAM16_EXT", 1024))      # blocks per extent
EPS = int(os.environ.get("RAM16_EPS", 476))       # extents per slot
N = 8
EXT_BYTES = EXT * BLOCK
OUT = os.environ.get("RAM16_OUT", "/storage/emulated/0/Download")
MAN = "ram16.manifest.json"
BUDGET = int(float(os.environ.get("RAM16_BUDGET_GB", "3")) * 1e9)
KV_MB = float(os.environ.get("RAM16_KV_MB", 1024))


def name(s): return f"mem-{s:02d}.bin"          # s is 0-based
def P(*a): return os.path.join(OUT, *a)


def ident():
    p = P(".ram16.id")
    if os.path.exists(p):
        d = json.load(open(p)); return bytes.fromhex(d["m"]), d["t"]
    m, t = uuid.uuid4().bytes, int(time.time())
    json.dump({"m": m.hex(), "t": t}, open(p, "w")); return m, t


def manifest():
    return json.load(open(P(MAN))) if os.path.exists(P(MAN)) else {}


class Src:
    """GGUF source: local file or http(s) URL (Range, buffered 32 MiB windows, retries)."""
    def __init__(s, spec):
        s.spec, s.url, s.buf, s.bo = spec, spec.startswith("http"), b"", -1
        s.name = spec.rsplit("/", 1)[-1].split("?")[0]
        if s.url:
            r = urllib.request.urlopen(urllib.request.Request(spec, headers={"Range": "bytes=0-0", "User-Agent": "ramslot"}), timeout=60)
            s.size = int(r.headers["Content-Range"].rsplit("/", 1)[1]); r.close()
        else: s.size = os.path.getsize(spec)
    def _raw(s, off, n):
        if not s.url:
            with open(s.spec, "rb") as f: f.seek(off); return f.read(n)
        for k in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(s.spec, headers={"Range": f"bytes={off}-{off+n-1}", "User-Agent": "ramslot"}), timeout=120) as r:
                    b = r.read()
                if len(b) == n: return b
            except Exception:
                if k == 5: raise
            time.sleep(2 ** k)
    def read(s, off, n):
        if not (s.bo <= off and off + n <= s.bo + len(s.buf)):
            s.bo, s.buf = off, s._raw(off, min(max(n, 32 << 20), s.size - off))
        return s.buf[off - s.bo:off - s.bo + n]


def plan(src):
    lay = plan_layout(N * EPS * EXT, BLOCK, src.size, KV_MB, align=EXT)
    sup = pack_super(lay, {"name": src.name, "size": src.size}); assert len(sup) <= EXT_BYTES, "superblock > 1 extent"
    return lay, sup


def make_room(s, need):
    held = [i for i in range(N) if i != s and os.path.exists(P(name(i)))]
    used = sum(os.path.getsize(P(name(i))) for i in held)
    if not held or used + need <= BUDGET: return
    names = ", ".join(name(i) for i in held)
    if sys.stdin.isatty() and input(f"  {names} on disk ({used/1e9:.1f} GB) would exceed the {BUDGET/1e9:.0f} GB budget. Delete them? (upload first!) [y/N] ").lower() == "y":
        for i in held: os.remove(P(name(i)))
    else:
        sys.exit(f"refusing: {names} already on disk; upload/delete first")


def forge(slot, fill="zero", src=None):
    s = slot - 1
    os.makedirs(OUT, exist_ok=True)
    lay, sup = plan(src) if src else (None, None)
    wlo = lay["weights"][0] * BLOCK if src else 0; whi = wlo + (src.size if src else 0)
    need = (min(src.size, EPS * EXT_BYTES) if src else 0) + (EPS * EXT_BYTES if fill == "tag" else 0) + (16 << 20)  # worst-case physical
    make_room(s, need)
    free = shutil.disk_usage(OUT).free
    if free < need + (256 << 20): sys.exit(f"need ~{need>>20} MiB + headroom, only {free>>20} MiB free")
    m, t = ident()
    path, tmp, t0 = P(name(s)), P(name(s)) + ".part", time.time()
    b = SlotBuilder(tmp, s, N, BLOCK, EXT, EPS, m, t)
    for e in range(EPS):
        ge, data = s * EPS + e, None
        a0 = ge * EXT_BYTES
        if fill == "tag":  # first 16 B of every block = b"PG" + global block number: addressing self-test (compresses away)
            data = bytearray(EXT_BYTES)
            for j in range(EXT): data[j * BLOCK:j * BLOCK + 16] = b"PG\0\0\0\0\0\0" + struct.pack("<Q", ge * EXT + j)
        if sup and ge == 0:
            if data is None: data = bytearray(EXT_BYTES)
            data[:len(sup)] = sup
        if src and wlo < a0 + EXT_BYTES and a0 < whi:
            lo, hi = max(a0, wlo), min(a0 + EXT_BYTES, whi)
            if data is None: data = bytearray(EXT_BYTES)
            data[lo - a0:hi - a0] = src.read(lo - wlo, hi - lo)
        b.add_extent(bytes(data) if data is not None else None)
        if e % 8 == 0 or e == EPS - 1:
            print(f"\r slot {slot}/8  extent {e+1}/{EPS}  {time.time()-t0:5.1f}s", end="", flush=True)
    sha, size = b.finish()
    os.replace(tmp, path)
    mf = manifest()
    mf.update(block=BLOCK, ext=EXT, eps=EPS, nslots=N, machine=m.hex(), created=t)
    mf.setdefault("slots", {})[name(s)] = {"bytes": size, "sha256": sha, "fill": fill}
    if src: mf["layout"] = lay; mf["model"] = {"name": src.name, "size": src.size}
    json.dump(mf, open(P(MAN), "w"), indent=1)
    print(f"\n -> {path}  {size:,} B physical (logical {EPS*EXT_BYTES:,})\n    sha256 {sha}")


def verify():
    mf = manifest()
    for s in range(N):
        p = P(name(s))
        if not os.path.exists(p): print(f"  {name(s)}  missing"); continue
        try:
            m = SlotMap(FileAsset(p)); k = [sum(1 for x in m.top if x[0] == q) for q in (0, 1, 2)]
            ok = m.slot == s and (m.bs, m.ext, m.eps) == (BLOCK, EXT, EPS)
            print(f"  {name(s)}  {os.path.getsize(p):>13,} B  header {'ok' if ok else 'BAD'}  extents zero/raw/mapped {k[0]}/{k[1]}/{k[2]}  in-manifest={name(s) in mf.get('slots', {})}")
        except Exception as e:
            print(f"  {name(s)}  BAD ({e})")


def upload(spec, slots, rm=False):
    repo, _, tag = spec.partition(":"); tag = tag or "ram16"
    if subprocess.call(["gh", "release", "view", tag, "-R", repo], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        subprocess.check_call(["gh", "release", "create", tag, "-R", repo, "-t", "RAM16 register", "-n", "memory bins"])
    files = [P(name(s - 1)) for s in slots if os.path.exists(P(name(s - 1)))] + [P(MAN)]
    subprocess.check_call(["gh", "release", "upload", tag, "-R", repo, "--clobber", *files])
    if rm:
        for f in files[:-1]: os.remove(f); print(f"  deleted local {os.path.basename(f)}")


def menu(src=None):
    repo = ""
    while True:
        print("\n RAM16 forge  ->", OUT)
        for s in range(N):
            p = P(name(s)); print(f"  {s+1}) {name(s)}  {'[on phone, %.1f MB]' % (os.path.getsize(p)/1e6) if os.path.exists(p) else '[ ]'}")
        c = input("  1-8 forge one | u upload+delete | v verify | q > ").strip().lower()
        if c == "q": return
        elif c == "v": verify()
        elif c == "u":
            repo = input(f"  owner/repo[:tag] [{repo}] > ").strip() or repo
            sl = [s + 1 for s in range(N) if os.path.exists(P(name(s)))]
            if repo and sl: upload(repo, sl, rm=True)
        elif c.isdigit() and 1 <= int(c) <= N:
            forge(int(c), "zero", src)
            if input("  upload now and delete the local copy? [y/N] ").lower() == "y":
                repo = input(f"  owner/repo[:tag] [{repo}] > ").strip() or repo
                if repo: upload(repo, [int(c)], rm=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    fill = a[a.index("--fill") + 1] if "--fill" in a else "zero"
    up = a[a.index("--upload") + 1] if "--upload" in a else None
    rm = "--rm" in a
    gg = a[a.index("--gguf") + 1] if "--gguf" in a else None
    if "--kv-mb" in a: KV_MB = float(a[a.index("--kv-mb") + 1])
    src = Src(gg) if gg else None
    pos = [x for x in a if not x.startswith("--") and x not in (fill, up, gg, a[a.index("--kv-mb") + 1] if "--kv-mb" in a else None)]
    if not pos: menu(src); sys.exit()
    if pos[0] == "verify": verify(); sys.exit()
    if pos[0] == "all":
        if not up: sys.exit("'all' needs --upload owner/repo (it forges, uploads, deletes one bin at a time)")
        for s in range(1, N + 1): forge(s, fill, src); upload(up, [s], rm=True)
    else:
        forge(int(pos[0]), fill, src)
        if up: upload(up, [int(pos[0])], rm=rm)
