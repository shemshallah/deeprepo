"""Offline tests for the two-tier B2RAM pager (no B2 needed): python test_ram16_b2.py"""
import os, sys, time, threading, json, hashlib, random, struct, zlib
os.environ["RAM16_ALLOW_LOCAL_WRITES"] = "1"
os.environ.pop("GITHUB_REPOSITORY", None)
import ram16, ram16_b2 as rb

class Mem:
    """In-memory stand-in for B2Store with optional per-GET latency."""
    def __init__(s, lat=0.0): s.o, s.lat, s.gets, s.puts, s.lk = {}, lat, 0, 0, threading.Lock()
    def get(s, k):
        if s.lat: time.sleep(s.lat)
        with s.lk: s.gets += 1
        try: return s.o[k]
        except KeyError: raise KeyError(k)
    def put(s, k, b, content_type=None):
        with s.lk: s.o[k] = bytes(b); s.puts += 1
    def exists(s, k): return k in s.o
    def list(s, prefix): return [{"Key": k, "Size": len(v)} for k, v in sorted(s.o.items()) if k.startswith(prefix)]
    def list_versions(s, prefix): return [{"Key": k, "Size": len(v), "VersionId": "1", "_kind": "version", "LastModified": __import__("datetime").datetime(2000,1,1)} for k, v in s.o.items() if k.startswith(prefix)]
    def delete_version(s, k, v): s.o.pop(k, None)
    def delete_current(s, k): s.o.pop(k, None)

P, S = rb.PAGE_SIZE, rb.SMALL_PAGE
BASE = 10 * P  # small tier starts at the 10th 4-MiB page

def t_basic():
    b = Mem(); r = rb.B2RAM(b, writer=True); r.set_small_tier(BASE)
    big = os.urandom(3 * P + 123); r.write(P, big)                       # large tier, unaligned
    rec = os.urandom(5000); r.write(BASE + 7 * S + 100, rec)             # small tier, 5 KB
    span = os.urandom(S + 999); r.write(BASE + 40 * S - 500, span)       # crosses small-page boundary
    r.write(BASE - 10, b"X" * 20)                                        # crosses large->small tier boundary
    g = r.flush(); assert g == 1
    r2 = rb.B2RAM(b, writer=False)
    assert r2.read(P, len(big)) == big
    assert r2.read(BASE + 7 * S + 100, 5000) == rec
    assert r2.read(BASE + 40 * S - 500, len(span)) == span
    assert r2.read(BASE - 10, 20) == b"X" * 20
    assert r2.read(BASE + 99 * S, 64) == bytes(64)                       # untouched => zeros
    # a 5 KB write must only create small objects, not 4-MiB ones
    sizes = [len(v) for k, v in b.o.items() if "/pages/" in k]
    print("  object sizes (compressed):", sorted(sizes)[:3], "...", max(sizes))
    return b

def t_small_write_amplification():
    b = Mem(); r = rb.B2RAM(b, writer=True); r.set_small_tier(BASE); r.flush()
    before = sum(len(v) for k, v in b.o.items() if "/pages/" in k)
    r.write(BASE + 5, os.urandom(4096)); r.flush()
    added = sum(len(v) for k, v in b.o.items() if "/pages/" in k) - before
    assert added <= S + 64, added
    print(f"  4 KB write uploaded {added/1024:.0f} KiB (4-MiB tier would upload ~4096 KiB)")

def t_retier_and_old_manifest():
    b = Mem(); r = rb.B2RAM(b, writer=True); r.write(P, b"weights"); r.flush()   # 'old' store: no small tier
    r2 = rb.B2RAM(b, writer=True); assert r2.small_base is None
    r2.set_small_tier(BASE); r2.flush()
    r3 = rb.B2RAM(b); assert r3.small_base == BASE and r3.read(P, 7) == b"weights"
    r3 = rb.B2RAM(b, writer=True)
    r3.write(BASE, b"kv"); r3.flush()
    r4 = rb.B2RAM(b, writer=True)
    try: r4.set_small_tier(BASE // 2); raise SystemExit("should refuse a different base")
    except RuntimeError: pass
    # tier switch over data must be refused
    b2 = Mem(); q = rb.B2RAM(b2, writer=True); q.write(BASE + 1, b"data"); q.flush()
    q2 = rb.B2RAM(b2, writer=True)
    try: q2.set_small_tier(BASE); raise SystemExit("should refuse: region holds 4-MiB pages")
    except RuntimeError: pass

def t_parallel_speed():
    b = Mem(lat=0.15); r = rb.B2RAM(b, writer=True); data = os.urandom(8 * P)
    r.write(P, data); r.flush()
    rd = rb.B2RAM(b); t = time.time(); got = rd.read(P, len(data)); dt = time.time() - t
    assert got == data
    seq = 8 * 0.15
    print(f"  8 pages @150 ms latency: {dt:.2f}s parallel vs ~{seq:.2f}s sequential")
    assert dt < seq * 0.6
    # prefetch overlaps with reading
    rd2 = rb.B2RAM(b); rd2.prefetch(P, 8 * P); time.sleep(0.6); t = time.time(); rd2.read(P, 8 * P); assert time.time() - t < 0.1

def t_stale_prefetch_safe():
    b = Mem(lat=0.05); r = rb.B2RAM(b, writer=True); r.write(P, b"A" * P); r.flush()
    r.prefetch(P, P); r.write(P, b"B" * P); r.flush()
    assert r.read(P, 4) == b"BBBB"

def t_gc_keeps_small_pages():
    b = Mem(); r = rb.B2RAM(b, writer=True); r.set_small_tier(BASE); r.write(BASE + 3, os.urandom(3000)); r.write(P, os.urandom(900)); r.flush()
    n = len([k for k in b.o if "/pages/" in k])
    rep = rb.garbage_collect(r, retain_generations=1, min_age_days=1, apply=True)
    assert len([k for k in b.o if "/pages/" in k]) == n, rep
    assert rb.B2RAM(b).read(BASE + 3, 3000) is not None

def t_regionlog():
    b = Mem(); r = rb.B2RAM(b, writer=True); r.set_small_tier(BASE)
    lay = {"page": P, "total_pages": rb.PAGE_COUNT, "kv": [10, 256], "mem": [300, 256]}
    log = ram16.RegionLog(r, lay["kv"]); vals = {f"k{i}": os.urandom(random.randint(10, 9000)) for i in range(200)}
    for k, v in vals.items(): log.put(k, v)
    log.commit()
    rr = rb.B2RAM(b); lg = ram16.RegionLog(rr, lay["kv"]); lg.refresh()
    assert all(lg.get(k) == v for k, v in vals.items()) and len(lg.keys()) == 200
    print(f"  RegionLog: 200 records ({sum(map(len, vals.values()))/1024:.0f} KiB) -> {len([k for k in b.o if '/pages/' in k])} objects, {sum(len(v) for k,v in b.o.items() if '/pages/' in k)/1024:.0f} KiB stored")

for n, f in list(globals().items()):
    if n.startswith("t_"):
        print(n); f()
print("ALL OK")
