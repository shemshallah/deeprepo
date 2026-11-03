"""Tier lifecycle: bootstrap registers, stream a GGUF into repovram bins (resumable),
trimmed compaction of reporam, and the hierarchical memory registers.
"""
import hashlib
import json
import os
import struct
import time

import ram16
from . import REPORAM, REPOVRAM, TAG_RAM, TAG_RAM_J, TAG_VRAM, TAG_VRAM_J, BINS, PAGE, PPS, BIN_HDR
from .geom import MAGIC, HF, MACHINE, pack_super, read_super, plan_ram_layout, plan_vram_layout, tier_pages, forge_header


# ---------------- bin-0 forge (bootstrap) ----------------

def forge_bin0(path, tier, layout, meta=None):
    """4 MiB header region + one 4 MiB payload page carrying the DPRSB1 superblock."""
    sb = pack_super(tier, layout, meta)
    assert len(sb) < PAGE - 64
    with open(path, "wb") as f:
        f.truncate(BIN_HDR + PAGE)
        f.seek(0)
        body = struct.pack(HF, MAGIC, 0, BINS, PAGE, PPS, BIN_HDR, MACHINE, 0)
        f.write(body + struct.pack("<I", __import__("zlib").crc32(body)))
        f.seek(BIN_HDR)
        f.write(sb)
    return os.path.getsize(path)


def ensure_release(store, tag):
    try:
        return store._rel()
    except Exception as e:
        if "404" in repr(e) or getattr(e, "code", None) == 404:
            r = store._req(f"https://api.github.com/repos/{store.repo}/releases",
                           json.dumps({"tag_name": tag, "name": tag, "body": "deeprepo sparse tier",
                                       "draft": False, "prerelease": True}).encode(),
                           "application/json", "POST")
            store._rid = r["id"]
            return r
        raise


def _has_blob(store, name):
    if hasattr(store, "_assets"):
        store._assets()
        return name in store._ix
    if hasattr(store, "size"):
        return store.size(name) is not None
    return False


def ensure_journals(vram_repo, ram_repo, backend, token, middle_url="", middle_token=""):
    """Create the journal releases (reporam-j / repovram-j) for GitHub-backed tiers."""
    if backend == "middle":
        return
    from .store import Journal
    for repo, tag in ((ram_repo, "reporam-j"), (vram_repo, "repovram-j")):
        j = Journal(repo, token, tag)
        try:
            j._rel()
        except Exception:
            j._create()
            log_ = print
            log_(f"journal release created: {repo}:{tag}")


def bootstrap(vram_store, ram_store, kv_pages=1024, reg_pages=256, force=False, log=print):
    """Create releases + bin-0 superblocks on both tier repos. Idempotent."""
    out = {}
    for store, tier, layout in (
        (vram_store, "vram", plan_vram_layout(tier_pages())),
        (ram_store, "ram", plan_ram_layout(tier_pages(), kv_pages, reg_pages)),
    ):
        ensure_release(store, getattr(store, "tag", tier))
        if _has_blob(store, "mem-00.bin") and not force:
            log(f"{getattr(store, 'repo', store)}:{getattr(store, 'tag', '')} already bootstrapped — skipping")
            out[tier] = "existing"
            continue
        p = os.path.join("/tmp", f"dpr-{tier}-bin0.bin")
        n = forge_bin0(p, tier, layout)
        _remove_blob(store, "mem-00.bin")
        store.put_file("mem-00.bin", p)
        os.remove(p)
        log(f"bootstrapped {store.repo}:{store.tag} (bin0 {n} B, layout {layout})")
        out[tier] = "created"
    return out


# ---------------- registers (hierarchical memory) ----------------

class Registers:
    """JSON records over a RegionLog region: ring state, jobs, runs, budgets, facts, stats."""

    def __init__(s, ram, region):
        s.r = ram16.RegionLog(ram, region)

    def put(s, key, obj, commit=False):
        s.r.put(key, json.dumps(obj, separators=(",", ":")).encode())
        if commit:
            s.commit()

    def get(s, key, default=None):
        try:
            return json.loads(s.r.get(key))
        except KeyError:
            return default

    def keys(s, prefix=""):
        return sorted(k for k in s.r.keys() if k.startswith(prefix))

    def latest(s, prefix, key_parse=lambda k: k):
        ks = s.keys(prefix)
        return (key_parse(ks[-1]), s.get(ks[-1])) if ks else (None, None)

    def delete(s, key):
        s.r.delete(key)

    def commit(s):
        s.r.commit()

    def refresh(s):
        return s.r.refresh()


# ---------------- model import (workflows only; phone never holds model bytes) ----------------

def model_geometry(gguf_size, page=PAGE):
    """GGUF source bytes -> vram pages (weights start at vram page 1)."""
    return -(-gguf_size // page)


def _bin_used_extent(b, weight_pages, pps=PPS):
    """Pages of bin b that carry model bytes: [max(1, b*pps), min(weight_pages, (b+1)*pps))"""
    lo, hi = max(1, b * pps), min(weight_pages, (b + 1) * pps)
    return lo, hi


def _remove_blob(store, name):
    try:
        if hasattr(store, "_assets"):
            store._assets()
            if name in store._ix:
                store.delete(name)
                return True
        elif hasattr(store, "size"):
            if store.size(name) is not None:
                store.delete(name)
                return True
    except Exception:
        pass
    return False


def import_model(vram_store, jstore, hf, budget=None, resume=True, work="/tmp", log=print):
    """Stream the GGUF from HF and tile it into repovram bins, one bin on disk at a time.
    Bin b holds vram pages [max(1,b*PPS), min(wp+1,(b+1)*PPS)); source bytes for page p
    live at (p-1)*PAGE. Resumable: bins already uploaded (imp records) are skipped."""
    if hasattr(vram_store, "_req"):
        ensure_release(vram_store, vram_store.tag)
    total = hf.size
    wp = model_geometry(total)
    log(f"import: {total/2**30:.2f} GiB -> {wp} vram pages")
    if wp + 1 > tier_pages():
        raise IOError(f"model needs {wp+1} pages, tier holds {tier_pages()} - pick a smaller quant")
    seq, bins_done, etag0 = 0, set(), hf.etag
    for name, _asset in jstore.list(r"imp-\d+\.json"):
        r = json.loads(jstore.get(name))
        seq = max(seq, r["seq"])
        if r.get("bin", -1) >= 0:
            bins_done.add(r["bin"])
        if resume and r.get("etag") and etag0 and r["etag"] != etag0:
            raise IOError("source changed since last import (ETag mismatch); re-run with resume=False")
    if not resume:
        bins_done, seq = set(), 0
    sha = hashlib.sha256()  # over all source bytes streamed in THIS run
    for b in range(BINS):
        lo, hi = _bin_used_extent(b, wp + 1)
        if lo >= hi:
            break
        if b in bins_done:
            log(f"import: bin {b:02d} already present - skipping")
            continue
        bin_src = (lo - 1) * PAGE
        page0_off = BIN_HDR + (lo - b * PPS) * PAGE     # bin 0 keeps payload page 0 = superblock
        size = BIN_HDR + (hi - b * PPS) * PAGE
        name = f"mem-{b:02d}.bin"
        path = os.path.join(work, name)
        exists = os.path.exists(path)
        t0 = time.time()
        sb_page = None
        if b == 0:
            # preserve the bootstrap superblock page; if it is missing/not DPRSB1, regenerate
            # it deterministically (never ship a weights bin without a valid superblock).
            try:
                a = vram_store.asset(name)
                if a.read(BIN_HDR, 6) == b"DPRSB1":
                    sb_page = a.read(BIN_HDR, PAGE)
                    log("import: preserved the vram superblock page")
            except Exception as e:
                log(f"import: warning: could not fetch existing superblock page: {e!r}")
            if not sb_page:
                sb_page = pack_super("vram", plan_vram_layout(tier_pages()))
                log("import: regenerated the vram superblock page")
        with open(path, "r+b" if exists else "wb") as f:
            if not exists:
                f.truncate(size)
            f.seek(0)
            f.write(forge_header(b, page=PAGE, pps=PPS, hdr=BIN_HDR, ns=BINS))
            if sb_page:
                f.seek(BIN_HDR)
                f.write(sb_page)
            f.seek(page0_off)
            pos, to_read = bin_src, (hi - lo) * PAGE
            while to_read > 0:
                k = min(8 << 20, to_read)
                avail = max(0, hf.size - pos)
                if avail == 0:
                    data = b"\0" * k  # final page is not page-aligned: pad zeros
                else:
                    data = hf.read(pos, min(k, avail))
                    if len(data) < k:
                        data = data + b"\0" * (k - len(data))
                f.write(data)
                sha.update(data)
                pos += k
                to_read -= k
            f.truncate(size)
        payload = (hi - lo) * PAGE
        if budget:
            budget.record("down", payload)
            budget.record("up", size)
        log(f"import: {name} {size/2**20:.0f} MiB streamed in {time.time()-t0:.0f}s; uploading")
        _remove_blob(vram_store, name)  # release assets must be unique (bin-0 replaces bootstrap)
        vram_store.put_file(name, path)
        os.remove(path)
        bins_done.add(b)
        seq += 1
        jstore.put(f"imp-{seq:06d}.json", json.dumps({
            "seq": seq, "bin": b, "done_bins": sorted(bins_done), "etag": etag0,
            "t": int(time.time())}).encode())
    jstore.put(f"imp-{seq+1:06d}.json", json.dumps({
        "seq": seq + 1, "bin": -1, "done": True, "size": total, "weight_pages": wp,
        "etag": etag0, "t": int(time.time())}).encode())
    log("import: complete")
    return {"weight_pages": wp, "size": total, "bins": sorted(bins_done), "sha_partial": sha.hexdigest()[:16]}


# ---------------- trimmed compaction (reporam) ----------------

def compact_trimmed(ram, work="/tmp", log=print):
    """Fold journal deltas into fresh bins, uploading only each bin's used extent.
    Only reporam (small used extents). Single-writer discipline required (lease-free)."""
    assert ram.writer and ram.journal, "not a writer"
    by = {}
    for p in ram.jmap:
        by.setdefault(p // ram.pps, []).append(p)
    if not by:
        log("compact: nothing to do")
        return None
    gen, names, old, floor = ram.gen + 1, list(ram.names or ram.tier_names), [], ram.seq
    for sl in sorted(by):
        pages = sorted(by[sl])
        last_in_bin = max(p % ram.pps for p in pages)
        new = f"mem-{sl:02d}.g{gen}.bin"
        size = ram.hdr + (last_in_bin + 1) * ram.page
        path = os.path.join(work, new)
        old_asset = ram.slots[sl]
        with open(path, "wb") as f:
            f.truncate(size)
            f.seek(0)
            f.write(forge_header(sl, page=ram.page, pps=ram.pps, hdr=ram.hdr, ns=(ram.tier_geom or {}).get("ns", BINS)))
            if sl == 0:  # payload page 0 carries the tier superblock; never lose it
                try:
                    f.seek(ram.hdr)
                    f.write(old_asset.read(ram.hdr, ram.page))
                except Exception as e:
                    log(f"compact: warning: could not preserve superblock page: {e!r}")
            for p in pages:
                f.seek(ram.hdr + (p % ram.pps) * ram.page)
                f.write(ram._get(p))
        ram.tier_store.put_file(new, path)
        os.remove(path)
        old.append(names[sl])
        names[sl] = new
    ram.journal.put(f"m-{gen:08d}.json", json.dumps(
        {"gen": gen, "slots": names, "floor": floor, "t": int(time.time())}).encode())
    for n, _ in ram.journal.list():
        if int(n[2:10]) <= floor:
            ram.journal.delete(n)
    for n, _ in ram.journal.list(ram16.MANI)[:-1]:
        ram.journal.delete(n)
    for n in old:
        if n not in names:
            ram.tier_store.delete(n)
    ram.names, ram.tier_names, ram.tier_gen = names, names, gen
    ram.gen, ram.floor = gen, max(ram.floor, floor)
    log(f"compact: gen {gen} published, floor {floor}, bins {sorted(by)}")
    return gen