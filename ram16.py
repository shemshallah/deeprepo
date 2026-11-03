"""RAM16 runner layer: 16 GB byte-addressable memory over 8 GitHub Release bins. stdlib only.

BASE      8 bins (mem-NN.bin, later mem-NN.g<gen>.bin), random access via HTTP Range, 4 MiB pages, LRU cache, coalesced readahead.
JOURNAL   writer appends compressed page-deltas d-<seq>.dlt to a journal release (an asset counts only once state=="uploaded":
          atomic publish). Readers tail it; latest delta wins per page.
COMPACT   writer folds deltas into fresh bins (new generation names), publishes manifest m-<gen>.json {slots, floor}, then
          deletes old bins and deltas <= floor. Readers swap on the manifest; a 404 on a deleted bin triggers refresh+retry.
CLI       python ram16.py status | compact | serve --minutes 335 --handoff owner/repo:ram16-runner.yml
"""
import collections, concurrent.futures as cf, json, os, re, struct, sys, threading, time, urllib.error, urllib.request, zlib

MAGIC, DMAGIC = b"RAM16\0\0\1", b"RAMDLT01"
HF = "<8s5I16sQ"
DELTA, MANI = r"d-\d{8}\.dlt", r"m-\d{8}\.json"


class _NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None


class Asset:
    """One remote file, ranged reads. api=True: private-repo API asset URL -> resolve to short-lived signed URL."""
    def __init__(s, url, token=None, api=False):
        s.url, s.token, s.api, s._sig, s._exp = url, token, api, None, 0

    def _resolve(s):
        rq = urllib.request.Request(s.url, headers={"Authorization": f"Bearer {s.token}",
                                                    "Accept": "application/octet-stream", "User-Agent": "ram16"})
        try:
            urllib.request.build_opener(_NoRedir).open(rq, timeout=30)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307): s._sig, s._exp = e.headers["Location"], time.time() + 240; return
            raise
        raise IOError("asset did not redirect")

    def read(s, off, n, tries=5):
        for k in range(tries):
            try:
                u = s.url
                if s.api and s.token:
                    if not s._sig or time.time() > s._exp: s._resolve()
                    u = s._sig
                rq = urllib.request.Request(u, headers={"Range": f"bytes={off}-{off+n-1}", "User-Agent": "ram16"})
                with urllib.request.urlopen(rq, timeout=60) as r:
                    b = r.read()
                    if r.status == 200: b = b[off:off + n]  # server ignored Range
                if len(b) != n: raise IOError(f"short read {len(b)}/{n}")
                return b
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 410): s._sig = None
                if k == tries - 1 or e.code in (404, 416): raise
            except (IOError, urllib.error.URLError):
                if k == tries - 1: raise
            time.sleep(min(0.25 * 2 ** k, 4))


class DirStore:
    """Local/simulated release: files in a dir served at base_url. Also usable as a journal (list/get/put/delete)."""
    def __init__(s, path, base_url): s.path, s.base = path, base_url; os.makedirs(path, exist_ok=True)
    def list(s, pat=DELTA): return [(n, Asset(f"{s.base}/{n}")) for n in sorted(os.listdir(s.path)) if re.fullmatch(pat, n)]
    def asset(s, n): return Asset(f"{s.base}/{n}")
    def get(s, n): return open(os.path.join(s.path, n), "rb").read()
    def put(s, n, data): t = os.path.join(s.path, n + ".part"); open(t, "wb").write(data); os.replace(t, os.path.join(s.path, n))
    def put_file(s, n, path):
        import shutil; t = os.path.join(s.path, n + ".part"); shutil.copyfile(path, t); os.replace(t, os.path.join(s.path, n))
    def delete(s, n):
        p = os.path.join(s.path, n)
        if os.path.exists(p): os.remove(p)
DirJournal = DirStore


class GitHubStore:
    """One GitHub release used as an asset store (bins, deltas, manifests). Works for public and private repos."""
    def __init__(s, repo, token=None, tag="ram16"): s.repo, s.token, s.tag, s._rid, s._ix = repo, token, tag, None, {}

    def _req(s, url, data=None, ctype=None, method=None, length=None):
        h = {"Accept": "application/vnd.github+json", "User-Agent": "ram16"}
        if s.token: h["Authorization"] = f"Bearer {s.token}"
        if ctype: h["Content-Type"] = ctype
        if length is not None: h["Content-Length"] = str(length)
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h, method=method), timeout=3600) as r:
            b = r.read(); return json.loads(b) if b else None

    def _rel(s):
        r = s._req(f"https://api.github.com/repos/{s.repo}/releases/tags/{s.tag}"); s._rid = r["id"]; return r

    def _assets(s, pat=r".*"):
        rid = s._rel()["id"]; out, pg = [], 1
        while True:
            a = s._req(f"https://api.github.com/repos/{s.repo}/releases/{rid}/assets?per_page=100&page={pg}")
            out += [x for x in a if x["state"] == "uploaded"]
            if len(a) < 100: break
            pg += 1
        s._ix = {x["name"]: x for x in out}
        return sorted((x for x in out if re.fullmatch(pat, x["name"])), key=lambda x: x["name"])

    def _asset(s, x): return Asset(x["url"], s.token, True) if s.token else Asset(x["browser_download_url"])
    def list(s, pat=DELTA): return [(x["name"], s._asset(x)) for x in s._assets(pat)]
    def asset(s, n):
        if n not in s._ix: s._assets()
        return s._asset(s._ix[n])
    def get(s, n):
        s._assets(re.escape(n)); x = s._ix[n]; return s._asset(x).read(0, x["size"]) if x["size"] else b""
    def put(s, n, data):
        if not s._rid: s._rel()
        s._req(f"https://uploads.github.com/repos/{s.repo}/releases/{s._rid}/assets?name={n}", data, "application/octet-stream", "POST")
    def put_file(s, n, path):
        if not s._rid: s._rel()
        with open(path, "rb") as f:
            s._req(f"https://uploads.github.com/repos/{s.repo}/releases/{s._rid}/assets?name={n}", f, "application/octet-stream", "POST", os.path.getsize(path))
    def delete(s, n):
        for x in s._assets(re.escape(n)): s._req(f"https://api.github.com/repos/{s.repo}/releases/assets/{x['id']}", method="DELETE")

class GitHubJournal(GitHubStore):
    def __init__(s, repo, token=None, tag="ram16-journal"): GitHubStore.__init__(s, repo, token, tag)


class Ram16:
    def __init__(s, slot_urls=None, journal=None, token=None, api=False, cache_pages=256, workers=8, readahead=8, maxrun=8,
                 writer=False, store=None, nslots=8):
        s.store, s.gen, s.floor, s.names = store, 0, 0, None
        if store is not None:  # slots come from the newest manifest (or the default names before any compaction)
            m = s._manifest(journal) if journal else None
            s.names = m["slots"] if m else [f"mem-{i:02d}.bin" for i in range(nslots)]
            if m: s.gen, s.floor = m["gen"], m["floor"]
            slot_urls = [store.asset(n) for n in s.names]
        s.slots = [Asset(u, token, api) if isinstance(u, str) else u for u in slot_urls]
        k = struct.calcsize(HF)
        h = s.slots[0].read(0, k + 4)
        mg, sl, ns, s.page, s.pps, s.hdr, mach, _ = struct.unpack(HF, h[:k])
        assert mg == MAGIC and zlib.crc32(h[:k]) == struct.unpack("<I", h[k:])[0], "bad slot-0 header"
        assert ns == len(s.slots), "slot count mismatch"
        s.npages, s.size = ns * s.pps, ns * s.pps * s.page
        s.machine = mach.hex()
        s.cache, s.dirty, s.fly, s.jmap = collections.OrderedDict(), {}, {}, {}
        s.cap, s.ra, s.maxrun, s.writer, s.journal = cache_pages, readahead, maxrun, writer, journal
        s.lk, s.pool, s.last = threading.RLock(), cf.ThreadPoolExecutor(workers), -2
        s.seen, s.seq = set(), s.floor
        s.st = collections.Counter()
        with cf.ThreadPoolExecutor(ns) as ex:  # verify every slot header in parallel (8 round-trips -> 1)
            for i, hh in enumerate(ex.map(lambda a: a.read(0, k + 4), s.slots)):
                m2, sl2, _, pg2, pps2, hd2, mach2, _ = struct.unpack(HF, hh[:k])
                assert (m2, sl2, pg2, pps2, hd2, mach2) == (MAGIC, i, s.page, s.pps, s.hdr, mach), f"slot {i} mismatch"
        if journal: s.refresh()

    # ---- manifest / journal ----
    @staticmethod
    def _manifest(journal):
        ms = journal.list(MANI)
        return json.loads(journal.get(ms[-1][0])) if ms else None

    def _swap(s, m):
        with s.lk:
            s.gen, s.floor, s.names = m["gen"], m["floor"], m["slots"]
            s.slots = [s.store.asset(n) for n in s.names]
            s.cache.clear(); s.jmap = {p: v for p, v in s.jmap.items() if v[3] > s.floor}; s.seq = max(s.seq, s.floor)
        s.st["swaps"] += 1

    def refresh(s):
        """Tail the journal: adopt a newer manifest, then index new deltas (one tiny ranged read of their table each)."""
        if s.store is not None:
            m = s._manifest(s.journal)
            if m and m["gen"] > s.gen: s._swap(m)
        new = [(n, a) for n, a in s.journal.list() if n not in s.seen and int(n[2:10]) > s.floor]
        def idx(na):
            n, a = na; h = a.read(0, 16); assert h[:8] == DMAGIC; c = struct.unpack("<I", h[8:12])[0]
            return n, a, c, a.read(16, 16 * c)
        with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(idx, new))
        with s.lk:
            for n, a, c, t in res:  # ascending name order => later deltas override
                for i in range(c):
                    p, off, cl = struct.unpack_from("<IQI", t, 16 * i)
                    s.jmap[p] = (a, off, cl, int(n[2:10])); s.cache.pop(p, None)
                s.seen.add(n); s.seq = max(s.seq, int(n[2:10]))
        s.st["deltas"] += len(res)
        return len(res)

    def flush(self, max_pages=128):
        assert self.writer and self.journal, "not a writer"
        with self.lk: items = sorted(self.dirty.items())
        for i in range(0, len(items), max_pages):
            ch = items[i:i + max_pages]; blobs = [zlib.compress(b, 1) for _, b in ch]
            off = 16 + 16 * len(ch); tab = b""
            for (p, _), bl in zip(ch, blobs): tab += struct.pack("<IQI", p, off, len(bl)); off += len(bl)
            self.seq += 1
            self.journal.put(f"d-{self.seq:08d}.dlt", DMAGIC + struct.pack("<II", len(ch), 0) + tab + b"".join(blobs))
            with self.lk:
                for p, b in ch:
                    if self.dirty.get(p) is b: del self.dirty[p]; self._put(p, b)
            self.st["flushed_pages"] += len(ch)
        self.refresh()

    # ---- paging ----
    def _put(s, p, b):
        s.cache[p] = b; s.cache.move_to_end(p)
        while len(s.cache) > s.cap: s.cache.popitem(last=False)

    def _fetch(s, pages):
        raw = None
        if any(p not in s.jmap for p in pages):
            sl, o = divmod(pages[0], s.pps)
            raw = s.slots[sl].read(s.hdr + o * s.page, s.page * len(pages)); s.st["range_reqs"] += 1
        out = {}
        for i, p in enumerate(pages):
            j = s.jmap.get(p)
            out[p] = zlib.decompress(j[0].read(j[1], j[2])) if j else raw[i * s.page:(i + 1) * s.page]
        return out

    def _runload(s, pages, futs):
        try:
            for att in (0, 1):
                try: out = s._fetch(pages); break
                except urllib.error.HTTPError as e:  # bin/delta deleted by a compaction: adopt the new manifest and retry once
                    if att or e.code != 404 or not s.journal: raise
                    s.refresh()
            with s.lk:
                for p, b in out.items():
                    if p not in s.dirty: s._put(p, b)
                    s.fly.pop(p, None)
            for p, f in futs.items(): f.set_result(out[p])
        except BaseException as e:
            with s.lk:
                for p in pages: s.fly.pop(p, None)
            for f in futs.values(): f.set_exception(e)

    def _want(s, pages):
        """Schedule loads for missing pages, coalescing contiguous same-slot runs into single Range requests."""
        with s.lk:
            miss = [p for p in pages if p not in s.dirty and p not in s.cache and p not in s.fly]
            runs, cur = [], []
            for p in miss:
                if cur and (p != cur[-1] + 1 or p // s.pps != cur[0] // s.pps or len(cur) >= s.maxrun): runs.append(cur); cur = []
                cur.append(p)
            if cur: runs.append(cur)
            for r in runs:
                futs = {p: cf.Future() for p in r}
                for p, f in futs.items(): s.fly[p] = f
                s.pool.submit(s._runload, r, futs)
            s.st["miss"] += len(miss); s.st["hit"] += len(pages) - len(miss)

    def _get(s, p):
        with s.lk:
            if p in s.dirty: return s.dirty[p]
            if p in s.cache: s.cache.move_to_end(p); return s.cache[p]
            f = s.fly.get(p)
        if f is None: s._want([p]); f = s.fly.get(p)
        if f is None:  # raced to completion
            with s.lk: return s.dirty.get(p) or s.cache[p]
        return f.result()

    def prefetch(s, addr, n): s._want(range(addr // s.page, (addr + n - 1) // s.page + 1))

    def read(s, addr, n):
        assert 0 <= addr and addr + n <= s.size, "out of range"
        p0, p1 = addr // s.page, (addr + n - 1) // s.page
        pages = list(range(p0, p1 + 1))
        if p0 == s.last + 1 or p0 == s.last:  # sequential => stream ahead
            pages += list(range(p1 + 1, min(p1 + 1 + s.ra, s.npages)))
        s._want(pages); s.last = p1
        out = b"".join(s._get(p) for p in range(p0, p1 + 1))
        o = addr - p0 * s.page
        return out[o:o + n]

    def write(s, addr, data):
        assert 0 <= addr and addr + len(data) <= s.size, "out of range"
        a, end = addr, addr + len(data)
        while a < end:
            p = a // s.page; o = a % s.page; k = min(end - a, s.page - o)
            if o == 0 and k == s.page: buf = bytes(data[a - addr:a - addr + k])
            else:
                buf = bytearray(s._get(p)); buf[o:o + k] = data[a - addr:a - addr + k]; buf = bytes(buf)
            with s.lk: s.dirty[p] = buf; s.cache.pop(p, None)
            a += k

    def __getitem__(s, k):
        if isinstance(k, slice): a = k.start or 0; return s.read(a, (k.stop if k.stop is not None else s.size) - a)
        return s.read(k, 1)[0]
    def __setitem__(s, k, v):
        if isinstance(k, slice): s.write(k.start or 0, bytes(v))
        else: s.write(k, bytes([v]))
    def u64(s, a): return struct.unpack("<Q", s.read(a, 8))[0]
    def put_u64(s, a, v): s.write(a, struct.pack("<Q", v))
    def stats(s): return dict(s.st, cached=len(s.cache), dirty=len(s.dirty), inflight=len(s.fly), journal_pages=len(s.jmap), gen=s.gen)


# ======================================================================================
# MEMORY MAP  (page 0 = superblock; weights | kv spill | mem log)   +  GGUF  +  region logs
# ======================================================================================
import hashlib

def plan_layout(total_pages, page, model_size, kv_mb=1024):
    wp = -(-model_size // page); kp = max(1, int(kv_mb * (1 << 20) // page))
    w0, k0 = 1, 1 + wp; h0 = k0 + kp
    assert h0 + 16 <= total_pages, f"model+kv need {h0} pages of {total_pages}"
    return {"page": page, "total_pages": total_pages, "super": [0, 1], "weights": [w0, wp], "kv": [k0, kp], "mem": [h0, total_pages - h0]}

def pack_super(layout, model):
    j = json.dumps({"v": 1, "layout": layout, "model": model}, separators=(",", ":")).encode()
    return b"RAM16MAP" + struct.pack("<I", len(j)) + j

def read_super(ram):
    h = ram.read(0, 12)
    if h[:8] != b"RAM16MAP": raise ValueError("register has no superblock (forge with --gguf)")
    return json.loads(ram.read(12, struct.unpack("<I", h[8:])[0]))

# ggml type -> (block elems, block bytes)
GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110),
        12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 30: (1, 2)}
_SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

class _Cur:
    """Sequential parser over a ranged reader; fetches 1 MiB windows lazily."""
    def __init__(s, rd, limit): s.rd, s.lim, s.buf, s.pos = rd, limit, b"", 0
    def take(s, n):
        if s.pos + n > len(s.buf):
            have = len(s.buf); s.buf += s.rd(have, min(max(n, 1 << 20), s.lim - have)) if have < s.lim else b""
            if s.pos + n > len(s.buf): raise EOFError("GGUF header truncated")
        b = s.buf[s.pos:s.pos + n]; s.pos += n; return b
    def u(s, f): return struct.unpack(f, s.take(struct.calcsize(f)))[0]
    def str(s): return s.take(s.u("<Q")).decode("utf8", "replace")

class T:
    __slots__ = "name dims type off nbytes addr".split()
    def __init__(s, **k):
        for a in s.__slots__: setattr(s, a, k.get(a))
    def __repr__(s): return f"T({s.name} {s.dims} t{s.type} @{s.addr} {s.nbytes}B)"

def _val(c, t, keep=True):
    if t in _SC: return c.u(_SC[t])
    if t == 8: return c.str()
    if t == 9:
        et, n = c.u("<I"), c.u("<Q")
        if et in _SC: c.take(struct.calcsize(_SC[et]) * n); return ("array", et, n)
        for _ in range(n): _val(c, et)
        return ("array", et, n)
    raise ValueError(f"bad gguf value type {t}")

class OutOfCoreGGUF:
    """A GGUF living inside the register at weights_addr+file_offset. Parses only the header; tensors are paged on demand."""
    def __init__(s, ram, sb=None):
        s.ram, s.sb = ram, sb or read_super(ram)
        s.wbase = s.sb["layout"]["weights"][0] * ram.page; s.size = s.sb["model"]["size"]
        c = _Cur(lambda a, n: ram.read(s.wbase + a, n), s.size)
        assert c.take(4) == b"GGUF", "not a GGUF"
        s.version = c.u("<I"); assert s.version >= 2, "GGUF v1 unsupported"
        nt, nk = c.u("<Q"), c.u("<Q"); s.meta = {}
        for _ in range(nk):
            k = c.str(); s.meta[k] = _val(c, c.u("<I"))
        infos = []
        for _ in range(nt):
            nm = c.str(); nd = c.u("<I"); dims = [c.u("<Q") for _ in range(nd)]; ty = c.u("<I"); off = c.u("<Q"); infos.append((nm, dims, ty, off))
        al = int(s.meta.get("general.alignment", 32)); s.data_off = (c.pos + al - 1) // al * al
        byoff = sorted(infos, key=lambda i: i[3]); s.tensors = collections.OrderedDict()
        for j, (nm, dims, ty, off) in enumerate(byoff):
            ne = 1
            for d in dims: ne *= d
            nb = ne // GGML[ty][0] * GGML[ty][1] if ty in GGML else ((byoff[j + 1][3] if j + 1 < len(byoff) else s.size - s.data_off) - off)
            s.tensors[nm] = T(name=nm, dims=dims, type=ty, off=off, nbytes=nb, addr=s.wbase + s.data_off + off)

    def addr(s, name): return s.tensors[name].addr
    def tensor(s, name): t = s.tensors[name]; return s.ram.read(t.addr, t.nbytes)

    def layers(s):
        """{-1: global tensors (embd/output/norm), 0..L-1: blk.N.* tensors}"""
        out = collections.defaultdict(list)
        for t in s.tensors.values():
            m = re.match(r"blk\.(\d+)\.", t.name); out[int(m[1]) if m else -1].append(t)
        return dict(sorted(out.items()))

    def iter_layers(s, window=2):
        """Yield (layer, {name: bytes}) with the next `window` layers prefetching in the background (latency hiding)."""
        L = list(s.layers().items())
        span = max(sum(t.nbytes for t in ts) for _, ts in L) // s.ram.page + 3
        with s.ram.lk: s.ram.cap = max(s.ram.cap, span * (window + 2))  # cache must hold the in-flight window
        def pre(i):
            for t in L[i][1]: s.ram.prefetch(t.addr, t.nbytes)
        for i in range(min(window, len(L))): pre(i)
        for i, (idx, ts) in enumerate(L):
            if i + window < len(L): pre(i + window)
            yield idx, {t.name: s.ram.read(t.addr, t.nbytes) for t in ts}

    def to_file(s, path, chunk_mb=32):
        """Materialize the GGUF to local disk (e.g. for llama.cpp): pipelined ranged reads, sha256 returned."""
        ch = chunk_mb << 20; h = hashlib.sha256(); r = s.ram
        with r.lk: r.cap = max(r.cap, 3 * (ch // r.page) + r.ra + 4)
        with open(path, "wb") as f:
            for o in range(0, s.size, ch):
                if o + ch < s.size: r.prefetch(s.wbase + o + ch, min(ch, s.size - o - ch))
                b = r.read(s.wbase + o, min(ch, s.size - o)); f.write(b); h.update(b)
        return h.hexdigest()


class RegionLog:
    """Append-only key/value log in a register region: KV-cache spill, checkpoints, long-term memory.
    Single writer. commit() is two-phase (data flush, then tail flush); readers verify per-record CRC so a torn publish is invisible."""
    H, M = 64, b"RLOG0001"
    def __init__(s, ram, region):
        s.r, s.base, s.cap = ram, region[0] * ram.page, region[1] * ram.page
        s.idx, s.tail, s.scanned = {}, s.H, s.H
        s.refresh(init=ram.writer)

    def _persisted(s):
        h = s.r.read(s.base, 16); return struct.unpack("<Q", h[8:])[0] if h[:8] == s.M else None

    def refresh(s, init=False):
        if s.r.journal and not s.r.writer: s.r.refresh()
        t = s._persisted()
        if t is None:
            if not init: return 0
            s.r.write(s.base, s.M + struct.pack("<Q", s.H)); s.r.flush(); t = s.H
        pos, n = s.scanned, 0
        while pos + 16 <= t:
            kl, vl, fl, crc = struct.unpack("<IIII", s.r.read(s.base + pos, 16)); sz = kl + vl
            if pos + 16 + sz > t: break
            body = s.r.read(s.base + pos + 16, sz)
            if zlib.crc32(body) != crc: break
            k = body[:kl].decode()
            if fl & 2: s.idx.pop(k, None)
            else: s.idx[k] = (pos + 16 + kl, vl, fl)
            pos += 16 + ((sz + 7) & ~7); n += 1
        s.scanned = pos; s.tail = max(s.tail, pos); return n

    def _append(s, key, val, fl):
        k = key.encode(); body = k + val; n = 16 + ((len(body) + 7) & ~7)
        assert s.tail + n <= s.cap, "region full"
        s.r.write(s.base + s.tail, struct.pack("<IIII", len(k), len(val), fl, zlib.crc32(body)) + body)
        if not fl & 2: s.idx[key] = (s.tail + 16 + len(k), len(val), fl)
        else: s.idx.pop(key, None)
        s.tail += n

    def put(s, key, val, compress=True):
        assert s.r.writer, "read-only"
        if compress and len(val) > 256:
            z = zlib.compress(val, 1)
            if len(z) < len(val): return s._append(key, z, 1)
        s._append(key, val, 0)

    def delete(s, key): assert s.r.writer; s._append(key, b"", 2)
    def get(s, key):
        a, n, fl = s.idx[key]; b = s.r.read(s.base + a, n); return zlib.decompress(b) if fl & 1 else b
    def __contains__(s, k): return k in s.idx
    def keys(s): return list(s.idx)

    def commit(s):
        # one delta is published atomically, so if everything fits in one, a single flush is enough (also saves release assets, cap 1000)
        if len(s.r.dirty) + 1 <= 128: s.r.write(s.base + 8, struct.pack("<Q", s.tail)); s.r.flush()
        else: s.r.flush(); s.r.write(s.base + 8, struct.pack("<Q", s.tail)); s.r.flush()
        s.scanned = s.tail


class Machine:
    """The whole thing: Machine(ram).weights / .kv / .mem"""
    def __init__(s, ram):
        s.ram, s.sb = ram, read_super(ram); L = s.sb["layout"]
        s.weights = OutOfCoreGGUF(ram, s.sb); s.kv = RegionLog(ram, L["kv"]); s.mem = RegionLog(ram, L["mem"])


# ======================================================================================
# COMPACTION  +  RUNNER (serve loop, GitHub-issue chat queue, self hand-off)  +  CLI
# ======================================================================================
def _download(asset, size, path, chunk=64 << 20, workers=4):
    with open(path, "wb") as f: f.truncate(size)
    def one(o):
        b = asset.read(o, min(chunk, size - o))
        with open(path, "r+b") as f: f.seek(o); f.write(b)
    with cf.ThreadPoolExecutor(workers) as ex: list(ex.map(one, range(0, size, chunk)))

def compact(store, journal, work="/tmp", log=print):
    """Fold journal deltas into fresh bins. Only slots with journaled pages are rewritten (one 2 GB bin on disk at a time).
    Order matters: upload new bins -> publish manifest (the atomic switch) -> delete old bins/deltas/manifests."""
    ram = Ram16(None, journal=journal, store=store, writer=True)
    by = collections.defaultdict(list)
    for p in ram.jmap: by[p // ram.pps].append(p)
    if not by: log("compact: nothing to do"); return None
    gen, names, old, floor = ram.gen + 1, list(ram.names), [], ram.seq
    for sl in sorted(by):
        new = f"mem-{sl:02d}.g{gen}.bin"; path = os.path.join(work, new)
        size = ram.hdr + ram.pps * ram.page
        log(f"compact: slot {sl} ({len(by[sl])} dirty pages) -> {new}")
        _download(ram.slots[sl], size, path)
        with open(path, "r+b") as f:
            for p in sorted(by[sl]): f.seek(ram.hdr + (p % ram.pps) * ram.page); f.write(ram._get(p))
        store.put_file(new, path); os.remove(path)
        old.append(names[sl]); names[sl] = new
    journal.put(f"m-{gen:08d}.json", json.dumps({"gen": gen, "slots": names, "floor": floor, "t": int(time.time())}).encode())
    for n, _ in journal.list():
        if int(n[2:10]) <= floor: journal.delete(n)
    for n, _ in journal.list(MANI)[:-1]: journal.delete(n)
    for n in old:
        if n not in names: store.delete(n)
    log(f"compact: gen {gen} published, floor {floor}")
    return gen


class Req:
    def __init__(s, id, messages, handle=None): s.id, s.messages, s.handle = id, messages, handle


class IssueQueue:
    """Chat over GitHub issues: label an issue `ram16-chat`; each issue is a session, each human comment a turn.
    Bot comments start with a hidden marker; an unfinished reply ends with ' ▌' so a crashed runner resumes it."""
    M, CUR = "<!--ram16-->", " ▌"
    def __init__(s, repo, token, label="ram16-chat", system=None):
        s.repo, s.tok, s.label, s.system, s.quiet = repo, token, label, system, {}
    def _api(s, path, data=None, method=None):
        h = {"Accept": "application/vnd.github+json", "User-Agent": "ram16", "Authorization": f"Bearer {s.tok}"}
        body = json.dumps(data).encode() if data is not None else None
        with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{s.repo}{path}", body, h, method), timeout=60) as r:
            b = r.read(); return json.loads(b) if b else None
    def next(s):
        for i in s._api(f"/issues?labels={s.label}&state=open&sort=created&direction=asc&per_page=30"):
            if "pull_request" in i or s.quiet.get(i["number"]) == i["updated_at"]: continue
            cs, pg = [], 1
            while True:
                c = s._api(f"/issues/{i['number']}/comments?per_page=100&page={pg}"); cs += c
                if len(c) < 100: break
                pg += 1
            msgs = [{"role": "user", "content": f"{i['title']}\n\n{i['body'] or ''}".strip()}]; handle = None
            for c in cs:
                b = c["body"] or ""
                if b.startswith(s.M):
                    t = b[len(s.M):]
                    if t.endswith(s.CUR): handle = c["id"]; continue  # unfinished reply: redo it in place
                    msgs.append({"role": "assistant", "content": t.strip()}); handle = None
                else: msgs.append({"role": "user", "content": b.strip()}); handle = None
            if msgs[-1]["role"] == "user": return Req(i["number"], msgs, handle)
            s.quiet[i["number"]] = i["updated_at"]
        return None
    def begin(s, req):
        return req.handle or s._api(f"/issues/{req.id}/comments", {"body": s.M + s.CUR})["id"]
    def update(s, h, text): s._api(f"/issues/comments/{h}", {"body": s.M + text + s.CUR}, "PATCH")
    def finish(s, h, text): s._api(f"/issues/comments/{h}", {"body": s.M + (text or "(empty)")}, "PATCH")


def handoff(spec, token, ref=None):
    """spec = owner/repo:workflow.yml. GITHUB_TOKEN may dispatch workflows in its own repo; other repos need a PAT (actions:write)."""
    repo, _, wf = spec.partition(":")
    ref = ref or os.environ.get("GITHUB_REF_NAME", "main")
    rq = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/dispatches", json.dumps({"ref": ref}).encode(),
                                {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "ram16"}, method="POST")
    urllib.request.urlopen(rq, timeout=30).read()


def llama_factory(model_path, n_ctx=4096):
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=os.cpu_count() or 2, use_mmap=True, verbose=False)


def serve(ram, mach, queue, minutes, llm_factory=llama_factory, journal=None, store=None, handoff_to=None, token=None, idle=20,
          compact_at=300, max_tokens=512, stream_every=4.0, log=print):
    deadline = time.time() + minutes * 60
    # Optional native remote GGUF mount: the FUSE file presents the model as a
    # normal read-only file. llama.cpp/ggml can mmap it; page faults are served
    # from B2 by the native C++ daemon, avoiding a full local GGUF copy.
    model = os.environ.get("RAM16_GGUF_MOUNT_PATH")
    expected_sha = mach.sb.get("model", {}).get("sha256", "")
    if model:
        if not os.path.isfile(model) or os.path.getsize(model) != mach.weights.size:
            raise IOError(f"native GGUF mount is missing or has the wrong size: {model}")
        log(f"using native B2-backed GGUF mmap file {model} ({mach.weights.size/1e6:.0f} MB); page hashes are verified by the native reader")
    else:
        model = "/tmp/ram16-model.gguf"
        t0 = time.time(); sha = mach.weights.to_file(model)
        if expected_sha and sha.lower() != expected_sha.lower():
            try: os.remove(model)
            except OSError: pass
            raise IOError(f"materialized GGUF SHA-256 mismatch: expected {expected_sha}, got {sha}")
        log(f"model materialized {mach.weights.size/1e6:.0f} MB in {time.time()-t0:.0f}s sha {sha[:12]} (verified)")
    llm = llm_factory(model)
    while time.time() < deadline - 30:
        req = queue.next()
        if req is None:
            if journal is not None and len(journal.list()) >= compact_at:  # idle: fold the journal into the bins
                compact(store, journal, log=log); ram.refresh()
            time.sleep(idle); continue
        last = req.messages[-1]["content"]; h = queue.begin(req); t1 = time.time()
        if last.startswith("/remember "):
            mach.mem.put(f"fact/{int(time.time()*1000)}", last[10:].encode()); out = "Stored."
        else:
            facts = [mach.mem.get(k).decode() for k in sorted(mach.mem.keys()) if k.startswith("fact/")][-20:]
            msgs = ([{"role": "system", "content": "Known facts:\n" + "\n".join(facts)}] if facts else []) + req.messages
            out, tick = "", time.time()
            for ch in llm.create_chat_completion(messages=msgs, max_tokens=max_tokens, stream=True):
                d = ch["choices"][0].get("delta", {}).get("content")
                if d:
                    out += d
                    if time.time() - tick >= stream_every: queue.update(h, out); tick = time.time()
        queue.finish(h, out)
        mach.mem.put(f"req/{req.id}/{int(t1)}", json.dumps({"q": last[:2000], "a": out[:4000], "s": round(time.time() - t1, 1)}).encode())
        mach.mem.commit(); log(f"served issue {req.id} in {time.time()-t1:.1f}s")
    mach.kv.commit(); mach.mem.commit()
    if handoff_to: handoff(handoff_to, token); log("handoff dispatched")


def _connect(writer, cache_pages=256):
    repo = os.environ.get("RAM16_REPO") or os.environ["GITHUB_REPOSITORY"]
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    store = GitHubStore(repo, tok, os.environ.get("RAM16_TAG", "ram16")); j = GitHubJournal(repo, tok, os.environ.get("RAM16_JTAG", "ram16-journal"))
    return Ram16(None, journal=j, store=store, writer=writer, cache_pages=cache_pages), j, store, tok, repo


def cli(argv):
    cmd = argv[0] if argv else "status"; arg = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    if cmd == "status":
        ram, j, store, tok, repo = _connect(False); m = Machine(ram)
        print(f"register {ram.size/1e9:.2f} GB gen {ram.gen} journal deltas {len(j.list())} pages-in-journal {len(ram.jmap)}")
        print("model", m.sb["model"], "layout", m.sb["layout"]); print("mem keys", len(m.mem.keys()), "kv keys", len(m.kv.keys()))
    elif cmd == "compact":
        ram, j, store, *_ = _connect(True); compact(store, j, arg("--work", "/tmp"))
    elif cmd == "serve":
        ram, j, store, tok, repo = _connect(True); m = Machine(ram)
        q = IssueQueue(repo, tok, os.environ.get("RAM16_LABEL", "ram16-chat"))
        serve(ram, m, q, float(arg("--minutes", 335)), journal=j, store=store, handoff_to=arg("--handoff"), token=tok)
    else: sys.exit(__doc__)


if __name__ == "__main__": cli(sys.argv[1:])
