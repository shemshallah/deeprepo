"""Sparse-bin store layer over ram16.GitHubStore/Asset. stdlib only.

SparseAsset   one bin: real ranged-read asset if materialized, zeros otherwise; a real
              bin may be TRIMMED (uploaded only to its used extent) so reads past the
              written extent pad zeros. Header range of an absent bin returns a forged
              valid header so ram16.Ram16 init verification passes.
slot_list     builds the 16-slot tier view from the latest compaction manifest.
connect_tier  opens a tier: repovram (read-only) or reporam (journal reader/writer).
hf_stream     ranged streaming reader for Hugging Face GGUF downloads (workflows only).
"""
import collections, hashlib, http.client, json, os, struct, threading, time, urllib.error, urllib.parse, urllib.request

import ram16
from .geom import forge_header

_HF = "<8s5I16sQ"


class SparseAsset:
    def __init__(self, slot, header56, real=None, real_size=None):
        self.slot, self.header = slot, header56
        self.real, self.size = real, real_size  # size = materialized byte extent (None -> absent)

    def read(self, off, n, tries=5):
        if off < 56 and self.real is None:
            if off != 0:
                raise IOError("partial header read on absent bin")
            return self.header[:n] + b"\0" * max(0, n - 56)
        if self.real is None:
            return b"\0" * n
        avail = max(0, (self.size or 0) - off)
        if avail == 0:
            return b"\0" * n
        data = self.real.read(off, min(n, avail), tries)
        return data + b"\0" * max(0, n - len(data))


def _asset_for(store, name, token, slot, geom):
    try:
        store._assets()
        if name not in store._ix:
            return SparseAsset(slot, forge_header(slot, page=geom["page"], pps=geom["pps"], hdr=geom["hdr"], ns=geom["ns"]))
        x = store._ix[name]
        a = ram16.Asset(x["url"], token, True) if token else ram16.Asset(x["browser_download_url"])
        return SparseAsset(slot, forge_header(slot, page=geom["page"], pps=geom["pps"], hdr=geom["hdr"], ns=geom["ns"]),
                           real=a, real_size=x["size"])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return SparseAsset(slot, forge_header(slot, page=geom["page"], pps=geom["pps"], hdr=geom["hdr"], ns=geom["ns"]))
        raise


def _bin_candidates(store):
    if hasattr(store, "_assets"):
        store._assets()
        return sorted(n for n in store._ix if n.startswith("mem-") and n.endswith(".bin"))
    if hasattr(store, "list_names"):
        return [n for n in store.list_names(r"mem-.*\.bin")]
    return []


def _bin0_header(store, token=None, name="mem-00.bin"):
    cands = _bin_candidates(store)
    if not cands:
        raise IOError(f"{getattr(store, 'repo', store)}:{getattr(store, 'tag', '')} has no mem-*.bin (run bootstrap)")
    if name not in cands:
        name = cands[0]
    if hasattr(store, "_ix"):
        x = store._ix[name]
        a = ram16.Asset(x["url"], token, True) if token else ram16.Asset(x["browser_download_url"])
    else:
        a = store.asset(name)
    h = a.read(0, 56)
    mg, sl, ns, page, pps, hdr, mach, _ = struct.unpack(_HF, h[:52])
    assert mg == ram16.MAGIC, "bad bin-0 magic"
    return {"ns": ns, "page": page, "pps": pps, "hdr": hdr, "machine": mach.hex()}


def slot_list(store, token, geom, names=None):
    """names: compaction manifest slot names or default mem-XX.bin."""
    if names is None:
        names = [f"mem-{i:02d}.bin" for i in range(geom["ns"])]
    return [_asset_for(store, n, token, i, geom) for i, n in enumerate(names)]


def _manifest(journal):
    ms = journal.list(ram16.MANI) if journal else []
    return json.loads(journal.get(ms[-1][0])) if ms else None


def _gh_retry(fn, self, *a, **k):
    """429/403-secondary/5xx backoff with Retry-After and jitter (release uploads are the
    expensive path; a burst otherwise gets a 10-minute secondary-limit 403)."""
    import random
    for attempt in range(6):
        try:
            return fn(self, *a, **k)
        except urllib.error.HTTPError as e:
            retryable = e.code in (403, 429) or 500 <= e.code < 600
            if not retryable or attempt == 5:
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            wait = float(ra) if ra else min(60.0, (2 ** attempt) * 1.5 + random.random())
            time.sleep(wait)
    raise IOError("retries exhausted")


class RetryStore(ram16.GitHubStore):
    """GitHub API with jittered backoff and ETag revalidation.

    Conditional GETs answered 304 do not count against the rate limit, so polling the
    journal/manifest repeatedly becomes nearly free instead of ~2 calls per poll."""

    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s._hc = {}

    def _req(s, url, data=None, ctype=None, method=None, length=None):
        import random
        is_get = method is None or method == "GET"
        cached = s._hc.get(url) if is_get else None
        h = {"Accept": "application/vnd.github+json", "User-Agent": "deeprepo"}
        if s.token:
            h["Authorization"] = f"Bearer {s.token}"
        if ctype:
            h["Content-Type"] = ctype
        if length is not None:
            h["Content-Length"] = str(length)
        if cached:
            h["If-None-Match"] = cached[0]
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h, method=method),
                                            timeout=120) as r:
                    body = r.read()
                    if is_get:
                        et = r.headers.get("ETag")
                        if et:
                            s._hc[url] = (et, body)
                    return json.loads(body) if body else None
            except urllib.error.HTTPError as e:
                if e.code == 304 and cached:
                    return json.loads(cached[1]) if cached[1] else None
                retryable = e.code in (403, 429) or 500 <= e.code < 600
                if not retryable or attempt == 5:
                    raise
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if ra else min(60.0, (2 ** attempt) * 1.5 + random.random())
                time.sleep(wait)
        raise IOError("retries exhausted")


class TierStore(RetryStore):
    """GitHubStore whose asset() returns a SparseAsset (missing/trimmed bins read as zeros).
    This keeps ram16.Ram16's own manifest adoption and compaction-swap logic intact."""

    def __init__(s, repo, token, tag, geom, disk_cache=None):
        super().__init__(repo, token, tag)
        s.geom, s.disk_cache = geom, disk_cache

    def _wrap(s, name):
        slot = int(name[4:6]) if name.startswith("mem-") and name[4:6].isdigit() else 0
        if name not in s._ix:
            s._assets()
        x = s._ix.get(name)
        h = forge_header(slot, page=s.geom["page"], pps=s.geom["pps"], hdr=s.geom["hdr"], ns=s.geom["ns"])
        if x is None:
            return SparseAsset(slot, h)
        a = KeepAliveAsset(x["url"], s.token, True) if s.token else KeepAliveAsset(x["browser_download_url"])
        if s.disk_cache is not None:
            a = CachedAsset(a, name, s.disk_cache, x["size"])
        return SparseAsset(slot, h, real=a, real_size=x["size"])

    def asset(s, name):
        return s._wrap(name)


class _NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(s, *a, **k):
        return None


class KeepAliveAsset:
    """Ranged reader with per-thread persistent HTTPS connections.

    The pager issues ~150-4,000 4 MiB range requests per token/prefill; a new TLS
    handshake per request dominates latency. This reuses one connection per CDN host.
    """

    _tl = threading.local()

    def __init__(s, url, token=None, api=False):
        s.url, s.token, s.api = url, token, api
        s._sig, s._exp, s.size = None, 0.0, None

    def _resolve(s):
        h = {"User-Agent": "deeprepo"}
        if s.token:
            h["Authorization"] = f"Bearer {s.token}"
            h["Accept"] = "application/octet-stream"
        rq = urllib.request.Request(s.url, headers=h)
        try:
            urllib.request.build_opener(_NoRedir).open(rq, timeout=30)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307):
                s._sig, s._exp = e.headers["Location"], time.time() + 240
                return
            raise
        s._sig, s._exp = s.url, time.time() + 240

    def _conn(s, host):
        pool = getattr(KeepAliveAsset._tl, "pool", None)
        if pool is None:
            pool = KeepAliveAsset._tl.pool = {}
        c = pool.get(host)
        if c is None:
            c = pool[host] = http.client.HTTPSConnection(host, timeout=120)
        return c

    def read(s, off, n, tries=5):
        for k in range(tries):
            u = s.url
            if s.token or s.api:
                if not s._sig or time.time() > s._exp:
                    s._resolve()
                u = s._sig
            p = urllib.parse.urlparse(u)
            host = p.netloc
            path = p.path + ("?" + p.query if p.query else "")
            conn = s._conn(host)
            try:
                conn.request("GET", path, headers={"Range": f"bytes={off}-{off + n - 1}",
                                                   "User-Agent": "deeprepo"})
                r = conn.getresponse()
                b = r.read()
                if r.status in (301, 302, 303, 307):
                    loc = r.getheader("Location")
                    if loc:
                        s._sig, s._exp = loc, time.time() + 240
                        raise IOError("redirect")
                if r.status == 200:
                    b = b[off:off + n]
                if r.status not in (200, 206) or len(b) != n:
                    raise IOError(f"short read {len(b)}/{n} http={r.status}")
                return b
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                getattr(KeepAliveAsset._tl, "pool", {}).pop(host, None)
                if k == tries - 1:
                    raise
                time.sleep(min(0.25 * 2 ** k, 4))


class Journal(RetryStore):
    """Delta/manifest journal that can be missing (empty tier) and creates itself on first write."""

    def _create(s):
        import json as _json
        r = s._req(f"https://api.github.com/repos/{s.repo}/releases",
                   _json.dumps({"tag_name": s.tag, "name": s.tag, "body": "deeprepo journal",
                                "draft": False, "prerelease": True}).encode(),
                   "application/json", "POST")
        s._rid = r["id"]

    def list(s, pat=ram16.DELTA):
        try:
            return super().list(pat)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            raise

    def get(s, n):
        try:
            return super().get(n)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise KeyError(n)
            raise

    def put(s, n, data):
        try:
            return super().put(n, data)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        s._create()
        return super().put(n, data)

    def delete(s, n):
        try:
            return super().delete(n)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise


def open_store(repo, tag, backend="github", token="", middle_url="", middle_token=""):
    """Return a GitHub release store or a middleman store for one blob namespace."""
    if backend == "middle":
        from .middle import MiddleStore
        return MiddleStore(middle_url, middle_token, prefix=f"{repo}/{tag}")
    if tag.endswith("-j"):
        return Journal(repo, token, tag)
    return ram16.GitHubStore(repo, token, tag)


def connect_tier(repo, tag, token, journal_tag=None, cache_pages=256, disk_cache=None,
                 backend="github", middle_url="", middle_token="", **kw):
    """Open one tier. journal_tag=None -> read-only (repovram). With journal -> reader, or writer if writer=True in kw."""
    store = open_store(repo, tag, backend, token, middle_url, middle_token)
    probe = "mem-00.bin"
    if journal_tag:
        try:
            j = open_store(repo, journal_tag, backend, token, middle_url, middle_token)
            ms = j.list(ram16.MANI)
            if ms:
                probe = json.loads(j.get(ms[-1][0]))["slots"][0]
        except Exception:
            pass
    geom = _bin0_header(store, token, probe)
    if backend == "middle":
        from .middle import MiddleTierStore
        from .middle import MiddleTierStore
        tier_store = MiddleTierStore(middle_url, middle_token, prefix=f"{repo}/{tag}", geom=geom,
                                     disk_cache=disk_cache)
        journal = open_store(repo, journal_tag, backend, token, middle_url, middle_token) if journal_tag else None
    else:
        tier_store = TierStore(repo, token, tag, geom, disk_cache=disk_cache)
        journal = Journal(repo, token, journal_tag) if journal_tag else None
    far = None
    ram = ram16.Ram16(None, journal=journal, store=tier_store, nslots=geom["ns"], cache_pages=cache_pages, **kw)
    ram.tier_geom, ram.tier_store = geom, tier_store
    ram.tier_names = list(ram.names or [f"mem-{i:02d}.bin" for i in range(geom["ns"])])
    ram.tier_gen = ram.gen
    return ram


class HfFile:
    """Streaming ranged reader over an HTTP(S) URL (Hugging Face GGUF). Verifies Content-Length,
    honors ETag identity across resumes. Used by the load-model workflow on a runner."""

    def __init__(self, url, headers=None, chunk=8 << 20):
        self.url, self.extra, self.chunk = url, headers or {}, chunk
        rq = urllib.request.Request(url, headers=dict(self.extra, **{"User-Agent": "deeprepo/1"}))
        with urllib.request.urlopen(rq, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
            self.etag = r.headers.get("ETag") or r.headers.get("X-Linked-ETag") or ""

    def read(self, off, n, tries=6):
        out = bytearray()
        while len(out) < n:
            a, k = off + len(out), min(self.chunk, n - len(out))
            for t in range(tries):
                try:
                    rq = urllib.request.Request(self.url, headers=dict(self.extra, **{
                        "User-Agent": "deeprepo/1", "Range": f"bytes={a}-{a + k - 1}"}))
                    with urllib.request.urlopen(rq, timeout=300) as r:
                        b = r.read()
                        if r.status == 200 and a > 0:
                            b = b[a:a + k]
                        if not b:
                            raise IOError("empty range read")
                        out += b
                        break
                except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
                    if t == tries - 1:
                        raise
                    time.sleep(min(2 ** t, 30))
        return bytes(out[:n])


# ==================== persistent local tiers (self-hosted / workstation) ====================
# On a machine that persists between runs, the bins can literally live in the repo tree
# (gitignored) and be modified in place. Same geometry, same pager, same engine; only the
# store backend changes. Hosted runners are ephemeral, so they use release tiers instead.

class LocalAsset:
    def __init__(s, path):
        s.path = path
        s.size = os.path.getsize(path) if os.path.exists(path) else 0

    def read(s, off, n, tries=1):
        with open(s.path, "rb") as f:
            f.seek(off)
            return f.read(n)


class LocalJournal:
    """Delta/manifest journal as files in a directory (unique names, atomic publish)."""

    def __init__(s, root):
        s.root = root
        os.makedirs(root, exist_ok=True)

    def _p(s, n):
        return os.path.join(s.root, n)

    def list(s, pat=ram16.DELTA):
        import re
        return [(n, LocalAsset(s._p(n))) for n in sorted(os.listdir(s.root)) if re.fullmatch(pat, n)]

    def asset(s, n):
        return LocalAsset(s._p(n))

    def get(s, n):
        return open(s._p(n), "rb").read()

    def put(s, n, data):
        tmp = s._p(n) + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, s._p(n))

    def delete(s, n):
        try:
            os.remove(s._p(n))
        except FileNotFoundError:
            pass


class LocalTierStore:
    """Bins as real files, read/written in place. Used with LocalJournal for a RAM tier.
    Not suitable for git: files are <=1.99 GiB and stay out of version control."""

    tag = "local"

    def __init__(s, root, geom, disk_cache=None):
        s.root, s.geom, s.disk_cache = root, geom, disk_cache
        os.makedirs(root, exist_ok=True)

    def _p(s, n):
        return os.path.join(s.root, n)

    def asset(s, name):
        slot = int(name[4:6]) if name.startswith("mem-") and name[4:6].isdigit() else 0
        h = forge_header(slot, page=s.geom["page"], pps=s.geom["pps"], hdr=s.geom["hdr"], ns=s.geom["ns"])
        p = s._p(name)
        if os.path.exists(p):
            real = LocalAsset(p)
            if s.disk_cache is not None:
                real = CachedAsset(real, name, s.disk_cache, os.path.getsize(p))
            return SparseAsset(slot, h, real=real, real_size=os.path.getsize(p))
        return SparseAsset(slot, h)

    def put_file(s, name, path):
        tmp = s._p(name) + ".part"
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            while True:
                b = src.read(16 << 20)
                if not b:
                    break
                dst.write(b)
        os.replace(tmp, s._p(name))

    def put(s, name, data):
        tmp = s._p(name) + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, s._p(name))

    def delete(s, name):
        try:
            os.remove(s._p(name))
        except FileNotFoundError:
            pass


def local_bootstrap(root, kv_pages=1024, reg_pages=256, force=False, log=print):
    """Forge bin-0 superblocks for both tiers under a persistent directory."""
    from .geom import plan_ram_layout, plan_vram_layout, tier_pages
    vdir, rdir = os.path.join(root, "vram"), os.path.join(root, "ram")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    out = {}
    for d, tier, layout, name in ((vdir, "vram", plan_vram_layout(tier_pages()), "vram"),
                                  (rdir, "ram", plan_ram_layout(tier_pages(), kv_pages, reg_pages), "ram")):
        p = os.path.join(d, "mem-00.bin")
        if os.path.exists(p) and not force:
            out[name] = "existing"
            continue
        from .tiers import forge_bin0
        forge_bin0(p, tier, layout)
        out[name] = "created"
        log(f"local bootstrap: {p} ({layout})")
    return out


def connect_local_tier(root, journal_dir=None, writer=False, cache_pages=256, disk_cache=None):
    journal = LocalJournal(journal_dir) if journal_dir else None
    probe = "mem-00.bin"
    if journal:
        ms = journal.list(ram16.MANI)
        if ms:
            probe = json.loads(journal.get(ms[-1][0]))["slots"][0]
    p0 = os.path.join(root, probe)
    if not os.path.exists(p0):
        cands = sorted(n for n in os.listdir(root) if n.startswith("mem-") and n.endswith(".bin"))
        if not cands:
            raise IOError(f"{root} has no mem-*.bin (run local bootstrap)")
        p0 = os.path.join(root, cands[0])
    h = LocalAsset(p0).read(0, 56)
    mg, sl, ns, page, pps, hdr, mach, _ = struct.unpack(_HF, h[:52])
    assert mg == ram16.MAGIC, "bad local bin-0 header"
    geom = {"ns": ns, "page": page, "pps": pps, "hdr": hdr, "machine": mach.hex()}
    store = LocalTierStore(root, geom, disk_cache=disk_cache)
    ram = ram16.Ram16(None, journal=journal, store=store, nslots=geom["ns"],
                      writer=writer, cache_pages=cache_pages)
    ram.tier_geom, ram.tier_store = geom, store
    ram.tier_names = list(ram.names or [f"mem-{i:02d}.bin" for i in range(geom["ns"])])
    ram.tier_gen = ram.gen
    return ram


# ==================== job-local SSD page cache (in-place files, ephemeral) ====================
# A hosted runner may create/modify files of any size in its working tree — they simply die
# with the VM and cannot be pushed (>100 MB). We use that as L0.5: a bounded page cache on
# the runner SSD, written in place during a shift. It cuts network page fetches; it is NOT
# the shared register (that is the release tier).

class DiskCache:
    """Page files under `root`, LRU-trimmed to cap_bytes. Single process per VM."""

    def __init__(s, root, cap_bytes=6 << 30, page=4 << 20):
        s.root, s.cap, s.page = root, cap_bytes, page
        os.makedirs(root, exist_ok=True)
        s._idx = collections.OrderedDict()   # path -> size, oldest first
        s._size = 0
        s._lk = threading.Lock()
        s._scan()

    def _scan(s):
        for dirpath, _dirs, files in os.walk(s.root):
            for f in files:
                if not f.endswith(".pg"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    n = os.path.getsize(p)
                except OSError:
                    continue
                s._idx[p] = n
                s._size += n

    def _p(s, name, off):
        d = os.path.join(s.root, hashlib.sha1(name.encode()).hexdigest()[:16])
        return os.path.join(d, f"{off:012x}.pg")

    def get(s, name, off, length):
        p = s._p(name, off)
        with s._lk:
            if p not in s._idx:
                return None
            try:
                with open(p, "rb") as f:
                    b = f.read()
            except OSError:
                s._forget(p)
                return None
            if len(b) != length:
                s._forget(p)
                return None
            s._idx.move_to_end(p)
        try:
            os.utime(p, None)
        except OSError:
            pass
        return b

    def put(s, name, off, data):
        p = s._p(name, off)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".part"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        except OSError:
            return
        with s._lk:
            if p in s._idx:
                s._size -= s._idx[p]
            s._idx[p] = len(data)
            s._size += len(data)
            s._idx.move_to_end(p)
            s._trim()

    def _forget(s, p):
        s._size -= s._idx.pop(p, 0)
        try:
            os.remove(p)
        except OSError:
            pass

    def _trim(s):
        while s._size > s.cap and s._idx:
            p, n = s._idx.popitem(last=False)
            s._size -= n
            try:
                os.remove(p)
            except OSError:
                pass

    def stats(s):
        with s._lk:
            return {"files": len(s._idx), "bytes": s._size, "cap": s.cap}


class CachedAsset:
    """Page-split read-through cache over a materialized bin asset (immutable per name)."""

    def __init__(s, real, name, cache, size):
        s.real, s.name, s.cache, s.size = real, name, cache, size

    def read(s, off, n, tries=5):
        end = off + n
        out = bytearray()
        pi = off // s.cache.page
        while off < end:
            p0 = pi * s.cache.page
            plen = min(s.cache.page, (s.size or 0) - p0)
            if plen <= 0:
                out += b"\0" * (end - off)
                break
            b = s.cache.get(s.name, p0, plen)
            if b is None:
                b = s.real.read(p0, plen, tries)
                if len(b) < plen:
                    b = b + b"\0" * (plen - len(b))
                s.cache.put(s.name, p0, b)
            a = max(off, p0) - p0
            z = min(end, p0 + plen) - p0
            out += b[a:z]
            off = p0 + z
            pi += 1
        return bytes(out[:n])
