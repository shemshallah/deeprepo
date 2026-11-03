#!/usr/bin/env python3
"""RAM16 + Backblaze B2: original GGUF/issue-chat/memory runner on sparse 32-GiB paging.

The original ram16.py remains the authoritative model, GGUF, GitHub issue queue,
layout, and RegionLog implementation. This module supplies a B2-backed RAM
object with the same byte-addressed interface expected by Machine/serve.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, threading, time, zlib, urllib.request, urllib.error, collections, shutil, uuid, datetime
import concurrent.futures as cf
from pathlib import Path

import ram16

# Credentials intentionally blank. Prefer GitHub Actions repository secrets.
B2_KEY_ID = ""
B2_APPLICATION_KEY = ""
B2_BUCKET = ""
B2_ENDPOINT = ""

VIRTUAL_SIZE = 32 * 1024**3
PAGE_SIZE = 4 * 1024**2
PAGE_COUNT = VIRTUAL_SIZE // PAGE_SIZE
PREFIX = os.getenv("RAM16_B2_PREFIX", "ram16")
FORMAT = 2
MAX_STORED_BYTES = int(os.getenv("RAM16_B2_MAX_BYTES", "9500000000"))
ZERO_PAGE = bytes(PAGE_SIZE)
# Two page tiers. Addresses below `small_base` (weights) use 4-MiB pages; addresses from `small_base` up
# (the KV-spill and memory regions) use small pages so a few-KB write rewrites ~256 KiB, not 4 MiB.
SMALL_PAGE = int(os.getenv("RAM16_B2_SMALL_KB", "256")) * 1024
WORKERS = int(os.getenv("RAM16_B2_WORKERS", "16"))
MIN_CACHE_BYTES = int(os.getenv("RAM16_B2_MIN_CACHE_MB", "256")) << 20
if SMALL_PAGE < 4096 or PAGE_SIZE % SMALL_PAGE:
    raise SystemExit("RAM16_B2_SMALL_KB must be >= 4 and divide 4096 (e.g. 16, 64, 256, 1024)")


def _config():
    c = {
        "key": os.getenv("B2_KEY_ID") or B2_KEY_ID,
        "secret": os.getenv("B2_APPLICATION_KEY") or B2_APPLICATION_KEY,
        "bucket": os.getenv("B2_BUCKET") or B2_BUCKET,
        "endpoint": os.getenv("B2_ENDPOINT") or B2_ENDPOINT,
    }
    missing = [k for k, v in c.items() if not v]
    if missing:
        raise SystemExit("Missing B2 settings: " + ", ".join(missing) +
                         ". Set B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET, B2_ENDPOINT as environment variables/secrets.")
    return c


class B2Store:
    def __init__(self):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise SystemExit("Install requirements: python -m pip install -r requirements.txt") from e
        c = _config(); self.bucket = c["bucket"]
        self.s3 = boto3.client("s3", endpoint_url=c["endpoint"],
            aws_access_key_id=c["key"], aws_secret_access_key=c["secret"],
            region_name="us-east-1", config=Config(signature_version="s3v4", retries={"max_attempts": 8, "mode": "standard"}))

    def get(self, key):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:
            code = getattr(e, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code == 404 or getattr(e, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise KeyError(key) from e
            raise

    def put(self, key, body, content_type="application/octet-stream"):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

    def exists(self, key):
        try: self.s3.head_object(Bucket=self.bucket, Key=key); return True
        except Exception as e:
            code = getattr(e, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code == 404: return False
            raise

    def list(self, prefix):
        token = None
        while True:
            q = {"Bucket": self.bucket, "Prefix": prefix}
            if token: q["ContinuationToken"] = token
            page = self.s3.list_objects_v2(**q)
            yield from page.get("Contents", [])
            if not page.get("IsTruncated"): break
            token = page["NextContinuationToken"]

    def list_versions(self, prefix):
        key_marker = version_marker = None
        while True:
            q = {"Bucket": self.bucket, "Prefix": prefix}
            if key_marker: q["KeyMarker"] = key_marker
            if version_marker: q["VersionIdMarker"] = version_marker
            page = self.s3.list_object_versions(**q)
            for item in page.get("Versions", []):
                yield {**item, "_kind": "version"}
            for item in page.get("DeleteMarkers", []):
                yield {**item, "_kind": "delete-marker"}
            if not page.get("IsTruncated"): break
            key_marker = page.get("NextKeyMarker")
            version_marker = page.get("NextVersionIdMarker")

    def delete_version(self, key, version_id):
        self.s3.delete_object(Bucket=self.bucket, Key=key, VersionId=version_id)

    def delete_current(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)


class B2RAM:
    """Sparse 32-GiB RAM16-compatible byte store on B2: immutable content-addressed pages in two tiers
    (4-MiB for weights, small pages for the KV/mem regions). Missing pages read as zeros.
    Reads are parallel (thread pool) with real prefetch; flush uploads dirty pages in parallel."""
    def __init__(self, backend=None, writer=False, cache_pages=32):
        self.b2 = backend or B2Store(); self.writer = writer
        self.page, self.size, self.npages = PAGE_SIZE, VIRTUAL_SIZE, PAGE_COUNT
        self.small, self.small_base = SMALL_PAGE, None
        self._check_writer_authority()
        self.cap, self.ra, self.last = max(2, cache_pages), 0, -2
        self.lk = threading.RLock(); self.cache = collections.OrderedDict(); self.cache_bytes = 0
        self.dirty = {}; self.fly = {}; self.epoch = 0
        self.pool = cf.ThreadPoolExecutor(WORKERS)
        self.journal = None; self.jmap = {}; self.gen = 0
        self.page_map = {}; self.spage_map = {}; self.known = set(); self._unpublished = False
        self.st = collections.Counter()
        self.stored_bytes = sum(int(x.get("Size", 0)) for x in self.b2.list(f"{PREFIX}/pages/"))
        self._load()

    def _check_writer_authority(self):
        if not self.writer: return
        current = os.getenv("GITHUB_REPOSITORY", "").strip()
        coordinator = (os.getenv("RAM16_COORDINATOR_REPO") or current).strip()
        if os.getenv("RAM16_ALLOW_LOCAL_WRITES") == "1" and not current:
            return  # explicit escape hatch for controlled local administration/tests
        if not coordinator:
            raise PermissionError("set RAM16_COORDINATOR_REPO to the canonical owner/repo (or run inside GitHub Actions); direct writers are disabled")
        if current.lower() != coordinator.lower():
            raise PermissionError(f"B2 writes are centralized in {coordinator}; dispatch the RAM16 workflow there instead of writing from {current or 'a non-coordinator environment'}")

    # ---- manifest ----
    def _load(self):
        try: ptr_raw = self.b2.get(f"{PREFIX}/CURRENT")
        except KeyError:
            if not self.writer: raise RuntimeError("RAM16 B2 store is not initialized; run `init` or `format --gguf` as writer")
            self._publish(0, {}, {}); return
        ptr = json.loads(ptr_raw); manifest_raw = self.b2.get(ptr["manifest"])
        if hashlib.sha256(manifest_raw).hexdigest() != ptr.get("sha256"):
            raise IOError("B2 CURRENT manifest SHA-256 mismatch")
        m = json.loads(manifest_raw)
        if m.get("format") != FORMAT or m.get("virtual_size") != VIRTUAL_SIZE or m.get("page_size") != PAGE_SIZE:
            raise RuntimeError("Unsupported RAM16-B2 manifest or virtual geometry mismatch")
        if int(m.get("generation", -1)) != int(ptr.get("generation", -2)):
            raise IOError("B2 CURRENT generation does not match its manifest")
        pages = m.get("pages")
        if not isinstance(pages, dict): raise IOError("B2 manifest page map is invalid")
        def ok(d): return isinstance(d, str) and len(d) == 64
        sbase = m.get("small_base"); small = int(m.get("small_page", self.small)); spages = m.get("spages", {}) or {}
        if sbase is not None and (int(sbase) % PAGE_SIZE or not 0 < int(sbase) < VIRTUAL_SIZE):
            raise IOError("B2 manifest small_base is invalid")
        if small < 4096 or PAGE_SIZE % small: raise IOError("B2 manifest small_page is invalid")
        nsmall = 0 if sbase is None else (VIRTUAL_SIZE - int(sbase)) // small
        for k, d in pages.items():
            if not str(k).isdigit() or not 0 <= int(k) < self.npages or not ok(d):
                raise IOError("B2 manifest contains an invalid page reference")
            if sbase is not None and int(k) * PAGE_SIZE >= int(sbase): raise IOError("large page inside the small-page tier")
        for k, d in spages.items():
            if sbase is None or not str(k).isdigit() or not 0 <= int(k) < nsmall or not ok(d):
                raise IOError("B2 manifest contains an invalid small-page reference")
        with self.lk:
            self.gen, self.page_map, self.spage_map = int(m["generation"]), dict(pages), dict(spages)
            self.small, self.small_base = small, (None if sbase is None else int(sbase))
            self.known = set(pages.values()) | set(spages.values())
            self._reset_cache()

    def _publish(self, generation, pages, spages):
        # Optimistic stale-writer detection. GitHub Actions concurrency in the coordinator repository is the
        # serialization authority; this catches stale snapshots but is not a storage-provider atomic CAS.
        if generation > 0:
            try:
                current = json.loads(self.b2.get(f"{PREFIX}/CURRENT"))
                if int(current.get("generation", -1)) != generation - 1:
                    raise RuntimeError("stale B2 writer: CURRENT advanced; refresh and retry")
            except KeyError:
                if generation != 0: raise RuntimeError("B2 CURRENT disappeared during commit")
        m = {"format": FORMAT, "generation": generation, "virtual_size": self.size,
             "page_size": self.page, "page_count": self.npages, "pages": pages,
             "small_base": self.small_base, "small_page": self.small, "spages": spages,
             "created_unix": int(time.time())}
        raw = json.dumps(m, sort_keys=True, separators=(",", ":")).encode()
        key = f"{PREFIX}/manifests/{generation:012d}.json"
        self.b2.put(key, raw, "application/json")
        ptr = json.dumps({"generation": generation, "manifest": key,
                          "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True).encode()
        self.b2.put(f"{PREFIX}/CURRENT", ptr, "application/json")
        with self.lk:
            self.gen, self.page_map, self.spage_map = generation, dict(pages), dict(spages)
            self.known = set(pages.values()) | set(spages.values())

    def set_small_tier(self, base):
        """Switch addresses >= base to small pages. Only legal while that range holds no data (right after format)."""
        if base % self.page or not 0 < base < self.size: raise ValueError("small-tier base must be page aligned")
        if self.small_base == base: return
        if self.small_base is not None: raise RuntimeError(f"small tier already starts at {self.small_base}; re-format to change it")
        with self.lk:
            if any(int(k) * self.page >= base for k in self.page_map) or any(t == 0 and i * self.page >= base for t, i in self.dirty):
                raise RuntimeError("the KV/mem region already holds 4-MiB pages; re-format the store to use small pages")
            self.small_base = base; self._unpublished = True

    # ---- tiers ----
    def _unit(self, addr):
        if self.small_base is not None and addr >= self.small_base:
            i = (addr - self.small_base) // self.small
            return 1, i, self.small_base + i * self.small, self.small
        i = addr // self.page
        return 0, i, i * self.page, self.page

    def _usize(self, t): return self.small if t else self.page
    def _zero(self, t):
        if t == 0: return ZERO_PAGE
        if len(getattr(self, "_szero", b"")) != self.small: self._szero = bytes(self.small)
        return self._szero
    def _map(self, t): return self.spage_map if t else self.page_map

    # ---- cache / paging ----
    def _budget(self): return max(self.cap * self.page, MIN_CACHE_BYTES)
    def _reset_cache(self):
        self.cache.clear(); self.cache_bytes = 0; self.epoch += 1

    def _put(self, k, data):
        old = self.cache.pop(k, None)
        if old is not None: self.cache_bytes -= len(old)
        self.cache[k] = data; self.cache_bytes += len(data)
        while self.cache_bytes > self._budget() and len(self.cache) > 1:
            _, v = self.cache.popitem(last=False); self.cache_bytes -= len(v)

    def _drop(self, k):
        v = self.cache.pop(k, None)
        if v is not None: self.cache_bytes -= len(v)

    def _fetch(self, digest, size):
        data = zlib.decompress(self.b2.get(f"{PREFIX}/pages/{digest}.z"))
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise IOError(f"B2 page {digest[:12]} failed size/hash verification")
        return data

    def _load_unit(self, k, digest, size, epoch):
        try:
            data = self._fetch(digest, size)
            with self.lk:
                if k not in self.dirty and epoch == self.epoch: self._put(k, data)
                self.st["fetched"] += 1
            return data
        finally:
            with self.lk: self.fly.pop((k, epoch), None)

    def _submit(self, k, digest, size):  # lk held
        fk = (k, self.epoch); f = self.fly.get(fk)
        if f is None:
            f = self.pool.submit(self._load_unit, k, digest, size, self.epoch); self.fly[fk] = f
        return f

    def _kick(self, t, i):  # lk held: start a background load if the unit is not already local
        k = (t, i)
        if k in self.dirty or k in self.cache: return
        d = self._map(t).get(str(i))
        if d: self._submit(k, d, self._usize(t))

    def _get_unit(self, t, i):
        k = (t, i)
        with self.lk:
            if k in self.dirty: return self.dirty[k]
            c = self.cache.get(k)
            if c is not None: self.cache.move_to_end(k); self.st["hit"] += 1; return c
            d = self._map(t).get(str(i))
            if d is None: return self._zero(t)
            f = self._submit(k, d, self._usize(t))
        return f.result()

    def _get(self, p):  # one 4-MiB page by index (compat with ram16.py helpers)
        return self.read(p * self.page, self.page)
    _page = _get

    def prefetch(self, addr, n):
        if n <= 0 or addr >= self.size: return
        a, end, cnt = addr, min(addr + n, self.size), 0
        with self.lk:
            while a < end and cnt < 4 * WORKERS:
                t, i, ub, us = self._unit(a); self._kick(t, i); a = ub + us; cnt += 1

    def read(self, addr, n):
        if addr < 0 or n < 0 or addr + n > self.size: raise AssertionError("out of range")
        if n == 0: return b""
        units, a, end = [], addr, addr + n
        while a < end:
            t, i, ub, us = self._unit(a); take = min(end - a, ub + us - a); units.append((t, i, a - ub, take)); a += take
        win = max(2, min(2 * WORKERS, (self._budget() // 2) // self.page))
        out = bytearray()
        for j, (t, i, off, take) in enumerate(units):
            with self.lk:
                for t2, i2, _, _ in units[j:j + win]: self._kick(t2, i2)
            out += self._get_unit(t, i)[off:off + take]
        return bytes(out)

    def write(self, addr, data):
        if not self.writer: raise PermissionError("B2 RAM opened read-only")
        data = memoryview(data); end = addr + len(data)
        if addr < 0 or end > self.size: raise AssertionError("out of range")
        pos = 0
        while addr + pos < end:
            absolute = addr + pos; t, i, ub, us = self._unit(absolute); off = absolute - ub
            take = min(end - absolute, us - off)
            if off == 0 and take == us: page = bytes(data[pos:pos + take])
            else:
                b = bytearray(self._get_unit(t, i)); b[off:off + take] = data[pos:pos + take]; page = bytes(b)
            with self.lk: self.dirty[(t, i)] = page; self._drop((t, i))
            pos += take

    def flush(self):
        if not self.writer: raise PermissionError("B2 RAM opened read-only")
        if not self.dirty and not self._unpublished: return self.gen
        with self.lk: items = sorted(self.dirty.items())
        pages, spages = dict(self.page_map), dict(self.spage_map); todo = {}
        for (t, i), data in items:
            m = spages if t else pages
            if data == self._zero(t): m.pop(str(i), None); continue
            d = hashlib.sha256(data).hexdigest(); m[str(i)] = d
            if d not in self.known and d not in todo: todo[d] = data
        def up(item):
            d, data = item; packed = zlib.compress(data, 1)
            with self.lk:
                if self.stored_bytes + len(packed) > MAX_STORED_BYTES:
                    raise RuntimeError(f"B2 page-object safety cap ({MAX_STORED_BYTES:,} bytes) reached; no manifest was published")
                self.stored_bytes += len(packed)
            self.b2.put(f"{PREFIX}/pages/{d}.z", packed)  # content-addressed: a PUT is idempotent, no Class-B HEAD needed
        list(self.pool.map(up, todo.items()))
        self._publish(self.gen + 1, pages, spages)
        with self.lk:
            self.dirty.clear(); self._reset_cache(); self._unpublished = False
        self.st["uploaded"] += len(todo)
        return self.gen

    def refresh(self): self._load(); return self.gen
    def stats(self):
        return {"backend": "Backblaze B2", "virtual_bytes": self.size, "virtual_gib": self.size / 1024**3,
                "page_bytes": self.page, "page_count": self.npages, "allocated_pages": len(self.page_map),
                "logical_allocated_bytes": len(self.page_map) * self.page, "stored_page_object_bytes": self.stored_bytes,
                "small_page_bytes": self.small, "small_tier_base": self.small_base, "allocated_small_pages": len(self.spage_map),
                "small_allocated_bytes": len(self.spage_map) * self.small,
                "dirty_pages": len(self.dirty), "generation": self.gen, "writer": self.writer,
                "cache_mib": round(self.cache_bytes / 1048576, 1), **{f"io_{k}": v for k, v in self.st.items()}}


def _write_stream(ram, path, address, batch_pages=8):
    path = Path(path); total = path.stat().st_size; done = 0; batch = max(1, batch_pages) * ram.page
    if address < 0 or address + total > ram.size: raise ValueError("GGUF does not fit in the 32-GiB virtual address space")
    with path.open("rb") as f:
        while True:
            block = f.read(batch)
            if not block: break
            ram.write(address + done, block); ram.flush(); done += len(block)
            print(f"B2 import: {done:,}/{total:,} bytes ({done*100/max(total,1):.1f}%)", flush=True)
    if done != total: raise IOError(f"short import: {done}/{total}")
    return done


def format_store(ram, gguf_path, kv_mb=1024, batch_pages=8):
    p = Path(gguf_path); model_size = p.stat().st_size
    layout = ram16.plan_layout(ram.npages, ram.page, model_size, kv_mb)
    ram.set_small_tier(layout["kv"][0] * ram.page)
    sha = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): sha.update(b)
    model = {"filename": p.name, "size": model_size, "sha256": sha.hexdigest(), "format": "GGUF"}
    ram.write(0, ram16.pack_super(layout, model)); ram.flush()
    _write_stream(ram, p, layout["weights"][0] * ram.page, batch_pages)
    print("formatted model:", json.dumps(model, indent=2))
    return model



def _source_headers(url):
    req = urllib.request.Request(url, headers={"User-Agent": "RAM16-B2/3.0"}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=180) as src:
            length = src.headers.get("Content-Length")
            return {"length": int(length) if length else None,
                    "etag": src.headers.get("ETag"),
                    "last_modified": src.headers.get("Last-Modified"),
                    "ranges": "bytes" in (src.headers.get("Accept-Ranges", "").lower())}
    except Exception:
        # Some model hosts reject HEAD. A zero-offset GET can still establish
        # the metadata; resumed imports will require a validated 206 response.
        req = urllib.request.Request(url, headers={"User-Agent": "RAM16-B2/3.0", "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=180) as src:
            cr = src.headers.get("Content-Range", "")
            import re
            m = re.match(r"bytes\s+0-0/(\d+)$", cr)
            length = int(m.group(1)) if m else src.headers.get("Content-Length")
            return {"length": int(length) if length else None,
                    "etag": src.headers.get("ETag"),
                    "last_modified": src.headers.get("Last-Modified"),
                    "ranges": bool(m)}


def _import_checkpoint_key(): return f"{PREFIX}/IMPORT.json"


def format_url(ram, url, kv_mb=1024, batch_pages=8, restart=False):
    """Resumable, bounded-memory HTTPS GGUF import with per-batch durable checkpoints."""
    meta = _source_headers(url)
    model_size = meta["length"]
    if not model_size: raise ValueError("model URL must expose a nonzero Content-Length")
    layout = ram16.plan_layout(ram.npages, ram.page, model_size, kv_mb)
    ram.set_small_tier(layout["kv"][0] * ram.page)
    identity = hashlib.sha256(url.encode()).hexdigest()
    checkpoint_key = _import_checkpoint_key()
    checkpoint = None
    try: checkpoint = json.loads(ram.b2.get(checkpoint_key))
    except KeyError: pass
    if restart and checkpoint:
        ram.b2.delete_current(checkpoint_key); checkpoint = None
    if checkpoint:
        if int(checkpoint.get("offset", 0)) and not (meta.get("etag") or meta.get("last_modified")):
            raise RuntimeError("cannot safely resume: source has no stable ETag or Last-Modified validator; use --restart")
        for field, expected in (("url_sha256", identity), ("size", model_size),
                                ("etag", meta.get("etag")), ("last_modified", meta.get("last_modified"))):
            if checkpoint.get(field) != expected:
                raise RuntimeError(f"existing import checkpoint does not match source ({field}); use --restart to abandon it")
        if checkpoint.get("layout") != layout:
            raise RuntimeError("existing import layout differs; use --restart to abandon it")
        done = int(checkpoint.get("offset", 0))
        if done < 0 or done > model_size or (done != model_size and done % (max(1, batch_pages) * ram.page)):
            raise IOError("import checkpoint offset is invalid")
        filename = checkpoint.get("filename", url.rsplit("/", 1)[-1] or "model.gguf")
        print(f"Resuming B2 import at byte {done:,}/{model_size:,}", flush=True)
    else:
        done = 0; filename = url.rsplit("/", 1)[-1] or "model.gguf"
        model_pending = {"filename": filename, "size": model_size,
                         "sha256": "pending-stream-verification", "format": "GGUF"}
        ram.write(0, ram16.pack_super(layout, model_pending)); ram.flush()
        checkpoint = {"format": 1, "url_sha256": identity, "size": model_size,
                      "etag": meta.get("etag"), "last_modified": meta.get("last_modified"),
                      "layout": layout, "filename": filename, "offset": 0,
                      "created_unix": int(time.time())}
        ram.b2.put(checkpoint_key, json.dumps(checkpoint, sort_keys=True).encode(), "application/json")

    # Rebuild the SHA state from already committed bytes. hashlib state is not
    # portable/serializable; bounded B2 reads make restart deterministic.
    h = hashlib.sha256(); weight_addr = layout["weights"][0] * ram.page
    hashed = 0; hash_chunk = max(1, batch_pages) * ram.page
    while hashed < done:
        n = min(hash_chunk, done - hashed); h.update(ram.read(weight_addr + hashed, n)); hashed += n
    headers = {"User-Agent": "RAM16-B2/3.0"}
    if done:
        headers["Range"] = f"bytes={done}-"
        if meta.get("etag"): headers["If-Range"] = meta["etag"]
        elif meta.get("last_modified"): headers["If-Range"] = meta["last_modified"]
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as src:
        response_etag = src.headers.get("ETag")
        response_modified = src.headers.get("Last-Modified")
        if meta.get("etag") and response_etag and meta["etag"] != response_etag:
            raise RuntimeError("model source ETag changed between metadata check and download")
        if meta.get("last_modified") and response_modified and meta["last_modified"] != response_modified:
            raise RuntimeError("model source Last-Modified changed between metadata check and download")
        if done:
            if getattr(src, "status", src.getcode()) != 206:
                raise RuntimeError("source did not honor HTTP Range resume; use --restart to start over")
            expected_prefix = f"bytes {done}-"
            if not src.headers.get("Content-Range", "").startswith(expected_prefix):
                raise IOError("resumed source Content-Range does not match requested offset")
        elif getattr(src, "status", src.getcode()) not in (200, 206):
            raise IOError("unexpected HTTP response for model source")
        batch = max(1, batch_pages) * ram.page
        while done < model_size:
            block = src.read(min(batch, model_size - done))
            if not block: break
            ram.write(weight_addr + done, block)
            h.update(block); done += len(block)
            # Data generation is durable before the import checkpoint advances.
            ram.flush()
            checkpoint["offset"] = done
            checkpoint["updated_unix"] = int(time.time())
            ram.b2.put(checkpoint_key, json.dumps(checkpoint, sort_keys=True).encode(), "application/json")
            print(f"B2 model stream: {done:,}/{model_size:,} bytes ({done*100/max(model_size,1):.1f}%)", flush=True)
        if done != model_size: raise IOError(f"model stream length mismatch: {done}/{model_size}")
    model = {"filename": filename, "size": model_size, "sha256": h.hexdigest(), "format": "GGUF"}
    ram.write(0, ram16.pack_super(layout, model)); ram.flush()
    ram.b2.delete_current(checkpoint_key)
    return model


def disk_preflight(model_size, path="/tmp/ram16-model.gguf", reserve_mb=1024):
    parent = Path(path).parent
    usage = shutil.disk_usage(parent)
    required = int(model_size) + int(reserve_mb) * 1024 * 1024
    result = {"path": str(path), "model_bytes": int(model_size), "reserve_bytes": int(reserve_mb) * 1024 * 1024,
              "free_bytes": usage.free, "required_bytes": required}
    if usage.free < required:
        raise OSError("insufficient runner disk before GGUF materialization: " + json.dumps(result))
    return result


def garbage_collect(ram, retain_generations=5, min_age_days=7, apply=False):
    """Conservative version-aware GC. Dry-run unless --apply is explicitly supplied."""
    if not ram.writer: raise PermissionError("GC requires writer access")
    if retain_generations < 1 or min_age_days < 1: raise ValueError("retain-generations must be >=1 and min-age-days must be >=1")
    now = int(time.time()); cutoff = now - int(min_age_days * 86400)
    manifest_rows = list(ram.b2.list(f"{PREFIX}/manifests/"))
    parsed = []
    for row in manifest_rows:
        key = row["Key"]
        try:
            raw = ram.b2.get(key); m = json.loads(raw)
            created = int(m.get("created_unix", row.get("LastModified", now).timestamp() if hasattr(row.get("LastModified"), "timestamp") else now))
            parsed.append((int(m["generation"]), created, key, m))
        except Exception as e:
            raise IOError(f"refusing GC: cannot validate manifest {key}: {e}") from e
    parsed.sort(key=lambda x: x[0], reverse=True)
    keep = {key for _, _, key, _ in parsed[:retain_generations]}
    keep.update(key for _, created, key, _ in parsed if created >= cutoff)
    keep.add(f"{PREFIX}/manifests/{ram.gen:012d}.json")
    referenced = set()
    for _, _, key, m in parsed:
        if key in keep:
            referenced.update(f"{PREFIX}/pages/{digest}.z" for digest in list(m.get("pages", {}).values()) + list((m.get("spages") or {}).values()))

    # Use the version listing, not only ListObjectsV2: B2 buckets are versioned
    # by default, and a simple delete can leave older versions consuming quota.
    all_versions = list(ram.b2.list_versions(PREFIX + "/"))
    versions_by_key = collections.defaultdict(list)
    for v in all_versions:
        if v.get("_kind") == "version": versions_by_key[v["Key"]].append(v)
    candidate_keys = set()
    for key, rows in versions_by_key.items():
        if key.startswith(f"{PREFIX}/pages/") and key not in referenced:
            if rows and all((r.get("LastModified").timestamp() if hasattr(r.get("LastModified"), "timestamp") else 0) < cutoff for r in rows):
                candidate_keys.add(key)
        elif key.startswith(f"{PREFIX}/manifests/") and key not in keep:
            meta = next((x for x in parsed if x[2] == key), None)
            if meta and meta[1] < cutoff:
                candidate_keys.add(key)
    candidate_versions = [v for k in candidate_keys for v in versions_by_key[k]]
    report = {"dry_run": not apply, "retained_manifests": len(keep),
              "candidate_objects": len(candidate_keys), "candidate_versions": len(candidate_versions),
              "candidate_page_bytes": sum(int(v.get("Size", 0)) for v in candidate_versions if v["Key"].startswith(f"{PREFIX}/pages/")),
              "min_age_days": min_age_days,
              "retained_generations": sorted(g for g, _, k, _ in parsed if k in keep)}
    if not apply: return report
    deleted = 0
    for v in candidate_versions:
        ram.b2.delete_version(v["Key"], v["VersionId"]); deleted += 1
    report["deleted_versions"] = deleted
    ram.stored_bytes = sum(int(x.get("Size", 0)) for x in ram.b2.list(f"{PREFIX}/pages/"))
    return report


def build_parser():
    ap = argparse.ArgumentParser(description="Original RAM16 GGUF/issue-chat runner backed by sparse 32-GiB Backblaze B2 paging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    f = sub.add_parser("format"); f.add_argument("--gguf", required=True); f.add_argument("--kv-mb", type=int, default=1024); f.add_argument("--batch-pages", type=int, default=8)
    fu = sub.add_parser("format-url"); fu.add_argument("url"); fu.add_argument("--kv-mb", type=int, default=1024); fu.add_argument("--batch-pages", type=int, default=8); fu.add_argument("--restart", action="store_true", help="abandon any existing resumable import checkpoint")
    gc = sub.add_parser("gc"); gc.add_argument("--retain-generations", type=int, default=5); gc.add_argument("--min-age-days", type=float, default=7); gc.add_argument("--apply", action="store_true", help="actually delete unreachable B2 object versions")
    s = sub.add_parser("serve"); s.add_argument("--minutes", type=float, default=330); s.add_argument("--label", default="ram16-chat"); s.add_argument("--max-tokens", type=int, default=512); s.add_argument("--idle", type=int, default=20); s.add_argument("--n-ctx", type=int, default=4096)
    s.add_argument("--cache-pages", type=int, default=32); s.add_argument("--handoff", action="store_true")
    c = sub.add_parser("commit")
    sub.add_parser("retier", help="switch an already-formatted store's KV/mem regions to small pages (no model re-import)")
    return ap


def main(argv=None):
    try: sys.stdout.reconfigure(line_buffering=True)  # Actions pipes stdout; without this the log looks frozen
    except Exception: pass
    args = build_parser().parse_args(argv)
    writer = args.cmd in ("init", "format", "format-url", "serve", "commit", "gc", "retier")
    ram = B2RAM(writer=writer, cache_pages=getattr(args, "cache_pages", 32))
    if args.cmd == "init":
        print(json.dumps(ram.stats(), indent=2))
    elif args.cmd == "status":
        if ram.read(0, 8) != b"RAM16MAP":
            print(json.dumps({**ram.stats(), "formatted": False, "message": "No GGUF model has been formatted yet."}, indent=2))
        else:
            m = ram16.Machine(ram)
            ready = len(m.sb["model"].get("sha256", "")) == 64
            print(json.dumps({**ram.stats(), "formatted": True, "model_ready": ready,
                              "model": m.sb["model"], "layout": m.sb["layout"],
                              "memory_keys": len(m.mem.keys()), "kv_keys": len(m.kv.keys())}, indent=2))
    elif args.cmd == "format":
        format_store(ram, args.gguf, args.kv_mb, args.batch_pages)
        print(json.dumps(ram.stats(), indent=2))
    elif args.cmd == "format-url":
        model = format_url(ram, args.url, args.kv_mb, args.batch_pages, args.restart)
        print(json.dumps({"model": model, **ram.stats()}, indent=2))
    elif args.cmd == "gc":
        print(json.dumps(garbage_collect(ram, args.retain_generations, args.min_age_days, args.apply), indent=2))
    elif args.cmd == "retier":
        if ram.read(0, 8) != b"RAM16MAP": raise SystemExit("store is not formatted yet; format-url sets the small tier itself")
        ram.set_small_tier(ram16.read_super(ram)["layout"]["kv"][0] * ram.page)
        print("published generation", ram.flush()); print(json.dumps(ram.stats(), indent=2))
    elif args.cmd == "commit":
        print("published generation", ram.flush())
    elif args.cmd == "serve":
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        repo = os.getenv("RAM16_REPO") or os.getenv("GITHUB_REPOSITORY")
        if not token or not repo: raise SystemExit("Set GH_TOKEN/GITHUB_TOKEN and GITHUB_REPOSITORY")
        if ram.small_base is None and ram.read(0, 8) == b"RAM16MAP":  # auto-upgrade a store formatted before small pages existed
            try:
                ram.set_small_tier(ram16.read_super(ram)["layout"]["kv"][0] * ram.page)
                print("KV/mem regions switched to small pages, generation", ram.flush(), flush=True)
            except RuntimeError as e: print("small-page tier not enabled:", e, flush=True)
        machine = ram16.Machine(ram)
        model_sha = machine.sb["model"].get("sha256", "")
        if len(model_sha) != 64 or any(c not in "0123456789abcdef" for c in model_sha.lower()):
            raise SystemExit("Model import is incomplete or unverified; rerun format-url to completion before serve.")
        model_size = int(machine.sb["model"].get("size", 0))
        print("runner disk preflight:", json.dumps(disk_preflight(model_size), indent=2), flush=True)
        queue = ram16.IssueQueue(repo, token, args.label)
        def factory(path): return ram16.llama_factory(path, n_ctx=args.n_ctx)
        ram16.serve(ram, machine, queue, args.minutes, llm_factory=factory,
                    journal=None, store=None, handoff_to=(f"{repo}:ram16-b2.yml" if args.handoff else None), token=token,
                    idle=args.idle, max_tokens=args.max_tokens)
        print(json.dumps(ram.stats(), indent=2))

if __name__ == "__main__": main()
