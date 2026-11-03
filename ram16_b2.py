#!/usr/bin/env python3
"""RAM16 + Backblaze B2: original GGUF/issue-chat/memory runner on sparse 32-GiB paging.

The original ram16.py remains the authoritative model, GGUF, GitHub issue queue,
layout, and RegionLog implementation. This module supplies a B2-backed RAM
object with the same byte-addressed interface expected by Machine/serve.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, threading, time, zlib, urllib.request, urllib.error, collections, shutil, uuid, datetime
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
    """Sparse 32-GiB RAM16-compatible byte store; 4-MiB immutable content-addressed pages."""
    def __init__(self, backend=None, writer=False, cache_pages=32):
        self.b2 = backend or B2Store(); self.writer = writer
        self.page, self.size, self.npages = PAGE_SIZE, VIRTUAL_SIZE, PAGE_COUNT
        self._check_writer_authority()
        self.cap, self.ra, self.last = max(2, cache_pages), 0, -2
        self.lk = threading.RLock(); self.cache = collections.OrderedDict(); self.dirty = {}
        self.pool = None; self.journal = None; self.jmap = {}; self.gen = 0
        self.page_map = {}; self._unpublished = False
        self.stored_bytes = sum(int(x.get("Size", 0)) for x in self.b2.list(f"{PREFIX}/pages/"))
        self._load()

    def _check_writer_authority(self):
        if not self.writer: return
        coordinator = os.getenv("RAM16_COORDINATOR_REPO", "").strip()
        current = os.getenv("GITHUB_REPOSITORY", "").strip()
        if os.getenv("RAM16_ALLOW_LOCAL_WRITES") == "1" and not current:
            return  # explicit escape hatch for controlled local administration/tests
        if not coordinator:
            raise PermissionError("set the GitHub repository variable RAM16_COORDINATOR_REPO to the canonical owner/repo; direct writers are disabled")
        if current.lower() != coordinator.lower():
            raise PermissionError(f"B2 writes are centralized in {coordinator}; dispatch the RAM16 workflow there instead of writing from {current or 'a non-coordinator environment'}")

    def _load(self):
        try: ptr_raw = self.b2.get(f"{PREFIX}/CURRENT")
        except KeyError:
            if not self.writer: raise RuntimeError("RAM16 B2 store is not initialized; run `init` or `format --gguf` as writer")
            self._publish(0, {}); return
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
        for k, digest in pages.items():
            if not str(k).isdigit() or not 0 <= int(k) < self.npages or not isinstance(digest, str) or len(digest) != 64:
                raise IOError("B2 manifest contains an invalid page reference")
        self.gen, self.page_map = int(m["generation"]), dict(pages)

    def _page(self, p):
        if not 0 <= p < self.npages: raise ValueError("virtual address out of range")
        if p in self.dirty: return self.dirty[p]
        if p in self.cache:
            self.cache.move_to_end(p); return self.cache[p]
        digest = self.page_map.get(str(p))
        if digest is None: data = ZERO_PAGE
        else:
            packed = self.b2.get(f"{PREFIX}/pages/{digest}.z")
            data = zlib.decompress(packed)
            if len(data) != self.page or hashlib.sha256(data).hexdigest() != digest:
                raise IOError(f"B2 page {p} failed size/hash verification")
        self._put(p, data); return data

    def _put(self, p, data):
        self.cache[p] = data; self.cache.move_to_end(p)
        while len(self.cache) > self.cap: self.cache.popitem(last=False)

    def _get(self, p): return self._page(p)
    def prefetch(self, addr, n):
        # Explicit no-op: B2 is object storage; synchronous reads are bounded by caller windows.
        return None

    def read(self, addr, n):
        if addr < 0 or n < 0 or addr + n > self.size: raise AssertionError("out of range")
        if n == 0: return b""
        out = bytearray(); end = addr + n
        while addr < end:
            p, off = divmod(addr, self.page); take = min(end - addr, self.page - off)
            out.extend(self._page(p)[off:off + take]); addr += take
        return bytes(out)

    def write(self, addr, data):
        if not self.writer: raise PermissionError("B2 RAM opened read-only")
        data = memoryview(data); end = addr + len(data)
        if addr < 0 or end > self.size: raise AssertionError("out of range")
        pos = 0
        while addr + pos < end:
            absolute = addr + pos; p, off = divmod(absolute, self.page)
            take = min(end - absolute, self.page - off)
            if off == 0 and take == self.page: page = bytes(data[pos:pos + take])
            else:
                b = bytearray(self._page(p)); b[off:off + take] = data[pos:pos + take]; page = bytes(b)
            with self.lk: self.dirty[p] = page; self.cache.pop(p, None)
            pos += take

    def _publish(self, generation, pages):
        # Optimistic stale-writer detection. GitHub Actions concurrency in the
        # coordinator repository is the serialization authority; this check
        # catches stale snapshots but is not a storage-provider atomic CAS.
        if generation > 0:
            try:
                current = json.loads(self.b2.get(f"{PREFIX}/CURRENT"))
                if int(current.get("generation", -1)) != generation - 1:
                    raise RuntimeError("stale B2 writer: CURRENT advanced; refresh and retry")
            except KeyError:
                if generation != 0: raise RuntimeError("B2 CURRENT disappeared during commit")
        m = {"format": FORMAT, "generation": generation, "virtual_size": self.size,
             "page_size": self.page, "page_count": self.npages, "pages": pages,
             "created_unix": int(time.time())}
        raw = json.dumps(m, sort_keys=True, separators=(",", ":")).encode()
        key = f"{PREFIX}/manifests/{generation:012d}.json"
        self.b2.put(key, raw, "application/json")
        ptr = json.dumps({"generation": generation, "manifest": key,
                          "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True).encode()
        self.b2.put(f"{PREFIX}/CURRENT", ptr, "application/json")
        self.gen, self.page_map = generation, dict(pages)

    def flush(self):
        if not self.writer: raise PermissionError("B2 RAM opened read-only")
        if not self.dirty and not self._unpublished: return self.gen
        pages = dict(self.page_map); uploaded = 0
        for p, data in sorted(self.dirty.items()):
            if data == ZERO_PAGE: pages.pop(str(p), None); continue
            digest = hashlib.sha256(data).hexdigest(); key = f"{PREFIX}/pages/{digest}.z"
            if not self.b2.exists(key):
                packed = zlib.compress(data, 1)
                if self.stored_bytes + len(packed) > MAX_STORED_BYTES:
                    raise RuntimeError(f"B2 page-object safety cap ({MAX_STORED_BYTES:,} bytes) reached; no manifest was published")
                self.b2.put(key, packed); self.stored_bytes += len(packed); uploaded += 1
            pages[str(p)] = digest
        self._publish(self.gen + 1, pages)
        self.dirty.clear(); self.cache.clear(); self._unpublished = False
        return self.gen

    def refresh(self): self._load(); self.cache.clear(); return self.gen
    def stats(self):
        return {"backend": "Backblaze B2", "virtual_bytes": self.size, "virtual_gib": self.size / 1024**3,
                "page_bytes": self.page, "page_count": self.npages, "allocated_pages": len(self.page_map),
                "logical_allocated_bytes": len(self.page_map) * self.page, "stored_page_object_bytes": self.stored_bytes,
                "dirty_pages": len(self.dirty), "generation": self.gen, "writer": self.writer}


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
            referenced.update(f"{PREFIX}/pages/{digest}.z" for digest in m.get("pages", {}).values())

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
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    writer = args.cmd in ("init", "format", "format-url", "serve", "commit", "gc")
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
    elif args.cmd == "commit":
        print("published generation", ram.flush())
    elif args.cmd == "serve":
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        repo = os.getenv("RAM16_REPO") or os.getenv("GITHUB_REPOSITORY")
        if not token or not repo: raise SystemExit("Set GH_TOKEN/GITHUB_TOKEN and GITHUB_REPOSITORY")
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
