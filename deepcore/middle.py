"""middle: a tiny self-hostable mutable blob/page store for deeprepo tiers.

Why: hosted runner VMs cannot accept inbound connections and their disks are wiped, so a
runner cannot serve its working-tree bins to other runners. If you want *low-latency mutable
shared memory* between runners without the release-journal round trip, run this middleman on
any host you control (VPS, Fly/Render/Railway, a Raspberry Pi behind a tunnel...). It speaks
HTTP with token auth and exposes:

  GET/HEAD  /v1/blob/<name>[?off=N]     ranged read (206) or full read
  PUT       /v1/blob/<name>?off=N       write bytes at an offset (sparse files)
  POST      /v1/blob/<name>             replace the whole blob
  DELETE    /v1/blob/<name>
  GET       /v1/list?prefix=            JSON list of names+sizes
  GET       /healthz

The deeprepo client (`MiddleStore`) plugs into the same Ram16 pager as the GitHub release
tiers, so you can run reporam here while repovram stays on GitHub (or both here).

Run:
  MIDDLE_TOKEN=... python -m deepcore.middle serve --root ./middle-data --port 8080
"""
import base64
import hashlib
import json
import os
import re
import sqlite3
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = 4 << 20


# ============================== server ====================================

class Store:
    def __init__(s, root):
        s.root = root
        s.blobs = os.path.join(root, "blobs")
        os.makedirs(s.blobs, exist_ok=True)
        s.db = sqlite3.connect(os.path.join(root, "index.sqlite"), check_same_thread=False)
        s.db.execute("CREATE TABLE IF NOT EXISTS blobs(name TEXT PRIMARY KEY, size INTEGER, updated INTEGER)")
        s.db.commit()
        s.lk = threading.Lock()

    def _path(s, name):
        return os.path.join(s.blobs, hashlib.sha1(name.encode()).hexdigest())

    def size(s, name):
        with s.lk:
            r = s.db.execute("SELECT size FROM blobs WHERE name=?", (name,)).fetchone()
        if r:
            return int(r[0])
        p = s._path(name)
        return os.path.getsize(p) if os.path.exists(p) else None

    def write(s, name, off, data):
        p = s._path(name)
        with s.lk:
            exists = os.path.exists(p)
            with open(p, "r+b" if exists else "wb") as f:
                if not exists:
                    f.truncate(max(off + len(data), 0))
                f.seek(off)
                f.write(data)
                end = f.tell()
                if f.seek(0, 2) < end:
                    f.truncate(end)
            new_size = os.path.getsize(p)
            s.db.execute("INSERT INTO blobs(name,size,updated) VALUES(?,?,?) "
                         "ON CONFLICT(name) DO UPDATE SET size=?, updated=?",
                         (name, new_size, int(time.time()), new_size, int(time.time())))
            s.db.commit()

    def read(s, name, off, n):
        p = s._path(name)
        with s.lk:
            if not os.path.exists(p):
                return None
            with open(p, "rb") as f:
                f.seek(off)
                return f.read(n)

    def delete(s, name):
        p = s._path(name)
        with s.lk:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            s.db.execute("DELETE FROM blobs WHERE name=?", (name,))
            s.db.commit()

    def list(s, prefix=""):
        with s.lk:
            rows = s.db.execute("SELECT name,size FROM blobs WHERE name LIKE ? ORDER BY name",
                                (prefix.replace("%", "") + "%",)).fetchall()
        return [{"name": n, "size": int(sz)} for n, sz in rows]


def make_handler(store, token, log):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _auth(s):
            if not token:
                return True
            got = s.headers.get("Authorization", "")
            return got == f"Bearer {token}"

        def _send(s, code, body=b"", ctype="application/octet-stream", extra=None):
            s.send_response(code)
            s.send_header("Content-Type", ctype)
            s.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                s.send_header(k, v)
            s.end_headers()
            if s.command != "HEAD":
                s.wfile.write(body)

        def _name(s):
            path = urllib.parse.urlparse(s.path)
            m = re.match(r"^/v1/blob/(.+)$", path.path)
            return urllib.parse.unquote(m.group(1)) if m else None, path

        def do_HEAD(s):
            name, _ = s._name()
            if name is None:
                return s._send(404)
            n = store.size(name)
            if n is None:
                return s._send(404)
            s._send(200, b"", extra={"X-Blob-Size": str(n)})

        def do_GET(s):
            name, path = s._name()
            if name is None:
                if path.path == "/healthz":
                    return s._send(200, b'{"ok":true}', "application/json")
                if path.path == "/v1/list":
                    if not s._auth():
                        return s._send(401)
                    q = urllib.parse.parse_qs(path.query)
                    return s._send(200, json.dumps(store.list((q.get("prefix") or [""])[0])).encode(),
                                   "application/json")
                return s._send(404)
            if not s._auth():
                return s._send(401)
            n = store.size(name)
            if n is None:
                return s._send(404)
            rng = s.headers.get("Range")
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                off = int(a or 0)
                end = int(b) if b else n - 1
                end = min(end, n - 1)
                if off > end or off >= n:
                    return s._send(416)
                data = store.read(name, off, end - off + 1) or b""
                return s._send(206, data, extra={"Content-Range": f"bytes {off}-{off+len(data)-1}/{n}"})
            data = store.read(name, 0, n) or b""
            return s._send(200, data)

        def do_PUT(s):
            name, path = s._name()
            if name is None or not s._auth():
                return s._send(401 if name else 404)
            length = int(s.headers.get("Content-Length", "0"))
            data = s.rfile.read(length)
            q = urllib.parse.parse_qs(path.query)
            off = int((q.get("off") or ["0"])[0])
            store.write(name, off, data)
            log(f"PUT {name} @{off} {len(data)}")
            return s._send(200, b'{"ok":true}', "application/json")

        def do_POST(s):
            name, _ = s._name()
            if name is None or not s._auth():
                return s._send(401 if name else 404)
            length = int(s.headers.get("Content-Length", "0"))
            left = length
            tmp = store._path(name) + ".part"
            with open(tmp, "wb") as f:
                while left > 0:
                    b = s.rfile.read(min(1 << 20, left))
                    if not b:
                        break
                    f.write(b)
                    left -= len(b)
            os.replace(tmp, store._path(name))
            store.write(name, 0, b"")  # update index/size
            log(f"POST {name} {length}")
            return s._send(200, b'{"ok":true}', "application/json")

        def do_DELETE(s):
            name, _ = s._name()
            if name is None or not s._auth():
                return s._send(401 if name else 404)
            store.delete(name)
            log(f"DELETE {name}")
            return s._send(200, b'{"ok":true}', "application/json")

        def log_message(s, *a):
            pass

    return H


def serve(root, token="", port=8080, host="0.0.0.0", log=print):
    store = Store(root)
    srv = ThreadingHTTPServer((host, port), make_handler(store, token, log))
    log(f"middle listening on {host}:{port} root={root} auth={'on' if token else 'OFF (dev only)'}")
    srv.serve_forever()


# ============================== client =====================================

class MiddleAsset:
    def __init__(s, store, name, size):
        s.store, s.name, s.size = store, name, size

    def read(s, off, n, tries=5):
        for k in range(tries):
            try:
                return s.store._range(s.name, off, n)
            except urllib.error.HTTPError as e:
                if e.code in (404, 416):
                    raise
                if k == tries - 1:
                    raise
            except (urllib.error.URLError, OSError):
                if k == tries - 1:
                    raise
            time.sleep(min(0.25 * 2 ** k, 4))


class MiddleStore:
    """Client for deepcore.middle. Duck-types GitHubStore for Ram16 + journals."""

    def __init__(s, base_url, token="", prefix=""):
        s.base = base_url.rstrip("/")
        s.token = token
        s.prefix = (prefix.rstrip("/") + "/") if prefix else ""
        s.tag = prefix.rsplit("/", 1)[-1] or prefix
        s.repo = prefix
        s._sizes = {}

    def _h(s):
        h = {"User-Agent": "deeprepo-middle"}
        if s.token:
            h["Authorization"] = f"Bearer {s.token}"
        return h

    def _req(s, method, path, data=None, headers=None):
        h = dict(s._h());
        h.update(headers or {})
        rq = urllib.request.Request(f"{s.base}{path}", data=data, headers=h, method=method)
        with urllib.request.urlopen(rq, timeout=120) as r:
            return r.read()

    def _full(s, name):
        return s.prefix + name

    def size(s, name):
        full = s._full(name)
        if full in s._sizes:
            return s._sizes[full]
        h = s._h()
        rq = urllib.request.Request(f"{s.base}/v1/blob/{urllib.parse.quote(full)}", headers=h, method="HEAD")
        try:
            with urllib.request.urlopen(rq, timeout=60) as r:
                n = int(r.headers.get("X-Blob-Size") or r.headers.get("Content-Length") or 0)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        s._sizes[full] = n
        return n

    def asset(s, name):
        n = s.size(name)
        if n is None:
            raise KeyError(name)
        return MiddleAsset(s, name, n)

    def _range(s, name, off, n):
        full = s._full(name)
        h = s._h()
        h["Range"] = f"bytes={off}-{off + n - 1}"
        rq = urllib.request.Request(f"{s.base}/v1/blob/{urllib.parse.quote(full)}", headers=h)
        with urllib.request.urlopen(rq, timeout=120) as r:
            b = r.read()
        if len(b) != n:
            raise IOError(f"short read {len(b)}/{n} {full}")
        return b

    def get(s, name):
        return s._req("GET", f"/v1/blob/{urllib.parse.quote(s._full(name))}")

    def put(s, name, data):
        s._req("PUT", f"/v1/blob/{urllib.parse.quote(s._full(name))}?off=0", data=data,
               headers={"Content-Type": "application/octet-stream"})
        s._sizes.pop(s._full(name), None)

    def put_file(s, name, path):
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            rq = urllib.request.Request(f"{s.base}/v1/blob/{urllib.parse.quote(s._full(name))}",
                                        data=f, headers=dict(s._h(), **{
                                            "Content-Type": "application/octet-stream",
                                            "Content-Length": str(size)}), method="POST")
            with urllib.request.urlopen(rq, timeout=3600) as r:
                r.read()
        s._sizes.pop(s._full(name), None)

    def delete(s, name):
        try:
            s._req("DELETE", f"/v1/blob/{urllib.parse.quote(s._full(name))}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        s._sizes.pop(s._full(name), None)

    def list(s, pat=r"d-\d{8}\.dlt"):
        return [(n, s.asset(n)) for n in s.list_names(pat)]

    def list_names(s, pat=r".*"):
        raw = s._req("GET", f"/v1/list?prefix={urllib.parse.quote(s.prefix)}")
        out = []
        for item in json.loads(raw):
            n = item["name"]
            if s.prefix and n.startswith(s.prefix):
                n = n[len(s.prefix):]
            if re.fullmatch(pat, n):
                out.append(n)
        return sorted(out)

    def ensure_release(s, tag=None):
        return None

    def close(s):
        pass


class MiddleTierStore(MiddleStore):
    """Bin tier view: raw bytes via ranges, forged headers for absent bins (sparse)."""

    def __init__(s, base_url, token="", prefix="", geom=None, disk_cache=None):
        super().__init__(base_url, token, prefix)
        s.geom, s.disk_cache = geom or {}, disk_cache

    def asset(s, name):
        from .store import CachedAsset, SparseAsset
        from .geom import forge_header
        slot = int(name[4:6]) if name.startswith("mem-") and name[4:6].isdigit() else 0
        h = forge_header(slot, page=s.geom.get("page", PAGE), pps=s.geom.get("pps", 510),
                         hdr=s.geom.get("hdr", PAGE), ns=s.geom.get("ns", 16))
        n = s.size(name)
        if n is None:
            return SparseAsset(slot, h)
        real = MiddleAsset(s, name, n)
        if s.disk_cache is not None:
            real = CachedAsset(real, name, s.disk_cache, n)
        return SparseAsset(slot, h, real=real, real_size=n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve"], nargs="?", default="serve")
    ap.add_argument("--root", default="./middle-data")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    serve(a.root, os.environ.get("MIDDLE_TOKEN", ""), a.port, a.host)
