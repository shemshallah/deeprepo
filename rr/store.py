"""Stores: DirStore (tests/local) and GitHubReleaseStore (public Range reads, metered uploads).
Interface: size(repo,rel,name) read(repo,rel,name,off,n) put(repo,rel,name,path) has(repo,rel,name) list(repo,rel)"""
import json, os, shutil, time, urllib.request, urllib.error, urllib.parse


class DirStore:
    def __init__(s, root): s.root = root

    def _p(s, repo, rel, name):
        return os.path.join(s.root, repo.replace("/", "__"), rel, name)

    def has(s, repo, rel, name): return os.path.exists(s._p(repo, rel, name))

    def size(s, repo, rel, name): return os.path.getsize(s._p(repo, rel, name))

    def read(s, repo, rel, name, off, n):
        with open(s._p(repo, rel, name), "rb") as f:
            f.seek(off); return f.read(n)

    def put(s, repo, rel, name, path):
        d = s._p(repo, rel, name); os.makedirs(os.path.dirname(d), exist_ok=True)
        tmp = d + ".part"; shutil.copyfile(path, tmp); os.replace(tmp, d)

    def put_bytes(s, repo, rel, name, data):
        d = s._p(repo, rel, name); os.makedirs(os.path.dirname(d), exist_ok=True)
        tmp = d + ".part"
        with open(tmp, "wb") as f: f.write(data)
        os.replace(tmp, d)

    def list(s, repo, rel):
        d = os.path.dirname(s._p(repo, rel, "x"))
        return sorted(x for x in os.listdir(d) if not x.endswith(".part")) if os.path.isdir(d) else []

    def delete(s, repo, rel, name):
        try: os.remove(s._p(repo, rel, name))
        except FileNotFoundError: pass


class _NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None


class GitHubReleaseStore:
    """Release assets. Reads: public browser_download_url + Range (redirect target cached until expiry).
    Writes: uploads.github.com, metered. NOTE: untested here (sandbox has no GitHub write token); exercised by workflows."""
    def __init__(s, token=None, meter=None, ua="deeprepo"):
        s.token, s.meter, s.ua = token, meter, ua
        s._sig = {}; s._rel = {}

    def _api(s, method, url, body=None, ctype="application/json", cls="core", raw=False):
        for _ in range(6):
            if s.meter: s.meter.acquire(cls)
            h = {"User-Agent": s.ua, "Accept": "application/vnd.github+json"}
            if s.token: h["Authorization"] = f"Bearer {s.token}"
            if body is not None: h["Content-Type"] = ctype
            data = body if isinstance(body, (bytes, type(None))) or hasattr(body, "read") else json.dumps(body).encode()
            rq = urllib.request.Request(url, data=data, headers=h, method=method)
            try:
                with urllib.request.urlopen(rq, timeout=120) as r:
                    b = r.read(); return b if raw else (json.loads(b) if b else None)
            except urllib.error.HTTPError as e:
                if s.meter and s.meter.observe(cls, e.code, dict(e.headers)): continue
                if e.code >= 500: time.sleep(2); continue
                raise
        raise IOError("api retries exhausted")

    def release(s, repo, tag):
        k = (repo, tag)
        if k not in s._rel: s._rel[k] = s._api("GET", f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
        return s._rel[k]

    def ensure_release(s, repo, tag):
        try: return s.release(repo, tag)
        except urllib.error.HTTPError as e:
            if e.code != 404: raise
        r = s._api("POST", f"https://api.github.com/repos/{repo}/releases",
                   dict(tag_name=tag, name=tag, draft=False, prerelease=True), cls="write")
        s._rel[(repo, tag)] = r; return r

    def list(s, repo, rel):
        r = s.release(repo, rel); out = []
        for a in r["assets"]:
            if a.get("state") == "uploaded": out.append(a["name"])
        return sorted(out)

    def _asset(s, repo, rel, name):
        for a in s.release(repo, rel)["assets"]:
            if a["name"] == name and a.get("state") == "uploaded": return a
        return None

    def has(s, repo, rel, name): return s._asset(repo, rel, name) is not None

    def size(s, repo, rel, name):
        a = s._asset(repo, rel, name)
        if not a: raise FileNotFoundError(name)
        return a["size"]

    def _url(s, repo, rel, name):
        k = (repo, rel, name); v = s._sig.get(k)
        if v and v[1] > time.time(): return v[0]
        pub = f"https://github.com/{repo}/releases/download/{urllib.parse.quote(rel)}/{urllib.parse.quote(name)}"
        if s.meter: s.meter.acquire("download")
        try:
            urllib.request.build_opener(_NoRedir).open(urllib.request.Request(pub, headers={"User-Agent": s.ua}), timeout=30)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307):
                s._sig[k] = (e.headers["Location"], time.time() + 240); return e.headers["Location"]
            raise
        return pub

    def read(s, repo, rel, name, off, n):
        for t in range(5):
            u = s._url(repo, rel, name)
            if s.meter: s.meter.acquire("download")
            rq = urllib.request.Request(u, headers={"User-Agent": s.ua, "Range": f"bytes={off}-{off + n - 1}"})
            try:
                with urllib.request.urlopen(rq, timeout=60) as r:
                    b = r.read()
                    if len(b) == n: return b
            except urllib.error.HTTPError as e:
                if e.code in (403, 404): s._sig.pop((repo, rel, name), None)
            except Exception: pass
            time.sleep(min(2 ** t, 20))
        raise IOError(f"range read failed {name}@{off}")

    def put(s, repo, rel, name, path):
        r = s.ensure_release(repo, rel)
        old = s._asset(repo, rel, name)
        if old: s._api("DELETE", f"https://api.github.com/repos/{repo}/releases/assets/{old['id']}", cls="write", raw=True)
        up = r["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(name)
        with open(path, "rb") as f:
            for _ in range(3):
                f.seek(0)
                rq = urllib.request.Request(up, data=f, method="POST", headers={
                    "Authorization": f"Bearer {s.token}", "Content-Type": "application/octet-stream",
                    "Content-Length": str(os.path.getsize(path)), "User-Agent": s.ua})
                if s.meter: s.meter.acquire("upload")
                try:
                    urllib.request.urlopen(rq, timeout=3600).read(); break
                except urllib.error.HTTPError as e:
                    if s.meter and s.meter.observe("upload", e.code, dict(e.headers)): continue
                    raise
        s._rel.pop((repo, rel), None)
