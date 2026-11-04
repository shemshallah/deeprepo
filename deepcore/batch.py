"""Batch chat: one independent turn per ring slot, no shared reporam writes.

The 24/7 ring serializes turns through the write lease (correct for handoff). Batch mode is
for throughput: each slot takes a different issue, keeps KV in local MemKV, and posts its
finished reply. Five slots = five concurrent conversations. Weights still page from repovram.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .tok import clean_reply

MARK = "<!--ram16-->"
CUR = " ▌"
BUSY = "deeprepo-busy"


class MemKV:
    """Sparse 4 MiB-paged KV in local RAM (batch only; lost with the VM)."""

    def __init__(s, page=4 << 20, pages=4096):
        s.page, s.cap = page, pages
        s.d = {}

    def _p(s, p):
        b = s.d.get(p)
        if b is None:
            b = s.d[p] = bytearray(s.page)
        return b

    def write(s, addr, data):
        i, n = 0, len(data)
        while i < n:
            p, o = addr // s.page, addr % s.page
            k = min(n - i, s.page - o)
            s._p(p)[o:o + k] = data[i:i + k]
            i += k
            addr += k

    def read(s, addr, n):
        out, i = bytearray(), 0
        while i < n:
            p, o = addr // s.page, addr % s.page
            k = min(n - i, s.page - o)
            out += s._p(p)[o:o + k]
            i += k
            addr += k
        return bytes(out)


def _api(repo, token, path, method="GET", body=None):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "deeprepo",
         "Authorization": f"Bearer {token}"}
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    url = f"https://api.github.com/repos/{repo}{path}"
    rq = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(rq, timeout=60) as r:
            b = r.read()
    except urllib.error.HTTPError as e:
        raise IOError(f"{e.code} {url}") from e
    return json.loads(b) if b else None


def messages_from(title, body, comments):
    """Pure parser: user/bot turns, unfinished bot comment -> handle, last role must be user."""
    msgs = [{"role": "user", "content": f"{title}\n\n{body or ''}".strip()}]
    handle = None
    for c in comments:
        b = c.get("body") or ""
        if b.startswith(MARK):
            t = b[len(MARK):]
            if t.endswith(CUR):
                handle = c.get("id")
                continue
            msgs.append({"role": "assistant", "content": t.strip()})
            handle = None
        else:
            msgs.append({"role": "user", "content": b.strip()})
            handle = None
    if msgs[-1]["role"] != "user":
        return None
    return {"messages": msgs, "handle": handle}


def fetch_issue(repo, token, n):
    issue = _api(repo, token, f"/issues/{n}")
    comments, page = [], 1
    while True:
        c = _api(repo, token, f"/issues/{n}/comments?per_page=100&page={page}")
        comments += c
        if len(c) < 100:
            break
        page += 1
    req = messages_from(issue.get("title", ""), issue.get("body"), comments)
    if not req:
        return None
    req["id"] = n
    return req


def pending(repo, token, label, limit=5):
    out = []
    for i in _api(repo, token, f"/issues?labels={label}&state=open&sort=created&direction=asc&per_page=50"):
        if "pull_request" in i:
            continue
        if any((l or {}).get("name") == BUSY for l in i.get("labels", [])):
            continue  # another slot owns this turn
        req = fetch_issue(repo, token, i["number"])
        if req:
            out.append(req)
        if len(out) >= limit:
            break
    return out


def _begin(repo, token, issue, handle):
    if handle:
        return handle
    c = _api(repo, token, f"/issues/{issue}/comments", "POST", {"body": MARK + CUR})
    return c["id"]


def _update(repo, token, handle, text, done=False):
    body = MARK + (text or "(no output)") + ("" if done else CUR)
    _api(repo, token, f"/issues/comments/{handle}", "PATCH", {"body": body})


def answer(env, issue, model_key=None, max_tokens=48, log=print):
    repo, token = env.main_repo, env.token
    try:
        if os.environ.get("DEEPREPO_FAST", "llama").lower() not in ("0", "numpy", "off"):
            try:
                return _answer_llama(env, repo, token, issue, model_key, max_tokens, log)
            except FileNotFoundError:
                log("batch: llama-server unavailable, falling back to numpy engine")
        return _answer(env, repo, token, issue, model_key, max_tokens, log)
    finally:
        try:
            _api(repo, token, f"/issues/{issue}/labels/{BUSY}", "DELETE")
        except Exception:
            pass


def _llama_server(path, model_path, port):
    """Start llama.cpp server with the materialized GGUF (ram16 pattern, CPU-only)."""
    env = os.environ.copy()
    root = os.path.dirname(path)
    env["LD_LIBRARY_PATH"] = root + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    logf = open(f"/tmp/llama-server-{port}.log", "wb")
    cmd = [path, "-m", model_path, "--host", "127.0.0.1", "--port", str(port),
           "-c", "8192", "-t", str(os.cpu_count() or 4), "-ngl", "0", "--jinja"]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    for _ in range(180):
        if proc.poll() is not None:
            logf.close()
            tail = open(f"/tmp/llama-server-{port}.log", "rb").read()[-1500:].decode("utf-8", "replace")
            raise RuntimeError(f"llama-server exited: {tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    proc._deeprepo_log = logf
                    return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill(); logf.close()
    raise RuntimeError("llama-server did not become healthy")


def _chat_stream(port, messages, max_tokens):
    payload = {"model": "local", "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.7, "top_p": 0.8, "stream": True}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(rq, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            text = delta.get("content")
            if text:
                yield text


def _answer_llama(env, repo, token, issue, model_key, max_tokens, log):
    """Fast tier exactly like ram16: materialize GGUF, then llama.cpp inference."""
    path = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("llama-server not installed (set LLAMA_SERVER)")
    req = fetch_issue(repo, token, int(issue))
    if not req:
        log(f"batch: issue {issue} has no pending user turn")
        return ""
    model_path = f"/tmp/deeprepo-{str(model_key or 'default')}.gguf"
    t0 = time.time()
    env.materialize_model(model_key, model_path)
    log(f"batch: materialized {model_path} in {time.time()-t0:.0f}s")
    port = int(os.environ.get("LLAMA_PORT", "18080"))
    proc = _llama_server(path, model_path, port)
    handle = _begin(repo, token, issue, req["handle"])
    text, last, first = "", 0.0, None
    made = 0
    try:
        for d in _chat_stream(port, req["messages"], max_tokens):
            text += d
            made += 1
            if first is None:
                first = time.time()
                log(f"batch: llama first token after {first-t0:.1f}s")
            if time.time() - last >= 3:
                _update(repo, token, handle, clean_reply(text))
                last = time.time()
        _update(repo, token, handle, clean_reply(text) or "(no output)", done=True)
    except Exception as e:
        _update(repo, token, handle, f"llama error: {e}", done=True)
        raise
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try: proc.kill()
            except Exception: pass
    dt = (time.time() - first) if first else 0
    log(f"batch: llama issue {issue} done in {time.time()-t0:.0f}s, {len(text)} chars, "
        f"{made} chunks, {made/max(dt,1e-6):.2f} chunks/s")
    return text


def _answer(env, repo, token, issue, model_key, max_tokens, log):
    req = fetch_issue(repo, token, int(issue))
    if not req:
        log(f"batch: issue {issue} has no pending user turn")
        return ""
    tok = env.tokenizer(model_key)
    from .tok import apply_no_think, clean_reply
    ids = tok.chat_ids(apply_no_think(req["messages"], model_key))
    kv = MemKV()
    mi = env.model_info(model_key) or {}
    f16 = 8000 if 0 < int(mi.get("size", 1)) < (6 << 30) else 1024
    tinfo = env.tune_vram_cache(model_key, f16)
    log(f"batch: cache {tinfo}")
    eng = env.engine(key=model_key, ramkv=kv,
                     kv_layout={"base": 0, "bytes": 4 << 30},
                     f16_cache_mb=f16, chunk=192)
    handle = _begin(repo, token, issue, req["handle"])
    text, last, t0 = "", time.time(), time.time()
    logits = None
    i = 0
    while i < len(ids):
        ch = ids[i:i + eng.chunk]
        logits = eng.forward(ch, i)
        i += len(ch)
    EOS = tok.special.get(tok.IM_END)
    rng = np.random.default_rng(int(time.time()))
    made = 0
    for _ in range(max_tokens):
        nid = eng.sample(logits, temperature=0.7, top_k=20, top_p=0.8, rng=rng)
        if nid == EOS:
            break
        text += tok.decode([nid])
        made += 1
        if time.time() - last >= 8:
            _update(repo, token, handle, text)
            last = time.time()
        if made >= max_tokens:
            break
        logits = eng.forward([nid], len(ids) + made - 1)
    _update(repo, token, handle, clean_reply(text), done=True)
    log(f"batch: issue {issue} done in {time.time()-t0:.0f}s, {len(text)} chars")
    return text


def serve(env, minutes=300, max_tokens=128, model_key=None, interval=8, log=print):
    """Fast 24/7 loop: one llama.cpp server, answers pending deeprepo-chat issues."""
    repo, token = env.main_repo, env.token
    key = model_key or os.environ.get("CHAT_MODEL") or None
    path = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("llama-server not installed (set LLAMA_SERVER)")
    if not key:
        raise IOError("fast-serve needs CHAT_MODEL or --model")
    model_path = f"/tmp/deeprepo-{key}.gguf"
    t0 = time.time()
    env.materialize_model(key, model_path)
    log(f"fast: materialized {model_path} in {time.time()-t0:.0f}s")
    port = int(os.environ.get("LLAMA_PORT", "18080"))
    proc = _llama_server(path, model_path, port)
    deadline = time.time() + max(1, minutes) * 60
    log(f"fast: serving {key} for {minutes}m on port {port}")
    try:
        while time.time() < deadline:
            reqs = pending(repo, token, "deeprepo-chat", limit=1)
            if not reqs:
                time.sleep(interval)
                continue
            issue = reqs[0]["id"]
            try:
                _api(repo, token, f"/issues/{issue}/labels", "POST", {"labels": [BUSY]})
            except Exception as e:
                log(f"fast: could not mark {issue} busy: {e!r}")
                time.sleep(interval)
                continue
            handle = None
            try:
                req = fetch_issue(repo, token, issue)
                if not req:
                    continue
                handle = _begin(repo, token, issue, req["handle"])
                text, last, first = "", 0.0, None
                made = 0
                for d in _chat_stream(port, req["messages"], max_tokens):
                    text += d
                    made += 1
                    if first is None:
                        first = time.time()
                        log(f"fast: issue {issue} first token after {first-t0:.1f}s")
                    if time.time() - last >= 3:
                        _update(repo, token, handle, clean_reply(text))
                        last = time.time()
                _update(repo, token, handle, clean_reply(text) or "(no output)", done=True)
                dt = (time.time() - first) if first else 0
                log(f"fast: issue {issue} done, {made} chunks, {made/max(dt,1e-6):.2f} chunks/s")
            except Exception as e:
                log(f"fast: issue {issue} failed: {e!r}")
                if handle:
                    try: _update(repo, token, handle, f"llama error: {e}", done=True)
                    except Exception: pass
            finally:
                try: _api(repo, token, f"/issues/{issue}/labels/{BUSY}", "DELETE")
                except Exception: pass
    finally:
        try:
            proc.terminate(); proc.wait(timeout=10)
        except Exception:
            try: proc.kill()
            except Exception: pass
        try: getattr(proc, "_deeprepo_log", None).close()
        except Exception: pass
        log("fast: server stopped")


def dispatch(env, slots, limit=5, model_key=None, max_tokens=48, log=print):
    repo, token = env.main_repo, env.token
    reqs = pending(repo, token, "deeprepo-chat", limit=limit)
    if not reqs:
        log("batch: no pending issues")
        return []
    sent = []
    for slot, req in zip(slots, reqs):
        from .ring import dispatch_workflow
        try:
            _api(repo, token, f"/issues/{req['id']}/labels", "POST", {"labels": [BUSY]})
        except Exception as e:
            log(f"batch: could not mark issue {req['id']} busy: {e!r}")
            continue
        dispatch_workflow(slot, "runner.yml", token, {"op": "chat", "issue": str(req["id"]),
                                                      "model": model_key or "", "minutes": "60",
                                                      "max_tokens": str(max_tokens)})
        sent.append({"slot": slot, "issue": req["id"]})
        log(f"batch: dispatched issue {req['id']} -> {slot}")
    return sent
