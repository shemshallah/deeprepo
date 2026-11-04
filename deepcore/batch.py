"""Batch chat: one independent turn per ring slot, no shared reporam writes.

The 24/7 ring serializes turns through the write lease (correct for handoff). Batch mode is
for throughput: each slot takes a different issue, keeps KV in local MemKV, and posts its
finished reply. Five slots = five concurrent conversations. Weights still page from repovram.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

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
        return _answer(env, repo, token, issue, model_key, max_tokens, log)
    finally:
        try:
            _api(repo, token, f"/issues/{issue}/labels/{BUSY}", "DELETE")
        except Exception:
            pass


def _answer(env, repo, token, issue, model_key, max_tokens, log):
    req = fetch_issue(repo, token, int(issue))
    if not req:
        log(f"batch: issue {issue} has no pending user turn")
        return ""
    tok = env.tokenizer(model_key)
    from .tok import apply_no_think
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
    _update(repo, token, handle, text, done=True)
    log(f"batch: issue {issue} done in {time.time()-t0:.0f}s, {len(text)} chars")
    return text


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
                                                      "model": model_key or "", "minutes": "60"})
        sent.append({"slot": slot, "issue": req["id"]})
        log(f"batch: dispatched issue {req['id']} -> {slot}")
    return sent
