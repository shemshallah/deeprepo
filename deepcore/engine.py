"""Out-of-core Qwen3-MoE inference in numpy.

Everything big lives in the repo tiers:
  repovram (L2): GGUF weights, read as 4 MiB pages through an LRU-capped Ram16 (L0/L1 pages).
  reporam  (L1): KV cache rows + all session/ring state, written as compressed page deltas.
The runner keeps only bounded caches; a model far larger than any runner is still inferable.

Follows src/models/qwen3moe.cpp: RMSNorm, Q/K per-head RMSNorm (before NEOX rope), GQA,
SiLU-gated experts with softmax top-k renorm, merged 3-D expert tensors (per-expert slices
are byte-contiguous -> per-expert paging).
"""
import collections
import json
import os
import time

import numpy as np

from . import gguf as G

SILU = None  # silu() below is the implementation


def rms(x, w, eps):
    v = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(v + eps)) * w.astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_neox(x, positions, base, half):
    """x: (..., n_heads, hd) f32; positions: (T,) int. NEOX: half-split pairs."""
    i = np.arange(half, dtype=np.float32)
    theta = base ** (-2.0 * i / (2 * half))          # (hd/2,)
    ang = positions[:, None].astype(np.float32) * theta[None, :]  # (T, hd/2)
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]   # (T,1,hd/2)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


class Qwen3MoE:
    def __init__(s, vram, wbase_page, model_size, ramkv=None, kv_layout=None,
                 f16_cache_mb=2560, chunk=256, strict=False, expert_cache_mb=1024):
        s.v, s.kv = vram, ramkv
        s.g = G.GGUFOnRam(vram, wbase_page, model_size)
        meta, arch = s.g.meta, s.g.meta["general.architecture"]
        assert arch in ("qwen3moe", "qwen3"), f"arch {arch} unsupported"
        s.arch = arch
        s.L = meta[f"{arch}.block_count"]
        s.d = meta[f"{arch}.embedding_length"]
        s.nh = meta[f"{arch}.attention.head_count"]
        s.nkv = meta[f"{arch}.attention.head_count_kv"]
        s.hd = int(meta.get(f"{arch}.attention.key_length", s.d // s.nh))
        s.eps = float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
        s.dense = arch == "qwen3"
        s.ne = int(meta.get(f"{arch}.expert_count", 1))
        s.nu = int(meta.get(f"{arch}.expert_used_count", 1))
        s.nff = int(meta.get(f"{arch}.expert_feed_forward_length",
                              meta.get(f"{arch}.feed_forward_length", 0)))
        s.rope = float(meta.get(f"{arch}.rope.freq_base", 1e6))
        s.nvoc = int(meta.get("tokenizer.ggml.tokens_count", meta.get("tokenizer.vocab_size", 0))) or len(meta.get("tokenizer.ggml.tokens", []))
        s.chunk = chunk
        s.strict = strict
        s.scale = 1.0 / np.sqrt(s.hd)
        s.half = s.hd // 2
        s.rep = s.nkv * s.hd * 2          # K row bytes (f16)
        # KV layout inside reporam kv region
        s.kv_base = int(kv_layout["base"]) if kv_layout else 0
        s.kv_bytes = int(kv_layout["bytes"]) if kv_layout else 0
        s.kv_stride = (s.kv_bytes // (2 * s.L)) & ~(4095) if s.kv_bytes else 0
        s.ctx = min(int(meta.get(f"{arch}.context_length", 0)) or 1 << 30,
                    (s.kv_stride // 2) // s.rep if s.kv_stride else 4096)
        # L0 dequant cache (f16)
        s.cache, s.cache_bytes, s.cache_cap = {}, 0, int(f16_cache_mb) << 20
        # hot-expert cache: frequency-aware, survives across tokens (routing is skewed)
        s.ecache, s.ecache_bytes = {}, 0
        s.ecache_cap = int(expert_cache_mb) << 20
        s.ecount = collections.Counter()
        s.head = None  # output head f16, loaded in bounded chunks (never an f32 copy)
        # per-expert slice geometry from merged 3-D tensors
        s.expert_bytes = {}
        if not s.dense:
            for nm in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                t = s.g.tensors[f"blk.0.{nm}.weight"]
                s.expert_bytes[nm] = t.nbytes // s.ne
        s.st = dict(vram_reads=0, vram_bytes=0, expert_slice_reads=0, expert_bytes=0,
                    kv_writes=0, kv_read_bytes=0, l0_hits=0, l0_misses=0,
                    expert_cache_hits=0, expert_cache_misses=0, tokens=0, seconds=0.0)

    # ---------- weight access ----------

    def _w(s, name, cache=True):
        """Dequantized f32 (optionally f16-cached) weights for small/attention tensors."""
        c = s.cache.get(name) if cache else None
        if c is not None:
            s.st["l0_hits"] += 1
            return c
        s.st["l0_misses"] += 1
        t = s.g.tensors[name]
        s.st["vram_reads"] += 1
        s.st["vram_bytes"] += t.nbytes
        w = G.dequant(s.g.read_tensor(name), t.type).reshape([int(d) for d in reversed(t.dims)])
        if s.strict or not cache:
            return w
        w16 = w.astype(np.float16)
        nb = w16.nbytes
        while s.cache_bytes + nb > s.cache_cap and s.cache:
            k0 = next(iter(s.cache))
            s.cache_bytes -= s.cache[k0].nbytes
            del s.cache[k0]
        s.cache[name] = w16
        s.cache_bytes += nb
        return w16

    def _mm(s, x, wname):
        w = s._w(wname)
        return (x @ w.astype(np.float32).T) if x.ndim == 2 else x @ w.astype(np.float32).T

    def _expert(s, layer, nm, e):
        """Dequantize one expert slice of a merged 3-D tensor, paged straight from repovram.
        Slice size is derived from THIS tensor (Q4_K_M mixes Q4_K/Q6_K per layer).
        Hot experts are kept in an LFU f16 cache to cut repeat paging/dequant."""
        key = (layer, nm, e)
        c = s.ecache.get(key)
        if c is not None:
            s.ecount[key] += 1
            s.st["expert_cache_hits"] += 1
            return c.astype(np.float32)
        s.st["expert_cache_misses"] += 1
        t = s.g.tensors[f"blk.{layer}.{nm}.weight"]
        ne_t = int(t.dims[-1])                      # expert count on the last axis
        per = t.nbytes // ne_t                      # contiguous expert slice
        off, n = e * per, per
        s.st["expert_slice_reads"] += 1
        s.st["expert_bytes"] += n
        s.st["vram_bytes"] += n
        w = G.dequant(s.g.read_range(f"blk.{layer}.{nm}.weight", off, n, fine=getattr(s, "_fine_decode", False)), t.type)
        rows = int(t.dims[1])                       # gate/up: n_ff; down: n_embd
        w = w.reshape(rows, int(t.dims[0]))
        s.ecount[key] += 1
        # admit only on second use: cold one-off experts would just churn the cache
        if not s.strict and s.ecache_cap > 0 and s.ecount[key] >= 2:
            w16 = w.astype(np.float16)
            while s.ecache_bytes + w16.nbytes > s.ecache_cap and s.ecache:
                k0 = min(s.ecache, key=lambda k: s.ecount[k])
                s.ecache_bytes -= s.ecache[k0].nbytes
                del s.ecache[k0]
            s.ecache[key] = w16
            s.ecache_bytes += w16.nbytes
        return w

    def _load_head(s):
        """Dequantize the output (lm) head into f16 in bounded chunks: no 1.2 GiB f32 transient."""
        t = s.g.tensors["output.weight"]
        cols = int(t.dims[0])
        rows = int(t.dims[-1])
        be, bb = G.GGML[t.type]
        brow = cols // be * bb
        s.head = np.empty((rows, cols), np.float16)
        step = 4096
        for c in range(0, rows, step):
            n = min(step, rows - c)
            blob = s.g.read_range("output.weight", c * brow, n * brow)
            s.head[c:c + n] = G.dequant(blob, t.type).reshape(n, cols).astype(np.float16)
            s.st["vram_bytes"] += n * brow
            s.st["vram_reads"] += 1

    def _logits(s, xn):
        """Streaming head: bounded f32 chunks over the f16 head cache (f32 dequant in strict mode)."""
        if "output.weight" not in s.g.tensors:
            W = s.g.gather_rows("token_embd.weight", np.arange(s.nvoc))
            return (xn @ W.T)[0]
        if s.strict:
            t = s.g.tensors["output.weight"]
            cols = int(t.dims[0]); rows = int(t.dims[-1])
            be, bb = G.GGML[t.type]; brow = cols // be * bb
            out = np.empty(rows, np.float32)
            step = 4096
            for c in range(0, rows, step):
                n = min(step, rows - c)
                blob = s.g.read_range("output.weight", c * brow, n * brow)
                w = G.dequant(blob, t.type).reshape(n, cols)
                out[c:c + n] = (xn @ w.T)[0]
                s.st["vram_bytes"] += n * brow
            return out
        if s.head is None:
            s._load_head()
        out = np.empty(s.nvoc, np.float32)
        step = 8192
        for c in range(0, s.nvoc, step):
            n = min(step, s.nvoc - c)
            out[c:c + n] = (xn @ s.head[c:c + n].astype(np.float32).T)[0]
        return out

    def prefetch(s, name):
        t = s.g.tensors.get(name)
        if t is not None:
            s.v.prefetch(t.addr, t.nbytes)

    def _embd_rows(s, ids):
        s.st["vram_reads"] += 1
        return s.g.gather_rows("token_embd.weight", ids)

    # ---------- KV through reporam ----------

    def _kv_addr(s, layer, side, pos):
        base = s.kv_base + layer * s.kv_stride + (s.kv_stride // 2 if side == "v" else 0)
        return base + pos * s.rep

    def _kv_put(s, layer, pos, k, v):
        if s.kv is None:
            return
        s.kv.write(s._kv_addr(layer, "k", pos), k.astype("<f2").tobytes())
        s.kv.write(s._kv_addr(layer, "v", pos), v.astype("<f2").tobytes())
        s.st["kv_writes"] += k.nbytes + v.nbytes

    def _kv_hist(s, layer, upto):
        """[0, upto) K/V history rows: (upto, nkv, hd) f32 each."""
        if upto <= 0 or s.kv is None:
            return None, None
        ka = s._kv_addr(layer, "k", 0)
        kb = s._kv_addr(layer, "k", upto)
        K = np.frombuffer(s.kv.read(ka, (kb - ka)), "<f2").astype(np.float32).reshape(upto, s.nkv, s.hd)
        va = s._kv_addr(layer, "v", 0)
        V = np.frombuffer(s.kv.read(va, (kb - ka)), "<f2").astype(np.float32).reshape(upto, s.nkv, s.hd)
        s.st["kv_read_bytes"] += 2 * (kb - ka)
        return K, V

    # ---------- forward ----------

    def _attn(s, l, X, pos0, T):
        """X: (C,d) input rows at positions pos0..pos0+C-1. Returns (C,d) attention out."""
        C = X.shape[0]
        q = (X @ s._w(f"blk.{l}.attn_q.weight").astype(np.float32).T).reshape(C, s.nh, s.hd)
        k = (X @ s._w(f"blk.{l}.attn_k.weight").astype(np.float32).T).reshape(C, s.nkv, s.hd)
        v = (X @ s._w(f"blk.{l}.attn_v.weight").astype(np.float32).T).reshape(C, s.nkv, s.hd)
        qw = s._w(f"blk.{l}.attn_q_norm.weight").astype(np.float32)
        kw = s._w(f"blk.{l}.attn_k_norm.weight").astype(np.float32)
        q = rms(q, qw, s.eps)
        k = rms(k, kw, s.eps)
        pos = np.arange(pos0, pos0 + C)
        q = rope_neox(q, pos, s.rope, s.half)
        k = rope_neox(k, pos, s.rope, s.half)
        s._kv_put(l, pos0, k, v)  # writes C rows per side
        K, V = s._kv_hist(l, pos0)              # past, fetched ONCE per layer
        T_past = 0 if K is None else K.shape[0]
        out = np.empty((C, s.nh, s.hd), np.float32)
        for h in range(s.nkv):
            kh = k[:, h, :]
            vh = v[:, h, :]
            Kfull = np.concatenate([K[:, h, :], kh], axis=0) if K is not None else kh
            Vfull = np.concatenate([V[:, h, :], vh], axis=0) if V is not None else vh
            qh = q[:, h * (s.nh // s.nkv):(h + 1) * (s.nh // s.nkv), :]  # (C, g, hd)
            scores = np.einsum("cgd,td->cgt", qh, Kfull) * s.scale       # (C, g, T)
            mask = np.triu(np.ones((C, C), bool), 1)
            scores[:, :, T_past:] = np.where(mask[:, None, :], -np.inf, scores[:, :, T_past:])
            p = softmax(scores, axis=-1)
            out[:, h * (s.nh // s.nkv):(h + 1) * (s.nh // s.nkv), :] = np.einsum("cgt,td->cgd", p, Vfull)
        wo = s._w(f"blk.{l}.attn_output.weight").astype(np.float32)
        return out.reshape(C, s.nh * s.hd) @ wo.T

    def _mlp(s, l, X):
        C, d = X.shape
        if C > 4:
            for nm in ("ffn_gate", "ffn_up", "ffn_down"):
                s.prefetch(f"blk.{l}.{nm}.weight")
        wg = s._w(f"blk.{l}.ffn_gate.weight").astype(np.float32)
        wu = s._w(f"blk.{l}.ffn_up.weight").astype(np.float32)
        h = silu(X @ wg.T) * (X @ wu.T)
        wd = s._w(f"blk.{l}.ffn_down.weight").astype(np.float32)
        return h @ wd.T

    def _moe(s, l, X):
        if s.dense:
            return s._mlp(l, X)
        C, d = X.shape
        router = X @ s._w(f"blk.{l}.ffn_gate_inp.weight").astype(np.float32).T  # (C, ne)
        probs = softmax(router)
        top = np.argsort(-probs, axis=1)[:, :s.nu]                              # (C, nu)
        topv = np.take_along_axis(probs, top, axis=1)
        topv = topv / topv.sum(axis=1, keepdims=True)                           # renorm
        out = np.zeros((C, d), np.float32)
        s._fine_decode = C <= 4   # decode only: 1 MiB fine reads beat 4 MiB overfetch
        if C > 4:  # prefill: fetch the layer's expert tensors in few large runs
            for nm in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                s.prefetch(f"blk.{l}.{nm}.weight")
        else:
            # decode: issue all selected expert slice fetches in parallel before dequant,
            # so latency overlaps instead of serializing 24 small range requests per layer
            used = np.unique(top)
            for e in used:
                for nm in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                    name = f"blk.{l}.{nm}.weight"
                    t = s.g.tensors[name]
                    per = t.nbytes // int(t.dims[-1])
                    s.g.prefetch_range(name, int(e) * per, per)
        for e in range(s.ne):
            rows = np.nonzero((top == e).any(axis=1))[0]
            if rows.size == 0:
                continue
            w_gate = s._expert(l, "ffn_gate_exps", e)
            w_up = s._expert(l, "ffn_up_exps", e)
            w_down = s._expert(l, "ffn_down_exps", e)
            xe = X[rows]
            h = silu(xe @ w_gate.T) * (xe @ w_up.T)
            oe = h @ w_down.T
            pe = np.array([topv[r, list(top[r]).index(e)] for r in rows], np.float32)
            out[rows] += pe[:, None] * oe
        return out

    def forward(s, ids, pos0):
        """Prefill/decode ids at positions pos0.. Return logits for the last position."""
        X = s._embd_rows(ids)
        progress = os.environ.get("DEEPREPO_PROGRESS") == "1"
        t_layer = time.time()
        for l in range(s.L):
            xn = rms(X, s._w(f"blk.{l}.attn_norm.weight").astype(np.float32), s.eps)
            attn = s._attn(l, xn, pos0, xn.shape[0])
            X = X + attn
            xn2 = rms(X, s._w(f"blk.{l}.ffn_norm.weight").astype(np.float32), s.eps)
            X = X + s._moe(l, xn2)
            if progress and (l % 8 == 7 or l == s.L - 1):
                print(f"layer {l+1}/{s.L} {(time.time()-t_layer)/8:.2f}s/layer "
                      f"vram={s.st['vram_bytes']/2**30:.1f}GiB", flush=True)
        xn = rms(X[-1:], s._w("output_norm.weight").astype(np.float32), s.eps)
        logits = s._logits(xn)
        s.st["tokens"] += len(ids)
        return logits

    # ---------- sampling ----------

    def sample(s, logits, temperature=0.7, top_k=20, top_p=0.8, penalty=1.05, recent=(), rng=None):
        lg = np.asarray(logits, np.float32).copy()
        if recent:
            uniq, cnt = np.unique(np.asarray(list(recent)[-64:], np.int64), return_counts=True)
            for u, c in zip(uniq, cnt):
                lg[u] = lg[u] / penalty if lg[u] > 0 else lg[u] * penalty
        if temperature <= 0:
            return int(np.argmax(lg))
        lg = lg / temperature
        if top_k and top_k < len(lg):
            k = np.argpartition(-lg, top_k - 1)[:top_k]
            keep = np.full(len(lg), -np.inf)
            keep[k] = lg[k]
            lg = keep
        p = softmax(lg)
        if top_p < 1.0:
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            cut = order[np.searchsorted(cum, top_p) + 1:]
            p[cut] = 0.0
            p /= p.sum()
        return int(rng.choice(len(p), p=p)) if rng is not None else int(np.argmax(p))

    def stats(s):
        d = dict(s.st)
        d["l0_cache_bytes"] = s.cache_bytes
        d["l0_cache_tensors"] = len(s.cache)
        d["l0_cap_bytes"] = s.cache_cap
        d["expert_cache_bytes"] = s.ecache_bytes
        d["expert_cache_entries"] = len(s.ecache)
        try:
            d["fine"] = s.g.fine.stats() if s.g.fine else None
        except Exception:
            d["fine"] = None
        return d