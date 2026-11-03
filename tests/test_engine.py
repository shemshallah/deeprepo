"""End-to-end: the engine must produce the same logits as a dense in-memory reference,
while all weights stream through sparse repovram bins and all KV goes through reporam
journal pages. Then: handoff across writer instances, trimmed compaction, re-hydration.
"""
import os
import shutil
import tempfile

import numpy as np

from deepcore import PAGE
from deepcore.engine import Qwen3MoE, rms, silu, softmax
from deepcore.geom import plan_ram_layout
from deepcore.tiers import compact_trimmed, forge_bin0

import sys
sys.path.insert(0, os.path.dirname(__file__))
from fixtures import TINY, dense_from_gguf, make_ram, make_tiny_gguf, tile_model


# ---------------- independent dense reference ----------------

def rope_ref(x, pos, base, half):
    i = np.arange(half, dtype=np.float32)
    ang = pos[:, None].astype(np.float32) * (base ** (-2.0 * i / (2 * half)))[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def dense_forward(w, c, ids, pos0=0, kvstore=None):
    L, d, nh, nkv, hd, nff, ne, nu = c["L"], c["d"], c["nh"], c["nkv"], c["hd"], c["nff"], c["ne"], c["nu"]
    half = hd // 2
    scale = 1.0 / np.sqrt(hd)
    X = w["token_embd.weight"][ids].astype(np.float32)
    C = len(ids)
    pos = np.arange(pos0, pos0 + C)
    for l in range(L):
        xn = rms(X, w[f"blk.{l}.attn_norm.weight"], c["eps"])
        q = (xn @ w[f"blk.{l}.attn_q.weight"].T).reshape(C, nh, hd)
        k = (xn @ w[f"blk.{l}.attn_k.weight"].T).reshape(C, nkv, hd)
        v = (xn @ w[f"blk.{l}.attn_v.weight"].T).reshape(C, nkv, hd)
        q = rms(q, w[f"blk.{l}.attn_q_norm.weight"], c["eps"])
        k = rms(k, w[f"blk.{l}.attn_k_norm.weight"], c["eps"])
        q, k = rope_ref(q, pos, c["rope"], half), rope_ref(k, pos, c["rope"], half)
        att = np.empty((C, nh, hd), np.float32)
        for h in range(nkv):
            Kh = kvstore["K"][l][:, h, :] if kvstore and kvstore["K"][l] is not None else np.zeros((0, hd), np.float32)
            Vh = kvstore["V"][l][:, h, :] if kvstore and kvstore["V"][l] is not None else np.zeros((0, hd), np.float32)
            Kf = np.concatenate([Kh, k[:, h, :]], 0)
            Vf = np.concatenate([Vh, v[:, h, :]], 0)
            qh = q[:, h * (nh // nkv):(h + 1) * (nh // nkv), :]
            sc = np.einsum("cgd,td->cgt", qh, Kf) * scale
            mask = np.triu(np.ones((C, C), bool), 1)
            sc[:, :, Kh.shape[0]:] = np.where(mask[:, None, :], -np.inf, sc[:, :, Kh.shape[0]:])
            pr = softmax(sc, -1)
            att[:, h * (nh // nkv):(h + 1) * (nh // nkv), :] = np.einsum("cgt,td->cgd", pr, Vf)
        X = X + att.reshape(C, nh * hd) @ w[f"blk.{l}.attn_output.weight"].T
        xn2 = rms(X, w[f"blk.{l}.ffn_norm.weight"], c["eps"])
        router = softmax(xn2 @ w[f"blk.{l}.ffn_gate_inp.weight"].T, -1)
        top = np.argsort(-router, axis=1)[:, :nu]
        tv = np.take_along_axis(router, top, axis=1)
        tv = tv / tv.sum(1, keepdims=True)
        moe = np.zeros((C, d), np.float32)
        for e in range(ne):
            rows = np.nonzero((top == e).any(1))[0]
            if not rows.size:
                continue
            Wg = w[f"blk.{l}.ffn_gate_exps.weight"][e]
            Wu = w[f"blk.{l}.ffn_up_exps.weight"][e]
            Wd = w[f"blk.{l}.ffn_down_exps.weight"][e]
            h = silu(xn2[rows] @ Wg.T) * (xn2[rows] @ Wu.T)
            pe = np.array([tv[r, list(top[r]).index(e)] for r in rows], np.float32)
            moe[rows] += pe[:, None] * (h @ Wd.T)
        X = X + moe
    xn = rms(X[-1:], w["output_norm.weight"], c["eps"])
    return (xn @ w["output.weight"].T)[0]


# ---------------- fixture setup ----------------

def setup(tmp, gtype=0, seed=7):
    gguf = os.path.join(tmp, "tiny.gguf")
    make_tiny_gguf(gguf, seed=seed, gtype=gtype)
    size = os.path.getsize(gguf)
    vram_dir = os.path.join(tmp, "vram")
    ram_dir = os.path.join(tmp, "ram")
    jdir = os.path.join(tmp, "journal")
    wp = tile_model(gguf, vram_dir, None)
    os.makedirs(ram_dir, exist_ok=True)
    if not os.path.exists(os.path.join(ram_dir, "mem-00.bin")):
        forge_bin0(os.path.join(ram_dir, "mem-00.bin"), "ram", plan_ram_layout(16 * 510, kv_pages=64, reg_pages=8))
    vram = make_ram(vram_dir, cache_pages=16)
    ram = make_ram(ram_dir, journal_dir=jdir, writer=True, cache_pages=64)
    from deepcore.geom import read_super
    lay = read_super(ram)["layout"]
    kv = {"base": lay["kv"][0] * PAGE, "bytes": lay["kv"][1] * PAGE}
    return gguf, size, vram, ram, kv, wp


def test_engine_matches_dense():
    tmp = tempfile.mkdtemp()
    try:
        gguf, size, vram, ram, kv, wp = setup(tmp)
        meta, w = dense_from_gguf(gguf)
        c = dict(TINY)
        eng = Qwen3MoE(vram, 1, size, ramkv=ram, kv_layout=kv, f16_cache_mb=4, strict=True)
        ids = [5, 9, 23, 71, 3, 44]
        got = eng.forward(ids, 0)
        ref = dense_forward(w, c, ids, 0)
        assert np.allclose(got, ref, atol=1e-4, rtol=1e-5), f"max diff {np.abs(got-ref).max()}"
        # one more decode step
        got2 = eng.forward([17], len(ids))
        assert np.allclose(got2, dense_forward(w, c, ids + [17], 0), atol=8e-3, rtol=1e-3)  # fp16 KV round-trip
        print(f"  tiny L={c['L']} d={c['d']} experts={c['ne']}: logits match dense reference")
    finally:
        shutil.rmtree(tmp)


def test_kv_lives_in_reporam():
    tmp = tempfile.mkdtemp()
    try:
        gguf, size, vram, ram, kv, wp = setup(tmp)
        c = dict(TINY)
        eng = Qwen3MoE(vram, 1, size, ramkv=ram, kv_layout=kv, f16_cache_mb=4)
        ids = [1, 2, 3, 4]
        eng.forward(ids, 0)
        row = c["nkv"] * c["hd"] * 2
        # first KV row of layer 0 is nonzero in reporam
        b = ram.read(kv["base"], row)
        assert any(b), "KV row should be written"
        assert eng.st["kv_writes"] > 0
        ram.flush()  # write-back: publish dirty KV pages as a journal delta
        # a fresh reader (new Ram16 instance, same journal) sees the same KV after refresh
        ram2 = make_ram(os.path.join(tmp, "ram"), journal_dir=os.path.join(tmp, "journal"), writer=False, cache_pages=8)
        assert ram2.read(kv["base"], row) == b
        # untouched KV region far ahead is implicit zeros (sparse)
        assert ram2.read(kv["base"] + 40 * PAGE, 4096) == b"\0" * 4096
        print("  KV bytes live in reporam journal; untouched pages stay sparse zeros")
    finally:
        shutil.rmtree(tmp)


def test_handoff_and_compaction():
    tmp = tempfile.mkdtemp()
    try:
        gguf, size, vram, ram, kv, wp = setup(tmp)
        c = dict(TINY)
        meta, w = dense_from_gguf(gguf)
        prompt = [11, 12, 13]
        eng = Qwen3MoE(vram, 1, size, ramkv=ram, kv_layout=kv, f16_cache_mb=4)
        lg = eng.forward(prompt, 0)
        toks = []
        for step in range(3):
            t = int(np.argmax(lg)); toks.append(t)
            lg = eng.forward([t], len(prompt) + step)
        # graceful shift end: flush, then drop the process-local engine (crash-safe state is in reporam)
        ram.flush()
        del eng
        # fresh writer instance = the next ring runner
        ram2 = make_ram(os.path.join(tmp, "ram"), journal_dir=os.path.join(tmp, "journal"), writer=True, cache_pages=64)
        eng2 = Qwen3MoE(vram, 1, size, ramkv=ram2, kv_layout=kv, f16_cache_mb=4)
        lg2 = eng2.forward([toks[-1]], len(prompt) + 2)   # idempotent re-encode of last token
        cont = []
        for step in range(3):
            t = int(np.argmax(lg2)); cont.append(t)
            lg2 = eng2.forward([t], len(prompt) + 3 + step)
        # continuous reference run for the same 6 tokens
        ram3 = make_ram(os.path.join(tmp, "ram_c"), journal_dir=os.path.join(tmp, "journal_c"), writer=True, cache_pages=64)
        # journal_c has no history; replay prompt + toks
        eng3 = Qwen3MoE(vram, 1, size, ramkv=ram3, kv_layout=kv, f16_cache_mb=4)
        lg3 = eng3.forward(prompt, 0)
        allt = []
        for step in range(6):
            t = int(np.argmax(lg3)); allt.append(t)
            lg3 = eng3.forward([t], len(prompt) + step)
        assert allt[:3] == toks and allt[3:] == cont, (toks, cont, allt)
        # compaction folds deltas into trimmed bins
        ram2.flush()
        compact_trimmed(ram2, work=tmp)
        # a fresh reader resolves the published manifest generation and must see the same KV
        ram5 = make_ram(os.path.join(tmp, "ram"), journal_dir=os.path.join(tmp, "journal"), writer=False, cache_pages=8)
        assert ram5.read(kv["base"], 256) == ram2.read(kv["base"], 256), "compacted bins must carry KV"
        names = [n for n in ram5.names if ".g" in n]
        assert names, f"expected a compacted generation bin, got {ram5.names[:3]}"
        print(f"  handoff continuity ok (tokens {toks}+{cont}); compaction+rehydrate ok (bin {names[0]})")
    finally:
        shutil.rmtree(tmp)


def test_sparse_bins_are_absent():
    tmp = tempfile.mkdtemp()
    try:
        gguf, size, vram, ram, kv, wp = setup(tmp)
        files = [f for f in os.listdir(os.path.join(tmp, "vram")) if f.startswith("mem-")]
        # tiny model must not materialize all 16 bins
        assert len(files) < 16, files
        assert len(files) >= 1
        # reading far past the model in repovram is implicit zero
        assert vram.read(vram.size - PAGE, PAGE) == b"\0" * PAGE
        print(f"  sparse: only {len(files)}/16 vram bins exist for the tiny model")
    finally:
        shutil.rmtree(tmp)


def test_prefix_kv_reuse_matches_full_prefill():
    """Encoding prompt then suffix at pos=len(prompt) must match a fresh full prefill."""
    tmp = tempfile.mkdtemp()
    try:
        gguf, size, vram, ram, kv, wp = setup(tmp)
        prompt, suffix = [5, 9, 23], [71, 3, 44]
        eng = Qwen3MoE(vram, 1, size, ramkv=ram, kv_layout=kv, f16_cache_mb=4, strict=True)
        full = eng.forward(prompt + suffix, 0)
        eng.forward(prompt, 0)
        reused = eng.forward(suffix, len(prompt))
        assert np.allclose(full, reused, atol=8e-3, rtol=1e-3), np.abs(full - reused).max()
        print("  prefix KV reuse matches full prefill (fp16 KV tolerance)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_engine_matches_dense()
    test_kv_lives_in_reporam()
    test_handoff_and_compaction()
    test_sparse_bins_are_absent()
    test_prefix_kv_reuse_matches_full_prefill()
    print("engine tests passed")