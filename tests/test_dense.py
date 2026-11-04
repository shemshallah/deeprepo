"""Dense Qwen3 fast-tier path: engine logits must match an independent dense reference."""
import os
import shutil
import sys
import tempfile

import numpy as np

from deepcore import PAGE
from deepcore.engine import Qwen3MoE, rms, silu, softmax
from deepcore.geom import plan_ram_layout, read_super
from deepcore.tiers import forge_bin0

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import TINY_DENSE, dense_from_gguf, make_ram, make_tiny_dense_gguf, tile_model
from test_engine import rope_ref


def dense_ref(w, c, ids, pos0=0):
    L, d, nh, nkv, hd, nff = c["L"], c["d"], c["nh"], c["nkv"], c["hd"], c["nff"]
    half, scale = hd // 2, 1.0 / np.sqrt(hd)
    X = w["token_embd.weight"][ids].astype(np.float32)
    C, pos = len(ids), np.arange(pos0, pos0 + len(ids))
    for l in range(L):
        xn = rms(X, w[f"blk.{l}.attn_norm.weight"], c["eps"])
        q = rms((xn @ w[f"blk.{l}.attn_q.weight"].T).reshape(C, nh, hd),
                w[f"blk.{l}.attn_q_norm.weight"], c["eps"])
        k = rms((xn @ w[f"blk.{l}.attn_k.weight"].T).reshape(C, nkv, hd),
                w[f"blk.{l}.attn_k_norm.weight"], c["eps"])
        v = (xn @ w[f"blk.{l}.attn_v.weight"].T).reshape(C, nkv, hd)
        q, k = rope_ref(q, pos, c["rope"], half), rope_ref(k, pos, c["rope"], half)
        att = np.empty((C, nh, hd), np.float32)
        mask = np.triu(np.ones((C, C), bool), 1)
        for h in range(nkv):
            Kf = k[:, h, :]
            Vf = v[:, h, :]
            qh = q[:, h * (nh // nkv):(h + 1) * (nh // nkv), :]
            sc = np.einsum("cgd,td->cgt", qh, Kf) * scale
            sc = np.where(mask[:, None, :], -np.inf, sc)
            att[:, h * (nh // nkv):(h + 1) * (nh // nkv), :] = np.einsum("cgt,td->cgd", softmax(sc, -1), Vf)
        X = X + att.reshape(C, nh * hd) @ w[f"blk.{l}.attn_output.weight"].T
        xn2 = rms(X, w[f"blk.{l}.ffn_norm.weight"], c["eps"])
        h = silu(xn2 @ w[f"blk.{l}.ffn_gate.weight"].T) * (xn2 @ w[f"blk.{l}.ffn_up.weight"].T)
        X = X + h @ w[f"blk.{l}.ffn_down.weight"].T
    xn = rms(X[-1:], w["output_norm.weight"], c["eps"])
    return (xn @ w["output.weight"].T)[0]


def test_dense_engine_matches():
    tmp = tempfile.mkdtemp()
    try:
        gguf = os.path.join(tmp, "dense.gguf")
        make_tiny_dense_gguf(gguf)
        size = os.path.getsize(gguf)
        vdir, rdir = os.path.join(tmp, "vram"), os.path.join(tmp, "ram")
        os.makedirs(vdir); os.makedirs(rdir)
        forge_bin0(os.path.join(vdir, "mem-00.bin"), "vram", {"weights": [1, 8159]})
        forge_bin0(os.path.join(rdir, "mem-00.bin"), "ram", plan_ram_layout(8160, 64, 8))
        tile_model(gguf, vdir, None)
        vram = make_ram(vdir, cache_pages=16)
        ram = make_ram(rdir, journal_dir=os.path.join(tmp, "j"), writer=True, cache_pages=32)
        lay = read_super(ram)["layout"]
        kv = {"base": lay["kv"][0] * PAGE, "bytes": lay["kv"][1] * PAGE}
        meta, w = dense_from_gguf(gguf)
        c = dict(TINY_DENSE)
        eng = Qwen3MoE(vram, 1, size, ramkv=ram, kv_layout=kv, strict=True, f16_cache_mb=4)
        ids = [5, 12, 33, 4]
        got = eng.forward(ids, 0)
        ref = dense_ref(w, c, ids, 0)
        assert np.allclose(got, ref, atol=1e-4, rtol=1e-5), np.abs(got - ref).max()
        print("  dense qwen3 engine matches dense reference")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_dense_engine_matches()
    print("dense tests passed")
