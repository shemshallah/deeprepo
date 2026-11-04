"""Local fixture tiers: real ram16.Ram16 paging over file-backed bins + a file journal,
so the entire engine path (sparse bins, deltas, compaction, re-hydration) is exercised
without any network. Also builds a tiny random Qwen3-MoE GGUF for end-to-end tests.
"""
import os
import re
import shutil

import numpy as np

import ram16
from deepcore import BINS, BIN_HDR, PAGE, PPS
from deepcore.geom import forge_header
from deepcore.store import SparseAsset, slot_list
from deepcore.tiers import forge_bin0


class LocalAsset:
    def __init__(s, path):
        s.path = path
        s.size = os.path.getsize(path)

    def read(s, off, n, tries=1):
        if off >= s.size:
            raise IOError(f"read past EOF {s.path} {off}>={s.size}")
        with open(s.path, "rb") as f:
            f.seek(off)
            b = f.read(min(n, s.size - off))
        if len(b) < n:
            raise IOError(f"short read {s.path} {off}+{n} (size {s.size})")
        return b


class LocalBins:
    def __init__(s, root):
        s.root = root
        os.makedirs(root, exist_ok=True)

    def _p(s, name):
        return os.path.join(s.root, name)

    def asset(s, name):
        slot = int(name[4:6]) if name.startswith("mem-") and name[4:6].isdigit() else 0
        h = forge_header(slot, page=PAGE, pps=PPS, hdr=BIN_HDR, ns=BINS)
        pth = s._p(name)
        if os.path.exists(pth):
            return SparseAsset(slot, h, real=LocalAsset(pth), real_size=os.path.getsize(pth))
        return SparseAsset(slot, h)

    def exists(s, name):
        return os.path.exists(s._p(name))

    def put_file(s, name, path):
        shutil.copyfile(path, s._p(name))

    def put(s, name, data):
        with open(s._p(name), "wb") as f:
            f.write(data)

    def delete(s, name):
        try:
            os.remove(s._p(name))
        except FileNotFoundError:
            pass

    def list(s, pat=r".*"):
        return sorted(n for n in os.listdir(s.root) if re.fullmatch(pat, n))


class LocalJournal:
    def __init__(s, root):
        s.root = root
        os.makedirs(root, exist_ok=True)

    def _p(s, n):
        return os.path.join(s.root, n)

    def list(s, pat=ram16.DELTA):
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


def _ram_geom():
    return {"ns": BINS, "page": PAGE, "pps": PPS, "hdr": BIN_HDR, "machine": "00" * 16}


def slots_for(bins_dir, geom):
    """Sparse slot list: real LocalAsset for present bins, forged-header zero bins otherwise."""
    out = []
    for i in range(geom["ns"]):
        name = f"mem-{i:02d}.bin"
        p = os.path.join(bins_dir, name)
        if os.path.exists(p):
            out.append(SparseAsset(i, forge_header(i), real=LocalAsset(p), real_size=os.path.getsize(p)))
        else:
            out.append(SparseAsset(i, forge_header(i)))
    return out


def make_ram(tier_dir, journal_dir=None, writer=False, cache_pages=64):
    geom = _ram_geom()
    bins = LocalBins(tier_dir)
    journal = LocalJournal(journal_dir) if journal_dir else None
    ram = ram16.Ram16(None, journal=journal, store=bins, nslots=BINS, writer=writer, cache_pages=cache_pages)
    ram.tier_store = bins
    ram.tier_geom = geom
    ram.tier_names = [f"mem-{i:02d}.bin" for i in range(BINS)]
    ram.tier_gen = 0
    return ram


def tile_model(src_gguf, tier_dir, weight_pages, pps=PPS, page=PAGE):
    """Tile a local GGUF into fixture bins exactly like import_model does (page 1+).
    Bin 0 keeps its header and payload page 0 (superblock); every bin gets a valid header."""
    os.makedirs(tier_dir, exist_ok=True)
    b0 = os.path.join(tier_dir, "mem-00.bin")
    if not os.path.exists(b0):
        forge_bin0(b0, "vram", {"weights": [1, BINS * pps - 1]})
    size = os.path.getsize(src_gguf)
    wp = -(-size // page)
    with open(src_gguf, "rb") as f:
        for b in range(BINS):
            lo, hi = max(1, b * pps), min(wp + 1, (b + 1) * pps)
            if lo >= hi:
                break
            path = os.path.join(tier_dir, f"mem-{b:02d}.bin")
            bin_size = BIN_HDR + (hi - b * pps) * page
            exists = os.path.exists(path)
            with open(path, "r+b" if exists else "wb") as out:
                if not exists:
                    out.truncate(bin_size)
                out.seek(0)
                out.write(forge_header(b, page=page, pps=pps, hdr=BIN_HDR, ns=BINS))
                out.seek(BIN_HDR + (lo - b * pps) * page)
                f.seek((lo - 1) * page)
                left = (hi - lo) * page
                while left > 0:
                    chunk = f.read(min(8 << 20, left))
                    if not chunk:
                        chunk = b"\0" * min(8 << 20, left)
                    out.write(chunk)
                    left -= len(chunk)
                out.truncate(bin_size)
    return wp


# ---------------- tiny Qwen3-MoE GGUF ----------------

TINY = dict(L=2, d=64, nh=8, nkv=2, hd=8, nff=32, ne=4, nu=2, vocab=128, eps=1e-6, rope=1e4)


TINY_DENSE = dict(L=2, d=64, nh=8, nkv=2, hd=8, nff=128, vocab=128, eps=1e-6, rope=1e4)


def make_tiny_dense_gguf(path, seed=17, gtype=0):
    """Random-weight dense qwen3 GGUF (no experts) for the fast-tier engine path."""
    from deepcore.gguf import write_gguf
    rng = np.random.default_rng(seed)
    c = TINY_DENSE
    meta = {
        "general.architecture": "qwen3", "general.name": "tiny-dense",
        "qwen3.block_count": c["L"], "qwen3.embedding_length": c["d"],
        "qwen3.attention.head_count": c["nh"], "qwen3.attention.head_count_kv": c["nkv"],
        "qwen3.attention.key_length": c["hd"], "qwen3.attention.layer_norm_rms_epsilon": c["eps"],
        "qwen3.feed_forward_length": c["nff"], "qwen3.rope.freq_base": c["rope"],
        "qwen3.context_length": 512,
        "tokenizer.ggml.tokens": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [f"t{i}" for i in range(c["vocab"] - 3)],
        "tokenizer.ggml.token_type": [3, 3, 3] + [1] * (c["vocab"] - 3),
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.tokens_count": c["vocab"],
    }
    tensors = [("token_embd.weight", [c["d"], c["vocab"]], gtype, rng.standard_normal((c["vocab"], c["d"])).astype(np.float32) * 0.1),
               ("output_norm.weight", [c["d"]], 0, np.ones(c["d"], np.float32))]
    for l in range(c["L"]):
        p = f"blk.{l}."
        tensors += [
            (p + "attn_norm.weight", [c["d"]], 0, np.ones(c["d"], np.float32) * 0.9),
            (p + "attn_q.weight", [c["d"], c["nh"] * c["hd"]], gtype, rng.standard_normal((c["nh"] * c["hd"], c["d"])).astype(np.float32) * 0.08),
            (p + "attn_k.weight", [c["d"], c["nkv"] * c["hd"]], gtype, rng.standard_normal((c["nkv"] * c["hd"], c["d"])).astype(np.float32) * 0.08),
            (p + "attn_v.weight", [c["d"], c["nkv"] * c["hd"]], gtype, rng.standard_normal((c["nkv"] * c["hd"], c["d"])).astype(np.float32) * 0.08),
            (p + "attn_output.weight", [c["nh"] * c["hd"], c["d"]], gtype, rng.standard_normal((c["d"], c["nh"] * c["hd"])).astype(np.float32) * 0.08),
            (p + "attn_q_norm.weight", [c["hd"]], 0, np.ones(c["hd"], np.float32)),
            (p + "attn_k_norm.weight", [c["hd"]], 0, np.ones(c["hd"], np.float32)),
            (p + "ffn_norm.weight", [c["d"]], 0, np.ones(c["d"], np.float32) * 0.9),
            (p + "ffn_gate.weight", [c["d"], c["nff"]], gtype, rng.standard_normal((c["nff"], c["d"])).astype(np.float32) * 0.08),
            (p + "ffn_up.weight", [c["d"], c["nff"]], gtype, rng.standard_normal((c["nff"], c["d"])).astype(np.float32) * 0.08),
            (p + "ffn_down.weight", [c["nff"], c["d"]], gtype, rng.standard_normal((c["d"], c["nff"])).astype(np.float32) * 0.08),
        ]
    tensors.append(("output.weight", [c["d"], c["vocab"]], gtype, rng.standard_normal((c["vocab"], c["d"])).astype(np.float32) * 0.2))
    return write_gguf(path, meta, tensors)


def make_tiny_gguf(path, seed=7, gtype=0):
    """Random-weight qwen3moe GGUF (F32 by default) with tokenizer metadata."""
    from deepcore.gguf import write_gguf
    rng = np.random.default_rng(seed)
    c = TINY
    meta = {
        "general.architecture": "qwen3moe", "general.name": "tiny-test",
        "qwen3moe.block_count": c["L"], "qwen3moe.embedding_length": c["d"],
        "qwen3moe.attention.head_count": c["nh"], "qwen3moe.attention.head_count_kv": c["nkv"],
        "qwen3moe.attention.key_length": c["hd"], "qwen3moe.attention.layer_norm_rms_epsilon": c["eps"],
        "qwen3moe.expert_count": c["ne"], "qwen3moe.expert_used_count": c["nu"],
        "qwen3moe.expert_feed_forward_length": c["nff"], "qwen3moe.rope.freq_base": c["rope"],
        "qwen3moe.context_length": 512,
        "tokenizer.ggml.tokens": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [f"t{i}" for i in range(c["vocab"] - 3)],
        "tokenizer.ggml.token_type": [3, 3, 3] + [1] * (c["vocab"] - 3),
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.tokens_count": c["vocab"],
    }
    eye = np.eye(c["d"], dtype=np.float32) * 0.5 + rng.standard_normal((c["d"], c["d"])).astype(np.float32) * 0.05
    tensors = [("token_embd.weight", [c["d"], c["vocab"]], gtype, rng.standard_normal((c["vocab"], c["d"])).astype(np.float32) * 0.1),
               ("output_norm.weight", [c["d"]], 0, np.ones(c["d"], np.float32))]
    for l in range(c["L"]):
        p = f"blk.{l}."
        tensors += [
            (p + "attn_norm.weight", [c["d"]], 0, np.ones(c["d"], np.float32) * 0.9),
            (p + "attn_q.weight", [c["d"], c["nh"] * c["hd"]], gtype, rng.standard_normal((c["nh"] * c["hd"], c["d"])).astype(np.float32) * 0.08),
            (p + "attn_k.weight", [c["d"], c["nkv"] * c["hd"]], gtype, rng.standard_normal((c["nkv"] * c["hd"], c["d"])).astype(np.float32) * 0.08),
            (p + "attn_v.weight", [c["d"], c["nkv"] * c["hd"]], gtype, rng.standard_normal((c["nkv"] * c["hd"], c["d"])).astype(np.float32) * 0.08),
            (p + "attn_output.weight", [c["nh"] * c["hd"], c["d"]], gtype, rng.standard_normal((c["d"], c["nh"] * c["hd"])).astype(np.float32) * 0.08),
            (p + "attn_q_norm.weight", [c["hd"]], 0, np.ones(c["hd"], np.float32)),
            (p + "attn_k_norm.weight", [c["hd"]], 0, np.ones(c["hd"], np.float32)),
            (p + "ffn_norm.weight", [c["d"]], 0, np.ones(c["d"], np.float32) * 0.9),
            (p + "ffn_gate_inp.weight", [c["d"], c["ne"]], gtype, rng.standard_normal((c["ne"], c["d"])).astype(np.float32) * 0.1),
            (p + "ffn_gate_exps.weight", [c["d"], c["nff"], c["ne"]], gtype, rng.standard_normal((c["ne"], c["nff"], c["d"])).astype(np.float32) * 0.1),
            (p + "ffn_up_exps.weight", [c["d"], c["nff"], c["ne"]], gtype, rng.standard_normal((c["ne"], c["nff"], c["d"])).astype(np.float32) * 0.1),
            (p + "ffn_down_exps.weight", [c["nff"], c["d"], c["ne"]], gtype, rng.standard_normal((c["ne"], c["d"], c["nff"])).astype(np.float32) * 0.1),
        ]
    tensors.append(("output.weight", [c["d"], c["vocab"]], gtype, rng.standard_normal((c["vocab"], c["d"])).astype(np.float32) * 0.2))
    return write_gguf(path, meta, tensors)


def dense_from_gguf(path):
    """All tensors dequantized to one dict (reference run)."""
    from deepcore import gguf as G
    from deepcore.gguf import parse_header
    f = open(path, "rb").read()
    meta, tensors, data_off, _ = parse_header(lambda off, n: f[off:off + n], len(f))
    w = {}
    for name, t in tensors.items():
        w[name] = G.dequant(f[t.addr:t.addr + t.nbytes], t.type).reshape([int(x) for x in reversed(t.dims)])
    return meta, w