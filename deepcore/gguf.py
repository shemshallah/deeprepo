"""GGUF: paged header parse, numpy dequant (byte-exact vs ggml), row-gather, test-model writer.

Dequant is vectorized numpy following ggml/src/ggml-quants.c exactly:
  F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K. IQ quants are rejected with a clear error.
"""
import collections
import struct

import numpy as np
import ram16

GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
        24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 30: (1, 2)}
NAME = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}
_SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


# ============================ dequant =====================================

def dq_f16(b):
    return np.frombuffer(b, "<f2").astype(np.float32)


def dq_f32(b):
    return np.frombuffer(b, "<f4").astype(np.float32)


def dq_q8_0(b):
    nb = len(b) // 34
    a = np.frombuffer(b[:nb * 34], np.uint8).reshape(nb, 34)
    d = a[:, 0:2].copy().view("<f2").astype(np.float32)
    q = a[:, 2:34].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)


def _qk_scales(sc):
    """(nb, 12) scale bytes -> ((nb, 8) d-part, (nb, 8) m-part), per get_scale_min_k4."""
    d = np.empty((sc.shape[0], 8), np.uint8)
    m = np.empty_like(d)
    d[:, :4] = sc[:, :4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    hi = sc[:, :4] >> 6
    d[:, 4:] = (sc[:, 8:12] & 0xF) | (hi << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)


def dq_q4_k(b):
    nb = len(b) // 144
    a = np.frombuffer(b[:nb * 144], np.uint8).reshape(nb, 144)
    d = a[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1)
    dmin = a[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1)
    sdp, mnp = _qk_scales(a[:, 4:16])
    qs = a[:, 16:144].reshape(nb, 4, 32)
    lo = (qs & 0xF).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)                       # (nb, 4, 32)
    sd = sdp.reshape(nb, 4, 2); mn = mnp.reshape(nb, 4, 2)
    d1 = d[:, None, None] * sd[:, :, 0:1]
    m1 = dmin[:, None, None] * mn[:, :, 0:1]
    d2 = d[:, None, None] * sd[:, :, 1:2]
    m2 = dmin[:, None, None] * mn[:, :, 1:2]
    out = np.empty((nb, 4, 2, 32), np.float32)
    out[:, :, 0, :] = d1 * lo - m1
    out[:, :, 1, :] = d2 * hi - m2
    return out.reshape(nb * 256)


def dq_q5_k(b):
    nb = len(b) // 176
    a = np.frombuffer(b[:nb * 176], np.uint8).reshape(nb, 176)
    d = a[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1)
    dmin = a[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1)
    sdp, mnp = _qk_scales(a[:, 4:16])
    qh = a[:, 16:48]                                        # (nb, 32) high bits
    qs = a[:, 48:176].reshape(nb, 4, 32)
    bits_lo = ((qh[:, None, :] >> np.array([0, 2, 4, 6], np.uint8)[None, :, None]) & 1).astype(np.float32) * 16.0
    bits_hi = ((qh[:, None, :] >> np.array([1, 3, 5, 7], np.uint8)[None, :, None]) & 1).astype(np.float32) * 16.0
    lo = (qs & 0xF).astype(np.float32) + bits_lo
    hi = (qs >> 4).astype(np.float32) + bits_hi
    sd = sdp.reshape(nb, 4, 2); mn = mnp.reshape(nb, 4, 2)
    out = np.empty((nb, 4, 2, 32), np.float32)
    out[:, :, 0, :] = d[:, None, None] * sd[:, :, 0:1] * lo - dmin[:, None, None] * mn[:, :, 0:1]
    out[:, :, 1, :] = d[:, None, None] * sd[:, :, 1:2] * hi - dmin[:, None, None] * mn[:, :, 1:2]
    return out.reshape(nb * 256)


def dq_q6_k(b):
    """block_q6_K: ql[128], qh[64], scales[16], d(fp16) LAST (ggml-common.h)."""
    nb = len(b) // 210
    a = np.frombuffer(b[:nb * 210], np.uint8).reshape(nb, 210)
    d = a[:, 208:210].copy().view("<f2").astype(np.float32).reshape(-1)  # (nb,)
    ql = a[:, 0:128].reshape(nb, 2, 2, 32)                  # [blk, c(128-chunk), half, l]
    qh = a[:, 128:192].reshape(nb, 2, 32)
    sc = a[:, 192:208].copy().view(np.int8).astype(np.float32).reshape(nb, 2, 8)
    isx = (np.arange(32) // 16)                             # per-l scale sub-index
    out = np.empty((nb, 2, 4, 32), np.float32)
    lo0 = (ql[:, :, 0, :] & 0xF).astype(np.float32)
    hi0 = (ql[:, :, 0, :] >> 4).astype(np.float32)
    lo1 = (ql[:, :, 1, :] & 0xF).astype(np.float32)
    hi1 = (ql[:, :, 1, :] >> 4).astype(np.float32)
    hb0 = (qh & np.uint8(0x03)).astype(np.float32)          # per c,l
    hb1 = ((qh >> np.uint8(2)) & np.uint8(0x03)).astype(np.float32)
    hb2 = ((qh >> np.uint8(4)) & np.uint8(0x03)).astype(np.float32)
    hb3 = ((qh >> np.uint8(6)) & np.uint8(0x03)).astype(np.float32)
    q1 = lo0 + hb0 * 16.0 - 32.0
    q2 = lo1 + hb1 * 16.0 - 32.0
    q3 = hi0 + hb2 * 16.0 - 32.0
    q4 = hi1 + hb3 * 16.0 - 32.0
    # scales: sc[:, c, is + {0,2,4,6}] with is = l//16
    s0 = np.take_along_axis(sc, (isx + 0)[None, None, :], axis=2)
    s2 = np.take_along_axis(sc, (isx + 2)[None, None, :], axis=2)
    s4 = np.take_along_axis(sc, (isx + 4)[None, None, :], axis=2)
    s6 = np.take_along_axis(sc, (isx + 6)[None, None, :], axis=2)
    out[:, :, 0, :] = d[:, None, None] * s0 * q1
    out[:, :, 1, :] = d[:, None, None] * s2 * q2
    out[:, :, 2, :] = d[:, None, None] * s4 * q3
    out[:, :, 3, :] = d[:, None, None] * s6 * q4
    return out.reshape(nb * 256)


_DEQ = {0: dq_f32, 1: dq_f16, 8: dq_q8_0, 12: dq_q4_k, 13: dq_q5_k, 14: dq_q6_k}


def dequant(b, gtype):
    if gtype == 30:  # BF16: sign/evalponent in high bytes
        u = np.frombuffer(b, "<u2").astype(np.uint32)
        return (u << np.uint32(16)).view(np.float32)
    f = _DEQ.get(gtype)
    if f is None:
        raise ValueError(f"unsupported ggml type {gtype} ({NAME.get(gtype, '?')}) — only F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K")
    return f(b)


def bits_per_weight(gtype):
    return {0: 32.0, 1: 16.0, 30: 16.0, 8: 8.5, 12: 4.5, 13: 5.5, 14: 6.5625}[gtype]


# ============================ header parse ================================

class T:
    __slots__ = "name dims type off nbytes addr".split()

    def __init__(s, **k):
        for a in s.__slots__:
            setattr(s, a, k.get(a))

    def __repr__(s):
        return f"T({s.name} {s.dims} {NAME.get(s.type, s.type)} @{s.addr} {s.nbytes}B)"


def _val_keep(c, t):
    """GGUF value reader that KEEPS string arrays (tokenizer vocab/merges)."""
    if t in _SC:
        return c.u(_SC[t])
    if t == 8:
        return c.str()
    if t == 9:
        et, n = c.u("<I"), c.u("<Q")
        if et == 8:
            return [c.str() for _ in range(n)]
        if et in _SC:
            c.take(struct.calcsize(_SC[et]) * n)
            return ("array", et, n)
        for _ in range(n):
            _val_keep(c, et)
        return ("array", et, n)
    raise ValueError(f"bad gguf value type {t}")


def parse_header(reader, limit):
    """reader(off, n) over the raw GGUF byte stream. Returns (meta, tensor_dict, data_off, header_end)."""
    c = ram16._Cur(reader, limit)
    assert c.take(4) == b"GGUF", "not a GGUF"
    version = c.u("<I")
    assert version >= 2, "GGUF v1 unsupported"
    nt, nk = c.u("<Q"), c.u("<Q")
    meta = {}
    for _ in range(nk):
        k = c.str()
        meta[k] = _val_keep(c, c.u("<I"))
    infos = []
    for _ in range(nt):
        nm = c.str(); nd = c.u("<I")
        dims = [c.u("<Q") for _ in range(nd)]
        ty = c.u("<I"); off = c.u("<Q")
        infos.append((nm, dims, ty, off))
    al = int(meta.get("general.alignment", 32))
    header_end = c.pos
    data_off = (c.pos + al - 1) // al * al
    tensors = collections.OrderedDict()
    for j, (nm, dims, ty, off) in enumerate(sorted(infos, key=lambda i: i[3])):
        ne = 1
        for d_ in dims:
            ne *= d_
        if ty in GGML:
            nb = ne // GGML[ty][0] * GGML[ty][1]
        else:
            nxt = infos[j + 1][3] if j + 1 < len(infos) else limit - data_off
            nb = nxt - off
        tensors[nm] = T(name=nm, dims=dims, type=ty, off=off, nbytes=nb, addr=data_off + off)
    return meta, tensors, data_off, header_end


# ============================ writer (test models) ========================

def write_gguf(path, meta, tensors, align=32):
    """tensors: list of (name, dims_ggml_order, gtype, np_array_or_bytes_for_qtypes).
    Float arrays are converted per gtype (F32/F16/Q8_0 supported for writing)."""
    import numpy as _np
    kvs = []
    for k, v in meta.items():
        if isinstance(v, str):
            kvs.append((k, 8, v))
        elif isinstance(v, int):
            kvs.append((k, 10 if v > 2**31 - 1 else 4, v))
        elif isinstance(v, float):
            kvs.append((k, 6, v))
        elif isinstance(v, list) and (not v or isinstance(v[0], str)):
            kvs.append((k, 9, (8, v)))
        elif isinstance(v, list):
            et = 4 if v and isinstance(v[0], int) else 6
            kvs.append((k, 9, (et, v)))
    body = bytearray()
    infos = []
    for name, dims, gtype, arr in tensors:
        ne = 1
        for d_ in dims:
            ne *= d_
        if gtype == 0:
            data = _np.asarray(arr, np.float32).tobytes()
        elif gtype == 1:
            data = _np.asarray(arr, np.float32).astype("<f2").tobytes()
        elif gtype == 8:
            x = _np.asarray(arr, np.float32).reshape(-1)
            nb = len(x) // 32
            d = _np.abs(x.reshape(nb, 32)).max(axis=1) / 127.0
            d[d == 0] = 1e-30
            q = _np.clip(_np.rint(x.reshape(nb, 32) / d[:, None]), -127, 127).astype(np.int8)
            data = d.astype("<f2").tobytes() + q.tobytes()
        elif gtype == 12:
            raise NotImplementedError("q4_k write path unused; tests use f32/q8_0")
        else:
            raise NotImplementedError(gtype)
        off = len(body)
        infos.append((name, dims, gtype, off, data))
        body += data
        pad = (-len(body)) % align
        body += b"\0" * pad
    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    def _str(s):
        b = s.encode()
        hdr.extend(struct.pack("<Q", len(b))); hdr.extend(b)
    for k, t, v in kvs:
        _str(k); hdr.extend(struct.pack("<I", t))
        if t == 8: _str(v)
        elif t == 4: hdr.extend(struct.pack("<I", v))
        elif t == 10: hdr.extend(struct.pack("<Q", v))
        elif t == 6: hdr.extend(struct.pack("<f", v))
        elif t == 9:
            et, vv = v
            hdr.extend(struct.pack("<IQ", et, len(vv)))
            if et == 8:
                for s_ in vv: _str(s_)
            elif et == 4:
                for s_ in vv: hdr.extend(struct.pack("<I", s_))
            else:
                for s_ in vv: hdr.extend(struct.pack("<f", s_))
    for name, dims, gtype, off, data in infos:
        _str(name)
        hdr.extend(struct.pack("<I", len(dims)))
        for d_ in dims: hdr.extend(struct.pack("<Q", d_))
        hdr.extend(struct.pack("<IQ", gtype, off))
    hdr += b"\0" * ((-len(hdr)) % align)   # data section starts aligned
    with open(path, "wb") as f:
        f.write(hdr); f.write(bytes(body))
        pad = (-f.tell()) % align
        f.write(b"\0" * pad)
    return len(hdr) + len(body)


# ============================ paged tensor access =========================

class GGUFOnRam:
    """A GGUF whose bytes live inside a tier register at weights base page. Header parsed
    once through the paged reader; tensors accessed by byte range (dequant on demand)."""

    def __init__(s, ram, wbase_page, size, expected_sha=None):
        s.ram, s.wbase, s.size = ram, wbase_page * ram.page, size
        s.meta, s.tensors, s.data_off, _ = parse_header(lambda off, n: ram.read(s.wbase + off, min(n, max(0, size - off))), size)
        # tensor addrs from the parser are relative to the GGUF stream; rebase onto the tier
        for t in s.tensors.values():
            t.addr += s.wbase
        s.arch = s.meta.get("general.architecture", "")

    def raw(s, name):
        return s.tensors[name]

    def read_tensor(s, name):
        t = s.tensors[name]
        return s.ram.read(t.addr, t.nbytes)

    def read_range(s, name, off, n):
        t = s.tensors[name]
        off = max(0, min(off, t.nbytes)); n = min(n, t.nbytes - off)
        return s.ram.read(t.addr + off, n)

    def prefetch_range(s, name, off, n):
        t = s.tensors[name]
        s.ram.prefetch(t.addr + max(0, off), min(n, max(0, t.nbytes - off)))

    def dequant_tensor(s, name):
        return dequant(s.read_tensor(name), s.tensors[name].type)

    def gather_rows(s, name, rows, dtype=np.float32):
        """Row gather for row-major tensors [rows, cols] (ne[0]=cols). Reads only needed blocks."""
        t = s.tensors[name]
        if t.type not in (0, 1, 30, 8, 12, 13, 14):
            raise ValueError(t.type)
        cols = t.dims[0]
        blk_e, blk_b = GGML[t.type]
        assert cols % blk_e == 0
        brow = cols // blk_e * blk_b
        rows_sorted = np.unique(np.asarray(rows, np.int64))
        blobs = []
        for r in rows_sorted:
            blobs.append(s.ram.read(t.addr + int(r) * brow, brow))
        out = dequant(b"".join(blobs), t.type).reshape(len(rows_sorted), -1) if len(blobs) else np.empty((0, cols), np.float32)
        idx = {int(r): i for i, r in enumerate(rows_sorted)}
        res = np.empty((len(rows), cols), dtype)
        for i, r in enumerate(rows):
            res[i] = out[idx[int(r)]]
        return res

    def layers(s):
        """{-1: global, 0..L-1: per-layer} tensor name lists."""
        import re as _re
        out = collections.defaultdict(list)
        for t in s.tensors.values():
            m = _re.match(r"blk\.(\d+)\.", t.name)
            out[int(m.group(1)) if m else -1].append(t.name)
        return dict(sorted(out.items()))
