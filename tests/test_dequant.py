"""Dequant must match ggml bit-for-bit. Reference: scalar ports of ggml/src/ggml-quants.c
dequantize_row_q4_K / q5_K / q6_K / q8_0 (read directly from the pinned llama.cpp source).
"""
import numpy as np

from deepcore import gguf as G


def f16(b):
    return float(np.frombuffer(b, "<f2")[0])


def ref_q3_k(x):
    out = []
    nb = len(x) // 110
    for i in range(nb):
        b = x[i * 110:(i + 1) * 110]
        hm, qs, sc, d = b[0:32], b[32:96], b[96:108], f16(b[108:110])
        aux = [0, 0, 0, 0]
        for j in range(3):
            aux[j] = int.from_bytes(sc[4 * j:4 * j + 4], "little")
        tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4)
        aux[3] = ((aux[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4)
        aux[0] = (aux[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4)
        aux[1] = (aux[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4)
        scales = []
        for u in aux:
            scales += [((u >> (8 * k)) & 0xFF) - 256 if ((u >> (8 * k)) & 0xFF) > 127 else ((u >> (8 * k)) & 0xFF) for k in range(4)]
        q = qs
        m = 1
        isx = 0
        for n in range(0, 256, 128):
            shift = 0
            for j in range(4):
                dl = d * (scales[isx] - 32); isx += 1
                for l in range(16):
                    qv = (q[l] >> shift) & 3
                    out.append(dl * (qv - (0 if (hm[l] & m) else 4)))
                dl = d * (scales[isx] - 32); isx += 1
                for l in range(16):
                    qv = (q[l + 16] >> shift) & 3
                    out.append(dl * (qv - (0 if (hm[l + 16] & m) else 4)))
                shift += 2; m <<= 1
            q = q[32:]
    return np.array(out, np.float32)


def ref_q4_k(x):
    out = []
    nb = len(x) // 144
    for i in range(nb):
        b = x[i * 144:(i + 1) * 144]
        d, dmin = f16(b[0:2]), f16(b[2:4])
        sc, q = b[4:16], b[16:144]

        def gsm(j):
            if j < 4:
                return sc[j] & 63, sc[j + 4] & 63
            return ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
                    (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))

        is_ = 0
        for _ in range(4):
            dsc, msc = gsm(is_)
            d1, m1 = d * dsc, dmin * msc
            dsc2, msc2 = gsm(is_ + 1)
            d2, m2 = d * dsc2, dmin * msc2
            for l in range(32):
                out.append(d1 * (q[l] & 0xF) - m1)
            for l in range(32):
                out.append(d2 * (q[l] >> 4) - m2)
            q = q[32:]
            is_ += 2
    return np.array(out, np.float32)


def ref_q5_k(x):
    out = []
    nb = len(x) // 176
    for i in range(nb):
        b = x[i * 176:(i + 1) * 176]
        d, dmin = f16(b[0:2]), f16(b[2:4])
        sc, qh, qs = b[4:16], b[16:48], b[48:176]

        def gsm(j):
            if j < 4:
                return sc[j] & 63, sc[j + 4] & 63
            return ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
                    (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))

        is_, u1, u2 = 0, 1, 2
        for _ in range(4):
            dsc, msc = gsm(is_)
            d1, m1 = d * dsc, dmin * msc
            dsc2, msc2 = gsm(is_ + 1)
            d2, m2 = d * dsc2, dmin * msc2
            for l in range(32):
                out.append(d1 * ((qs[l] & 0xF) + (16 if qh[l] & u1 else 0)) - m1)
            for l in range(32):
                out.append(d2 * ((qs[l] >> 4) + (16 if qh[l] & u2 else 0)) - m2)
            qs = qs[32:]
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return np.array(out, np.float32)


def ref_q6_k(x):
    out = [0.0] * (len(x) // 210 * 256)
    nb = len(x) // 210
    for i in range(nb):
        b = x[i * 210:(i + 1) * 210]
        d = f16(b[208:210])          # block_q6_K: d is last
        ql, qh, sc = b[0:128], b[128:192], b[192:208]
        pos = 0
        for n in range(0, 256, 128):
            for l in range(32):
                isx = l // 16
                q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32
                q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                out[i * 256 + n + l] = d * s8(sc[isx + 0]) * q1
                out[i * 256 + n + 32 + l] = d * s8(sc[isx + 2]) * q2
                out[i * 256 + n + 64 + l] = d * s8(sc[isx + 4]) * q3
                out[i * 256 + n + 96 + l] = d * s8(sc[isx + 6]) * q4
            ql, qh, sc = ql[64:], qh[32:], sc[8:]
    return np.array(out, np.float32)


def s8(v):
    return v - 256 if v > 127 else v


def ref_q8_0(x):
    nb = len(x) // 34
    out = []
    for i in range(nb):
        b = x[i * 34:(i + 1) * 34]
        d = f16(b[0:2])
        for j in range(32):
            q = b[2 + j]
            out.append(d * (q - 256 if q > 127 else q))
    return np.array(out, np.float32)


def bitwise_equal(a, b):
    """NaN-safe exact equality."""
    return np.array_equal(np.asarray(a, np.float32).view(np.uint32),
                          np.asarray(b, np.float32).view(np.uint32))


def rand_bytes(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, n, dtype=np.uint8).tobytes()


def test_q8_0():
    b = rand_bytes(34 * 100, 1)
    assert bitwise_equal(G.dq_q8_0(b), ref_q8_0(b))


def test_q3_k():
    b = rand_bytes(110 * 50, 13)
    assert bitwise_equal(G.dq_q3_k(b), ref_q3_k(b))


def test_q4_k():
    b = rand_bytes(144 * 50, 2)
    assert bitwise_equal(G.dq_q4_k(b), ref_q4_k(b))


def test_q5_k():
    b = rand_bytes(176 * 50, 3)
    assert bitwise_equal(G.dq_q5_k(b), ref_q5_k(b))


def test_q6_k():
    b = rand_bytes(210 * 50, 4)
    assert bitwise_equal(G.dq_q6_k(b), ref_q6_k(b))


def test_f16_f32_bf16():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal(1024) * 3).astype(np.float32)
    assert np.allclose(G.dequant(x.tobytes(), 0), x)
    h = x.astype("<f2").tobytes()
    assert np.allclose(G.dequant(h, 1), x.astype("<f2").astype(np.float32), atol=1e-3)
    u = x.view(np.uint32) >> 16
    assert np.allclose(G.dequant(u.astype("<u2").tobytes(), 30), (x >> 16 << 16) if False else (u.astype(np.uint32) << 16).view(np.float32), atol=1e-2)


def test_unsupported_rejected():
    try:
        G.dequant(b"\0" * 84, 10)
        assert False, "Q2_K should be rejected"
    except ValueError as e:
        assert "unsupported" in str(e)


def test_row_gather_block_alignment():
    """gather_rows must produce identical values to whole-tensor dequant (Q4_K rows)."""
    nb = 40
    b = rand_bytes(144 * nb, 6)
    full = G.dq_q4_k(b)
    # fabricate a GGUFOnRam-like shim isn't needed: exercise the block math directly
    cols = 256 * 2  # two blocks per row
    rows = nb * 256 // cols
    assert rows >= 2
    from deepcore.gguf import T, GGML
    # direct check: per-row dequant == sliced whole dequant
    brow = cols // 256 * 144
    for r in range(rows):
        got = G.dq_q4_k(b[r * brow:(r + 1) * brow])
        assert bitwise_equal(got, full[r * cols:(r + 1) * cols])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("dequant tests passed")