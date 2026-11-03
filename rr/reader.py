"""Verified ranged reader over manifest+store. Raw codec: 1 MiB sub-block ranged reads verified against header.
Overlay: overlay dict {(space,vseg): (repo,rel,name,off,len)} not needed here; manifest generations cover it."""
import hashlib, threading
from . import SEG_PAYLOAD, SUB, HDR
from .rrbin import parse_header, unpack_segment, RAW, ZLIB, ZERO, verify_sub, RRError


class Reader:
    def __init__(s, manifest, store, hdr_cache=256):
        s.m, s.store = manifest, store; s._h = {}; s.lock = threading.Lock(); s.bytes_fetched = 0

    def _loc(s, space, vseg):
        g = s.m.seg(space, vseg)
        if g is None: return None
        b = s.m.d["bins"][g["bin"]]; return g, b

    def header(s, space, vseg):
        k = (space, vseg)
        with s.lock:
            if k in s._h: return s._h[k]
        g, b = s._loc(space, vseg)
        raw = s.store.read(b["repo"], b["release"], g["bin"], g["off"], HDR)
        if hashlib.sha256(raw).hexdigest() != g["hdr"]: raise RRError("header hash != manifest pin")
        p = parse_header(raw); s.bytes_fetched += HDR
        with s.lock: s._h[k] = p
        return p

    def read_sub(s, space, vseg, idx):
        """Return one verified 1 MiB logical sub-block."""
        loc = s._loc(space, vseg)
        if loc is None: return bytes(SUB)
        g, b = loc; p = s.header(space, vseg)
        if p["codec"] == ZERO: return bytes(SUB)
        if p["codec"] == RAW:
            d = s.store.read(b["repo"], b["release"], g["bin"], g["off"] + HDR + idx * SUB, SUB)
            s.bytes_fetched += SUB; verify_sub(p, idx, d); return d
        full = s.store.read(b["repo"], b["release"], g["bin"], g["off"] + HDR, g["len"])
        s.bytes_fetched += len(full)
        full = unpack_segment(_raw_header(s, g, b), full)
        return full[idx * SUB:(idx + 1) * SUB]

    def read(s, space, voff, n):
        out = bytearray(); end = voff + n
        while voff < end:
            vseg, o = divmod(voff, SEG_PAYLOAD); idx, o2 = divmod(o, SUB)
            blk = s.read_sub(space, vseg, idx); take = min(SUB - o2, end - voff)
            out += blk[o2:o2 + take]; voff += take
        return bytes(out)


def _raw_header(s, g, b): return s.store.read(b["repo"], b["release"], g["bin"], g["off"], HDR)
