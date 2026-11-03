"""RRBIN v1 segment format. Self-describing: header carries per-1MiB sha256 so a ranged read verifies alone.

header (4096 B): <8sIIIIQIIQQ32s> magic,ver,space,bin,seg,voff,pclass,codec,stored_len,gen,payload_sha
                 @128 : 64 x 32B sub-block sha256 (of the LOGICAL payload)
                 @4064: sha256(header[:4064])
codec 0 raw | 1 zlib | 2 zero (stored_len 0, no payload bytes)
"""
import hashlib, struct, zlib
from . import SEG_PAYLOAD, SUB, NSUB, HDR

MAGIC = b"RRSEG1\0\0"
FMT = "<8sIIIIQIIQQ32s"
RAW, ZLIB, ZERO = 0, 1, 2
PIN, STREAM, CACHE = 0, 1, 2
SPACES = {"ram": 0, "vram": 1}
_ZERO_SUB = hashlib.sha256(bytes(SUB)).digest()


class RRError(Exception): pass


def sha(b): return hashlib.sha256(b).digest()


def pack_segment(payload, space="ram", bin_id=0, seg=0, voff=0, pclass=STREAM, gen=1, codec=None):
    """payload: bytes <= SEG_PAYLOAD (zero-padded to full segment logically). Returns (header, stored_bytes)."""
    if len(payload) > SEG_PAYLOAD: raise RRError("payload too large")
    full = bytes(payload) + bytes(SEG_PAYLOAD - len(payload))
    subs = [sha(full[i * SUB:(i + 1) * SUB]) for i in range(NSUB)]
    allzero = all(s == _ZERO_SUB for s in subs)
    if codec is None: codec = ZERO if allzero else RAW
    if codec == ZERO:
        if not allzero: raise RRError("ZERO codec on nonzero data")
        stored = b""
    elif codec == ZLIB: stored = zlib.compress(full, 3)
    elif codec == RAW: stored = full
    else: raise RRError("bad codec")
    h = bytearray(HDR)
    struct.pack_into(FMT, h, 0, MAGIC, 1, SPACES[space], bin_id, seg, voff, pclass, codec, len(stored), gen, sha(full))
    for i, s in enumerate(subs): h[128 + 32 * i:160 + 32 * i] = s
    h[HDR - 32:] = sha(bytes(h[:HDR - 32]))
    return bytes(h), stored


def parse_header(h):
    if len(h) != HDR: raise RRError("short header")
    if sha(bytes(h[:HDR - 32])) != bytes(h[HDR - 32:]): raise RRError("header checksum")
    m, ver, sp, b, sg, voff, pc, cod, sl, gen, psha = struct.unpack_from(FMT, h, 0)
    if m != MAGIC or ver != 1: raise RRError("bad magic/version")
    subs = [bytes(h[128 + 32 * i:160 + 32 * i]) for i in range(NSUB)]
    return dict(space=sp, bin=b, seg=sg, voff=voff, pclass=pc, codec=cod, stored_len=sl, gen=gen, sha=psha, subs=subs)


def unpack_segment(h, stored):
    """Full decode + verify. Returns the SEG_PAYLOAD logical bytes."""
    p = parse_header(h)
    if len(stored) != p["stored_len"]: raise RRError("stored length mismatch")
    if p["codec"] == RAW: full = stored
    elif p["codec"] == ZLIB: full = zlib.decompress(stored)
    elif p["codec"] == ZERO: full = bytes(SEG_PAYLOAD)
    else: raise RRError("bad codec")
    if len(full) != SEG_PAYLOAD: raise RRError("decoded length")
    for i in range(NSUB):
        if sha(full[i * SUB:(i + 1) * SUB]) != p["subs"][i]: raise RRError(f"sub-block {i} hash")
    if sha(full) != p["sha"]: raise RRError("payload hash")
    return full


def verify_sub(header_parsed, idx, data):
    """Verify one 1 MiB sub-block read by range (raw codec only)."""
    if sha(data) != header_parsed["subs"][idx]: raise RRError(f"sub-block {idx} hash")
    return True
