"""Tier geometry, sparse-bin semantics, superblocks. stdlib only.

Bin layout on the wire (one GitHub release asset per bin):
  [ 4 MiB header region ][ 510 x 4 MiB payload pages ]  = 2044 MiB < 2 GiB per asset
Bin 0 page 0 of each tier carries the tier superblock (DPRSB1 JSON). Bins are
SPARSE: an absent bin reads as zeros; only bins touched by data ever exist.
"""
import json, struct, time, zlib

from . import BIN_HDR, BINS, PAGE, PPS

MAGIC = b"RAM16\0\0\1"          # ram16 slot header magic (compat)
HF = "<8s5I16sQ"                 # magic, slot, ns, page, pps, hdr, machine, crc64
MACHINE = b"DEEPREPO-SPARSE-V1"  # 16-byte machine id binds tiers together

def tier_pages(bins=BINS, pps=PPS):
    return bins * pps

def tier_bytes(bins=BINS, pps=PPS):
    return tier_pages(bins, pps) * PAGE

def bin_asset_size(pps=PPS):
    return BIN_HDR + pps * PAGE

def bin_of(page, pps=PPS):
    return page // pps, page % pps

# ---- superblock (page 0, tier-local) -------------------------------------

def pack_super(tier, layout, meta=None):
    j = json.dumps({"v": 1, "tier": tier, "page": PAGE, "pps": PPS, "bins": BINS,
                    "machine": MACHINE.hex(), "layout": layout, "meta": meta or {},
                    "made": int(time.time())}, separators=(",", ":")).encode()
    return b"DPRSB1" + struct.pack("<I", len(j)) + j

def read_super(ram):
    """Read a deepcore superblock from page 0 of a connected tier (ram16-compatible fallback too)."""
    h = ram.read(0, 12)
    if h[:6] == b"DPRSB1":
        return json.loads(ram.read(6 + 4, struct.unpack("<I", h[6:10])[0]))
    raise ValueError("tier has no DPRSB1 superblock (run bootstrap)")

# ---- reporam layout -------------------------------------------------------
# regions given in [start_page, n_pages]; page 0 is the superblock.

def plan_ram_layout(total_pages, kv_pages=1024, reg_pages=256):
    assert 1 + kv_pages + reg_pages <= total_pages - 1, "kv+reg do not fit tier"
    return {"kv": [1, kv_pages], "regs": [1 + kv_pages, reg_pages],
            "scratch": [1 + kv_pages + reg_pages, total_pages - 1 - kv_pages - reg_pages]}

def plan_vram_layout(total_pages):
    return {"weights": [1, total_pages - 1]}

# ---- forged headers for absent bins (sparse semantics) --------------------

def _hf_crc(head52):
    return struct.unpack("<Q", head52[44:52])[0]

def forge_header(slot, page=PAGE, pps=PPS, hdr=BIN_HDR, ns=BINS, machine=MACHINE):
    """56 bytes ram16.Ram16 accepts as a valid slot header for an absent (all-zero) bin."""
    body = struct.pack(HF, MAGIC, slot, ns, page, pps, hdr, machine, 0)
    crc = zlib.crc32(body)
    return body + struct.pack("<I", crc)

def header_of(assets_read_fn):
    h = assets_read_fn(0, 56)
    mg, sl, ns, page, pps, hdr, mach, _ = struct.unpack(HF, h[:52])
    assert mg == MAGIC and zlib.crc32(h[:52]) == struct.unpack("<I", h[52:])[0], "bad bin-0 header"
    return {"ns": ns, "page": page, "pps": pps, "hdr": hdr, "machine": mach.hex()}
