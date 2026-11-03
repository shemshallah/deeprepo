"""rr: repoRAM/repoVRAM bin format, manifest, stores, meter. stdlib only."""
SEG_PAYLOAD = 64 << 20          # 64 MiB segment payload
SUB = 1 << 20                   # 1 MiB verification sub-block
NSUB = SEG_PAYLOAD // SUB       # 64
HDR = 4096                      # segment header bytes
SEGS_PER_BIN = 31               # 31 * (64MiB+4KiB) = 2,080,501,760 < 2 GiB
