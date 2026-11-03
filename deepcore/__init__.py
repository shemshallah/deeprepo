"""deepcore: repo-backed sparse RAM/VRAM for a Qwen3-MoE that cannot fit in any runner.

Tiers:
  L0 physram  runner-local bounded caches (page LRU + optional dequant f16 cache) -- strictly capped
  L1 reporam  32 GB-class sparse virtual RAM over GitHub release bins + page-delta journal (KV, registers, ring state)
  L2 repovram 32 GB-class sparse virtual VRAM over GitHub release bins (GGUF weights, read-only)
"""

__version__ = "1.0.0"

PAGE = 4 << 20          # 4 MiB logical page
BIN_HDR = 4 << 20       # 4 MiB bin header region
PPS = 510               # pages per bin -> bin = 4 MiB + 2040 MiB = 2044 MiB < 2 GiB asset cap
BINS = 16               # bins per tier
# tier virtual size = 16 * 510 * 4 MiB = 32,640 MiB ~= 31.9 GiB ~= "32 GB sparse" per tier

REPORAM = "reporam"
REPOVRAM = "repovram"
TAG_VRAM = "repovram"        # weights bins
TAG_VRAM_J = "repovram-j"    # import progress journal (small json assets)
TAG_RAM = "reporam"          # ram bins
TAG_RAM_J = "reporam-j"      # page-delta journal (the actual write path)

CHAT_LABEL = "deeprepo-chat"
