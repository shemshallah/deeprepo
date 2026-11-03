"""Runtime wiring shared by coordinator and CPU runners."""
import os

import ram16
from . import (REPORAM, REPOVRAM, TAG_RAM, TAG_RAM_J, TAG_VRAM, TAG_VRAM_J, PAGE)
from .geom import read_super
from .store import DiskCache, connect_tier, connect_local_tier
from .tiers import Registers
from .meter import Budget, ApiMeter


class Env:
    def __init__(s, token=None, writer=False, cache_pages=256, kv_cache_pages=256):
        s.token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
        owner = os.environ.get("GITHUB_REPOSITORY_OWNER") or (
            os.environ.get("GITHUB_REPOSITORY", "").split("/")[0] or None)
        def _full(name):
            if name and "/" in name or not owner:
                return name
            return f"{owner}/{name}"
        s.ram_repo = _full(os.environ.get("DEEPREPO_RAM", REPORAM))
        s.vram_repo = _full(os.environ.get("DEEPREPO_VRAM", REPOVRAM))
        s.main_repo = _full(os.environ.get("DEEPREPO_MAIN", "") or "")
        s.writer = writer
        s.local_dir = os.environ.get("DEEPREPO_TIER_DIR") or None
        s.ram_backend = os.environ.get("DEEPREPO_RAM_BACKEND", "github")
        s.vram_backend = os.environ.get("DEEPREPO_VRAM_BACKEND", "github")
        s.middle_url = os.environ.get("DEEPREPO_MIDDLE_URL", "")
        s.middle_token = os.environ.get("DEEPREPO_MIDDLE_TOKEN", "")
        cache_dir = os.environ.get("DEEPREPO_DISK_CACHE", "")
        disk_cache = None
        if cache_dir:
            cap = int(float(os.environ.get("DEEPREPO_DISK_CACHE_GB", "6")) * (1 << 30))
            disk_cache = DiskCache(cache_dir, cap_bytes=cap)
        s.disk_cache = disk_cache
        if s.local_dir:
            s.vram = connect_local_tier(os.path.join(s.local_dir, "vram"), cache_pages=cache_pages, disk_cache=disk_cache)
            s.ram = connect_local_tier(os.path.join(s.local_dir, "ram"),
                                       journal_dir=os.path.join(s.local_dir, "ram-j"),
                                       writer=writer, cache_pages=kv_cache_pages)
        else:
            s.vram = connect_tier(s.vram_repo, TAG_VRAM, s.token, cache_pages=cache_pages,
                                  disk_cache=disk_cache, maxrun=16, workers=8, backend=s.vram_backend,
                                  middle_url=s.middle_url, middle_token=s.middle_token)
            s.ram = connect_tier(s.ram_repo, TAG_RAM, s.token, journal_tag=TAG_RAM_J,
                                 writer=writer, cache_pages=kv_cache_pages,
                                 backend=s.ram_backend, middle_url=s.middle_url,
                                 middle_token=s.middle_token)
        ram_layout = read_super(s.ram)["layout"]
        s.regs = Registers(s.ram, ram_layout["regs"])
        # budget: adopt today's persisted ledger, then keep updating it
        s.budget = Budget(ledger=s.regs.get("budget/today") or {})
        s.api = ApiMeter()
        saved_caps = s.regs.get("budget/caps")
        if saved_caps:
            s.budget.caps.update(saved_caps)
        elif writer:
            s.regs.put("budget/caps", dict(s.budget.caps))

    def persist_budget(s):
        s.regs.put("budget/today", s.budget.snapshot())

    def safe_commit(s):
        """Only the lease-free writer may flush the shared journal."""
        lease = s.regs.get("ring/lease") or {}
        if lease.get("holder"):
            return False
        s.regs.commit()
        return True

    def model_info(s):
        return s.regs.get("vram/model")

    def engine(s, **opts):
        mi = s.model_info()
        if not mi or not mi.get("ready"):
            raise IOError("no model loaded in repovram (run load-model)")
        from .engine import Qwen3MoE
        kv_layout = {"base": read_super(s.ram)["layout"]["kv"][0] * PAGE,
                     "bytes": read_super(s.ram)["layout"]["kv"][1] * PAGE}
        return Qwen3MoE(s.vram, 1, int(mi["size"]), ramkv=s.ram if s.writer else None,
                        kv_layout=kv_layout, **opts)

    def tokenizer(s):
        from .tok import BPETokenizer
        from .gguf import GGUFOnRam
        mi = s.model_info()
        g = GGUFOnRam(s.vram, 1, int(mi["size"]))
        return BPETokenizer.from_meta(g.meta)