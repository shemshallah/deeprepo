#!/usr/bin/env python3
"""Read-only FUSE file over a repovram model, for llama.cpp mmap.

  sudo -E python tools/repovram_fuse.py /tmp/mnt --model qwen3-30b-a3b-instruct-2507-q4_k_m

The file is exactly the GGUF byte stream; reads are served by the same 4 MiB sparse
page reader used by the custom engine. No full local copy is made.
"""
import argparse
import errno
import os
import stat
import sys
import time

from fuse import FUSE, FuseOSError, Operations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepcore import PAGE  # noqa: E402
from deepcore.runtime import Env  # noqa: E402


class RepoGGUF(Operations):
    def __init__(self, env, base, size, name="model.gguf"):
        self.env, self.base, self.size, self.name = env, base, size, name
        self.t0 = time.time()
        self.reads = self.bytes = 0

    def getattr(self, path, fh=None):
        now = int(time.time())
        if path == "/":
            return {"st_mode": stat.S_IFDIR | 0o555, "st_nlink": 2,
                    "st_uid": 0, "st_gid": 0, "st_atime": now, "st_mtime": now, "st_ctime": now}
        if path == "/" + self.name:
            return {"st_mode": stat.S_IFREG | 0o444, "st_nlink": 1,
                    "st_size": self.size, "st_blksize": 1 << 20,
                    "st_uid": 0, "st_gid": 0, "st_atime": now, "st_mtime": now, "st_ctime": now}
        raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        return [".", "..", self.name]

    def open(self, path, flags):
        if path != "/" + self.name:
            raise FuseOSError(errno.ENOENT)
        if flags & (os.O_WRONLY | os.O_RDWR):
            raise FuseOSError(errno.EROFS)
        return 0

    def read(self, path, size, offset, fh):
        if path != "/" + self.name:
            raise FuseOSError(errno.ENOENT)
        if offset >= self.size:
            return b""
        n = min(size, self.size - offset)
        try:
            data = self.env.vram.read(self.base + offset, n)
        except Exception as e:
            print(f"fuse: read failed offset={offset} size={size}: {e!r}", flush=True)
            raise
        self.reads += 1
        if self.reads <= 5:
            print(f"fuse: read offset={offset} size={size} n={n} got={len(data)} head={data[:4].hex()}", flush=True)
        self.bytes += len(data)
        if self.reads % 256 == 0:
            print(f"fuse: {self.reads} reads, {self.bytes/2**20:.1f} MiB, "
                  f"{self.bytes/max(1e-6, time.time()-self.t0)/2**20:.2f} MiB/s", flush=True)
        return data

    def statfs(self, path):
        return {"f_bsize": 1 << 20, "f_frsize": 1 << 20,
                "f_blocks": (self.size + (1 << 20) - 1) // (1 << 20),
                "f_bfree": 0, "f_bavail": 0, "f_files": 1, "f_ffree": 0, "f_namemax": 255}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mountpoint")
    ap.add_argument("--model", default=None, help="model key (default: active vram/model)")
    ap.add_argument("--cache-pages", type=int, default=768)
    ap.add_argument("--name", default="model.gguf")
    a = ap.parse_args()
    os.makedirs(a.mountpoint, exist_ok=True)
    env = Env(writer=False, cache_pages=a.cache_pages, kv_cache_pages=16)
    mi = env.model_info(a.model) if a.model else env.model_info()
    if not mi or not mi.get("ready"):
        raise SystemExit(f"no ready model ({a.model or 'active'})")
    base = int(mi.get("base", 1)) * PAGE
    size = int(mi["size"])
    print(f"fuse: model={mi.get('key')} base_page={mi.get('base')} size={size} "
          f"cache_pages={a.cache_pages}", flush=True)
    FUSE(RepoGGUF(env, base, size, a.name), a.mountpoint, foreground=True,
         ro=True, max_read=1 << 20, nothreads=False)


if __name__ == "__main__":
    main()
