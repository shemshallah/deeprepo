"""Batch mode: MemKV correctness and issue-turn parsing (no network)."""
import shutil
import sys
import tempfile
import os

import numpy as np

from deepcore.batch import CUR, MARK, MemKV, messages_from

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import TINY, dense_from_gguf, make_ram, make_tiny_gguf
from test_engine import dense_forward, setup
from deepcore.engine import Qwen3MoE
from deepcore import PAGE


def test_memkv_rw():
    kv = MemKV()
    payload = bytes(range(256)) * 40000   # 10 MiB, crosses pages
    kv.write(12345, payload)
    assert kv.read(12345, len(payload)) == payload
    assert kv.read(0, 100) == b"\0" * 100
    print("  MemKV sparse read/write across pages")


def test_batch_engine_matches_dense():
    tmp = tempfile.mkdtemp()
    try:
        gguf, size, vram, ram, kvlay, wp = setup(tmp)
        meta, w = dense_from_gguf(gguf)
        eng = Qwen3MoE(vram, 1, size, ramkv=MemKV(),
                       kv_layout={"base": 0, "bytes": 4 << 30}, f16_cache_mb=4, strict=True)
        ids = [7, 12, 40, 3]
        got = eng.forward(ids, 0)
        ref = dense_forward(w, dict(TINY), ids, 0)
        assert np.allclose(got, ref, atol=1e-4, rtol=1e-5), np.abs(got - ref).max()
        print("  batch engine with local MemKV matches dense")
    finally:
        shutil.rmtree(tmp)


def test_issue_turn_parsing():
    user = [{"id": 1, "body": "hello"}]
    assert messages_from("T", "", user)["handle"] is None
    # unfinished bot -> resume handle, last role user
    req = messages_from("T", "", user + [{"id": 2, "body": MARK + "partial" + CUR}])
    assert req["handle"] == 2 and len(req["messages"]) == 2
    # completed bot -> nothing pending
    assert messages_from("T", "", user + [{"id": 3, "body": MARK + "done"}]) is None
    # assistant then user -> pending
    req = messages_from("T", "", user + [{"id": 3, "body": MARK + "done"},
                                         {"id": 4, "body": "next"}])
    assert req["handle"] is None and req["messages"][-1]["role"] == "user"
    print("  issue turn parsing (pending/unfinished/completed)")


if __name__ == "__main__":
    test_memkv_rw()
    test_batch_engine_matches_dense()
    test_issue_turn_parsing()
    print("batch tests passed")
