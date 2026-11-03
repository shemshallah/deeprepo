"""Middleman backend: server + client + Ram16 tier + compaction, all over HTTP on loopback.
Also verifies auth, Range writes, sparse absent bins, and manifest rehydration.
"""
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer

from deepcore import BIN_HDR, BINS, PAGE, PPS
from deepcore.geom import plan_ram_layout, read_super
from deepcore.middle import MiddleStore, Store, make_handler
from deepcore.store import connect_tier
from deepcore.tiers import compact_trimmed, forge_bin0


def start(root, token="tok"):
    store = Store(root)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, token, lambda *a: None))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_blob_api():
    tmp = tempfile.mkdtemp()
    srv, base = start(tmp)
    try:
        c = MiddleStore(base, "tok", "t/x")
        c.put("hello.bin", b"0123456789")
        assert c.get("hello.bin") == b"0123456789"
        assert c.size("hello.bin") == 10
        assert c.asset("hello.bin").read(3, 4) == b"3456"
        # ranged write extends the blob
        c._req("PUT", "/v1/blob/t/x/hello.bin?off=10", data=b"abc")
        assert c.get("hello.bin") == b"0123456789abc"
        names = c.list_names()
        assert "hello.bin" in names
        c.delete("hello.bin")
        assert c.list_names() == []
        # auth is enforced
        bad = MiddleStore(base, "wrong", "t/x")
        try:
            bad.put("nope.bin", b"x")
            assert False, "bad token should fail"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        print("  middle blob API: put/get/range/append/list/delete/auth ok")
    finally:
        srv.shutdown()
        shutil.rmtree(tmp)


def test_ram_tier_over_middle():
    tmp = tempfile.mkdtemp()
    srv, base = start(tmp)
    try:
        # forge a bin-0 and upload it to the middleman under test/ram
        p = os.path.join(tmp, "mem-00.bin")
        forge_bin0(p, "ram", plan_ram_layout(16 * 510, kv_pages=64, reg_pages=8))
        c = MiddleStore(base, "tok", "test/ram")
        c.put_file("mem-00.bin", p)
        ram = connect_tier("test", "ram", token="", journal_tag="ram-j", backend="middle",
                           middle_url=base, middle_token="tok", writer=True, cache_pages=16)
        lay = read_super(ram)["layout"]
        addr = lay["kv"][0] * PAGE + 12345
        payload = bytes(range(256)) * 32
        ram.write(addr, payload)
        ram.flush()
        reader = connect_tier("test", "ram", token="", journal_tag="ram-j", backend="middle",
                              middle_url=base, middle_token="tok", writer=False, cache_pages=8)
        assert reader.read(addr, len(payload)) == payload
        # absent bin / untouched page is sparse zeros
        assert reader.read(200 * PAGE, 4096) == b"\0" * 4096
        # compaction folds deltas into a generation bin on the middleman
        compact_trimmed(ram, work=tmp)
        reader2 = connect_tier("test", "ram", token="", journal_tag="ram-j", backend="middle",
                               middle_url=base, middle_token="tok", writer=False, cache_pages=8)
        assert reader2.read(addr, len(payload)) == payload
        names = c.list_names()
        assert any("mem-00.g" in n for n in names), names
        print(f"  middle RAM tier: journal write/read, sparse zeros, compaction ({sorted(names)[:2]})")
    finally:
        srv.shutdown()
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_blob_api()
    test_ram_tier_over_middle()
    print("middleman tests passed")