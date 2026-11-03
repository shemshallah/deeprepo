import os, sys, tempfile, unittest, random, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rr import SEG_PAYLOAD, SUB, HDR
from rr.rrbin import *
from rr.pack import pack_stream
from rr.store import DirStore
from rr.reader import Reader
from rr.manifest import Manifest
from rr.meter import Meter, Bucket
import io, time


class T(unittest.TestCase):
    def test_segment_roundtrip(self):
        d = os.urandom(SEG_PAYLOAD - 12345)
        for codec in (RAW, ZLIB):
            h, s = pack_segment(d, codec=codec)
            full = unpack_segment(h, s); self.assertEqual(full[:len(d)], d); self.assertEqual(full[len(d):], bytes(12345))
        h, s = pack_segment(b""); self.assertEqual(parse_header(h)["codec"], ZERO); self.assertEqual(s, b"")
        self.assertEqual(unpack_segment(h, s), bytes(SEG_PAYLOAD))

    def test_corruption_detected(self):
        h, s = pack_segment(os.urandom(5 << 20)); bad = bytearray(s); bad[3 * SUB + 17] ^= 1
        with self.assertRaises(RRError): unpack_segment(h, bytes(bad))
        bh = bytearray(h); bh[40] ^= 1
        with self.assertRaises(RRError): parse_header(bytes(bh))

    def test_pack_read_stripe(self):
        with tempfile.TemporaryDirectory() as t:
            st = DirStore(t); total = SEG_PAYLOAD * 5 + 777
            src = os.urandom(3 << 20) * (total // (3 << 20) + 1); src = src[:total]
            m = pack_stream(io.BytesIO(src), total, st, ["o/a", "o/b"], "rel", segs_per_bin=2, log=lambda *a: None)
            self.assertEqual(len(m.d["bins"]), 3)
            self.assertEqual({b["repo"] for b in m.d["bins"].values()}, {"o/a", "o/b"})
            r = Reader(m, st); rnd = random.Random(1)
            for _ in range(30):
                o = rnd.randrange(total - 1); n = rnd.randrange(1, min(5 * SUB, total - o))
                self.assertEqual(r.read("ram", o, n), src[o:o + n])
            # ranged read fetched far less than the whole thing
            self.assertLess(r.bytes_fetched, total)
            # corruption inside a bin is caught on ranged read
            p = st._p("o/a", "rel", "rrram-0.bin")
            with open(p, "r+b") as f: f.seek(HDR + 100); b = f.read(1); f.seek(HDR + 100); f.write(bytes([b[0] ^ 255]))
            r2 = Reader(m, st)
            with self.assertRaises(RRError): r2.read("ram", 0, 10)

    def test_manifest_cow(self):
        m = Manifest(); m.add_space("ram", 10); m.set_seg("ram", 0, "b", 0, 1, 0, b"\1" * 32)
        n = m.next_generation(); n.set_seg("ram", 0, "c", 0, 1, 0, b"\2" * 32)
        self.assertEqual(n.d["generation"], 1); self.assertEqual(m.seg("ram", 0)["bin"], "b")
        self.assertEqual(n.d["parent"], m.digest())
        self.assertEqual(Manifest.loads(n.dumps(), n.digest()).digest(), n.digest())
        with self.assertRaises(ValueError): Manifest.loads(n.dumps(), "0" * 64)

    def test_meter(self):
        b = Bucket(3600, burst=5)
        t = time.time()
        for _ in range(5): b.take()
        self.assertLess(time.time() - t, 0.5)
        self.assertFalse(b.take(block=False))
        m = Meter(); self.assertTrue(m.observe("core", 429, {"Retry-After": "0.01"})); self.assertEqual(m.trips, 1)

if __name__ == "__main__": unittest.main()
