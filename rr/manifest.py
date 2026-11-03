"""Manifest with generations (copy-on-write). Maps (space, vseg) -> location in a bin."""
import hashlib, json
from . import SEG_PAYLOAD, HDR


class Manifest:
    def __init__(s, d=None):
        s.d = d or dict(version=1, generation=0, parent=None, spaces={}, bins={}, segs={}, heat=[], meta={})

    # spaces: name -> dict(size, page_kb); bins: name -> dict(repo, release, size); segs: "space:vseg" -> loc
    def add_space(s, name, size, page_kb=256): s.d["spaces"][name] = dict(size=size, page_kb=page_kb)

    def add_bin(s, name, repo, release, size): s.d["bins"][name] = dict(repo=repo, release=release, size=size)

    def set_seg(s, space, vseg, bin, off, stored_len, codec, hdr_sha):
        s.d["segs"][f"{space}:{vseg}"] = dict(bin=bin, off=off, len=stored_len, codec=codec, hdr=hdr_sha.hex() if isinstance(hdr_sha, bytes) else hdr_sha)

    def seg(s, space, vseg): return s.d["segs"].get(f"{space}:{vseg}")

    def next_generation(s):
        """Copy-on-write: new generation references same segs; caller overrides changed ones."""
        n = Manifest(json.loads(json.dumps(s.d)))
        n.d["parent"] = s.digest(); n.d["generation"] = s.d["generation"] + 1
        return n

    def dumps(s): return json.dumps(s.d, sort_keys=True, separators=(",", ":"))

    def digest(s): return hashlib.sha256(s.dumps().encode()).hexdigest()

    @staticmethod
    def loads(txt, expect_sha=None):
        if expect_sha and hashlib.sha256(txt.encode()).hexdigest() != expect_sha: raise ValueError("manifest sha mismatch")
        return Manifest(json.loads(txt))

    def nsegs(s, space): return -(-s.d["spaces"][space]["size"] // SEG_PAYLOAD)

    def referenced_bins(s): return sorted({v["bin"] for v in s.d["segs"].values()})
