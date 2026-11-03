"""Status board composition and the 80%-quota pause threshold (no network)."""
from deepcore.status import StatusBoard


class Regs(dict):
    def get(self, k, d=None): return dict.get(self, k, d)
    def put(self, k, v): self[k] = v


class Budget:
    def snapshot(self):
        return {"day": "2026-11-03", "down": 3 << 30, "up": 1 << 30, "api": 120, "tokens": 400,
                "caps": {"down": 96 << 30, "up": 12 << 30, "api": 288000, "tokens": 24000}}


class Ram:
    gen, seq = 7, 142


class Env:
    def __init__(s):
        s.regs = Regs({"ring/started": 1, "ring/health": {f"r{i}": {"ok": True} for i in range(5)},
                       "gate/single": {"ok": True}, "ring/lease": {"holder": "deeprepo-cpu-3", "epoch": 4}})
        s.budget, s.ram = Budget(), Ram()
    def model_info(s): return {"key": "qwen3-30b-a3b", "ready": True}


def test_render_and_threshold():
    b = StatusBoard("owner/repo", "token")
    b.enabled = True
    b.quota = lambda ttl=60: (3241, 5000)
    body = b.render(Env(), runner="deeprepo-cpu-3", phase="serving")
    for needle in ("Runner", "deeprepo-cpu-3", "3241 / 5000", "v-ckpt-7", "**Journal seq** 142",
                   "Uptime", "Gate** pass"):
        assert needle in body, (needle, body)
    assert "paused" not in body
    # 80% used == 20% remaining == pause
    b.quota = lambda ttl=60: (1000, 5000)
    assert b.quota_low() is True
    b.quota = lambda ttl=60: (1001, 5000)
    assert b.quota_low() is False
    # missing token -> disabled, no network
    assert StatusBoard("owner/repo", "").enabled is False
    assert StatusBoard("reporam", "token").enabled is False
    print("  status render + 80% quota threshold ok")


if __name__ == "__main__":
    test_render_and_threshold()
    print("status tests passed")
