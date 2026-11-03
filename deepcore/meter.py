"""ToS-safety metering: token buckets per GitHub call class + a persistent daily byte budget.

The daily ledger lives in the reporam registers ("budget/<UTC-date>" records) so every
writer (coordinator + active runner) shares one budget across all repos. Hard caps fail
closed: BudgetExceeded stops work instead of silently burning bandwidth.
"""
import datetime, json, threading

from . import PAGE


class Bucket:
    def __init__(s, per_hour, burst=None):
        s.rate = per_hour / 3600.0
        s.cap = burst or max(1, per_hour // 60)
        s.tok = float(s.cap)
        s.t = 0.0
        s.lock = threading.Lock()

    def take(s, n=1, block=True):
        import time
        while True:
            with s.lock:
                now = time.monotonic()
                s.tok = min(s.cap, s.tok + (now - s.t) * s.rate)
                s.t = now
                if s.tok >= n:
                    s.tok -= n
                    return True
                wait = (n - s.tok) / s.rate if s.rate else 60
            if not block:
                return False
            import time
            time.sleep(min(max(wait, 0.05), 20))


class Budget:
    """Daily caps. Defaults are conservative for one standard-runner ring:
    down 96 GB/day (paged weights+KV reads, GitHub-internal CDN),
    up   12 GB/day (journal deltas + bin compaction uploads),
    api  288k calls/day, tokens 24k/day generated."""

    CAPS = dict(down=96 << 30, up=12 << 30, api=288_000, tokens=24_000)

    def __init__(s, ledger=None, caps=None, log=None):
        s.caps = dict(s.CAPS)
        s.caps.update(caps or {})
        s.log = log or print
        s.day = None
        s.today = dict(down=0, up=0, api=0, tokens=0)
        s.lk = threading.Lock()
        if ledger:
            s.adopt(ledger)

    def adopt(s, ledger_dict):
        """Load today's persisted usage from a snapshot ({"day","down",...}) or a date-keyed map."""
        day = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if ledger_dict and "day" in ledger_dict:
            rec = ledger_dict if ledger_dict.get("day") == day else {}
        else:
            rec = (ledger_dict or {}).get(day) or {}
        s.day = day
        s.today = dict(down=int(rec.get("down", 0)), up=int(rec.get("up", 0)),
                       api=int(rec.get("api", 0)), tokens=int(rec.get("tokens", 0)))

    def snapshot(s):
        with s.lk:
            return {"day": s.day, **s.today, "caps": dict(s.caps)}

    def _roll(s):
        day = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if s.day != day:
            s.day = day
            s.today = dict(down=0, up=0, api=0, tokens=0)

    def check(s, kind, n=1):
        """Raise BudgetExceeded if this would blow today's cap."""
        with s.lk:
            s._roll()
            if s.today[kind] + n > s.caps[kind]:
                raise BudgetExceeded(f"daily {kind} budget exhausted: {s.today[kind]}+{n} > {s.caps[kind]}")

    def spend(s, kind, n=1):
        with s.lk:
            s._roll()
            s.today[kind] += n
            return dict(s.today)

    def record(s, kind, n):
        s.check(kind, n)
        s.spend(kind, n)

    def exhausted(s):
        with s.lk:
            s._roll()
            return any(s.today[k] >= s.caps[k] for k in ("down", "up", "api", "tokens"))


class BudgetExceeded(RuntimeError):
    pass


class ApiMeter:
    """Token buckets in front of GitHub API call classes (cores/writes/uploads/dispatches)."""

    DEFAULTS = dict(core=3600, write=360, dispatch=180, upload=240, download=200_000)

    def __init__(s, caps=None):
        c = dict(s.DEFAULTS)
        c.update(caps or {})
        s.b = {k: Bucket(v) for k, v in c.items()}

    def acquire(s, cls="core", n=1):
        s.b[cls].take(n)


# Typical per-token paging arithmetic (for the budget doc + tests)
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim):
    return n_layers * n_kv_heads * head_dim * 2 * 2  # K+V, fp16


def expert_bytes_per_token(n_layers, experts_used, n_embd, n_ff, bpw=4.5):
    per = 3 * n_embd * n_ff * bpw / 8.0
    return n_layers * experts_used * per
