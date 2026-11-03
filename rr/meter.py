"""Token-bucket Meter in front of every GitHub call. Honors x-ratelimit-* and Retry-After."""
import threading, time


class Bucket:
    def __init__(s, per_hour, burst=None):
        s.rate = per_hour / 3600.0; s.cap = burst or max(1, per_hour // 60); s.tok = s.cap
        s.t = time.monotonic(); s.lock = threading.Lock(); s.blocked_until = 0.0

    def take(s, n=1, block=True):
        while True:
            with s.lock:
                now = time.monotonic()
                s.tok = min(s.cap, s.tok + (now - s.t) * s.rate); s.t = now
                if now >= s.blocked_until and s.tok >= n:
                    s.tok -= n; return True
                wait = max(s.blocked_until - now, (n - s.tok) / s.rate if s.rate else 1)
            if not block: return False
            time.sleep(min(max(wait, 0.01), 30))

    def block_for(s, secs):
        with s.lock: s.blocked_until = max(s.blocked_until, time.monotonic() + secs)


class Meter:
    # per-hour caps well under GitHub limits (core 5000/h per PAT, secondary ~500 content writes/h)
    DEFAULTS = dict(core=4000, write=400, dispatch=200, upload=300, download=100000)

    def __init__(s, **caps):
        c = dict(s.DEFAULTS); c.update(caps)
        s.b = {k: Bucket(v) for k, v in c.items()}
        s.counts = {k: 0 for k in c}; s.trips = 0; s.lock = threading.Lock()

    def acquire(s, cls="core", n=1):
        s.b[cls].take(n)
        with s.lock: s.counts[cls] += n

    def observe(s, cls, status, headers):
        """Feed response headers; returns True if caller should retry."""
        h = {k.lower(): v for k, v in (headers or {}).items()}
        if status in (403, 429):
            ra = h.get("retry-after")
            if ra: s.b[cls].block_for(float(ra))
            elif h.get("x-ratelimit-remaining") == "0":
                s.b[cls].block_for(max(1.0, float(h.get("x-ratelimit-reset", time.time() + 60)) - time.time()))
            else: s.b[cls].block_for(60)
            with s.lock: s.trips += 1
            return True
        rem = h.get("x-ratelimit-remaining")
        if rem is not None and int(rem) < 100 and cls == "core":
            s.b[cls].block_for(5)
        return False

    def stats(s):
        with s.lock: return dict(counts=dict(s.counts), trips=s.trips)
