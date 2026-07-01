"""A tiny Redis counter for demonstrating/e2e-testing cross-process exploration.

Package-internal so a spawned worker can import it by ``module:callable`` name.
``increment`` is the racy GET/SET target; ``increment_atomic`` uses Redis INCR.
``setup`` and ``read`` run in the coordinator process. All connect to the Redis
URL in the ``FRONTRUN_XPROC_REDIS_URL`` environment variable (default local).
"""

from __future__ import annotations

import os


def _client():
    import redis  # type: ignore[import-not-found]  # optional dep; lazily imported

    url = os.environ.get("FRONTRUN_XPROC_REDIS_URL", "redis://127.0.0.1:6379/0")
    return redis.Redis.from_url(url)


def setup() -> None:
    """Reset the counter key to 0 (coordinator side)."""
    client = _client()
    client.set("frontrun:xproc:counter", 0)


def read() -> int:
    """Return the current counter value (coordinator-side invariant check)."""
    client = _client()
    raw = client.get("frontrun:xproc:counter")
    return int(raw) if raw is not None else 0


def increment() -> None:
    """Racy read-modify-write: GET then SET (two scheduling points)."""
    client = _client()
    current = int(client.get("frontrun:xproc:counter") or 0)
    client.set("frontrun:xproc:counter", current + 1)


def increment_atomic() -> None:
    """Safe increment via Redis INCR (one atomic scheduling point)."""
    client = _client()
    client.incr("frontrun:xproc:counter")
