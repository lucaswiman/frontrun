"""A tiny Redis counter used by cross-process Redis e2e tests."""

from __future__ import annotations

import os


def _client():
    import redis  # type: ignore[import-not-found]  # optional dep; lazily imported

    url = os.environ.get("FRONTRUN_XPROC_REDIS_URL", "redis://127.0.0.1:6379/0")
    return redis.Redis.from_url(url)


def setup() -> None:
    """Reset the counter key to 0."""
    client = _client()
    client.set("frontrun:xproc:counter", 0)


def read() -> int:
    """Return the current counter value."""
    client = _client()
    raw = client.get("frontrun:xproc:counter")
    return int(raw) if raw is not None else 0


def increment() -> None:
    """Racy read-modify-write: GET then SET."""
    client = _client()
    current = int(client.get("frontrun:xproc:counter") or 0)
    client.set("frontrun:xproc:counter", current + 1)


def increment_atomic() -> None:
    """Safe increment via Redis INCR."""
    client = _client()
    client.incr("frontrun:xproc:counter")
