"""End-to-end cross-process Redis exploration (Phase 2c).

Spawns real subprocesses running plain ``redis-py`` code against a live Redis
server; the coordinator interleaves them via the DPOR engine and checks the
counter. Marked ``integration``+``e2e``; skipped unless the ``redis`` package is
installed and a server answers on ``FRONTRUN_XPROC_REDIS_URL`` (default local).
"""

from __future__ import annotations

import pytest

import frontrun

redis = pytest.importorskip("redis")

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

_TARGET = "tests.xproc_demo_redis:increment"
_ATOMIC_TARGET = "tests.xproc_demo_redis:increment_atomic"


@pytest.fixture
def redis_demo():
    from tests import xproc_demo_redis as mod

    try:
        mod.setup()
    except redis.exceptions.RedisError as exc:  # server not running
        pytest.skip(f"redis server unavailable: {exc}")
    return mod


def test_redis_lost_update_found_across_processes(redis_demo) -> None:
    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(_TARGET),
            "w1": frontrun.Subprocess(_TARGET),
        },
        setup=redis_demo.setup,
        invariant=lambda _state: redis_demo.read() == 2,
    )
    assert not result.ok
    assert result.failure_kind == "invariant"


def test_redis_atomic_incr_has_no_race(redis_demo) -> None:
    result = frontrun.explore_processes(
        {
            "w0": frontrun.Subprocess(_ATOMIC_TARGET),
            "w1": frontrun.Subprocess(_ATOMIC_TARGET),
        },
        setup=redis_demo.setup,
        invariant=lambda _state: redis_demo.read() == 2,
    )
    assert result.ok, f"unexpected {result.failure_kind}: {result.failure!r}"
    assert result.exhausted
