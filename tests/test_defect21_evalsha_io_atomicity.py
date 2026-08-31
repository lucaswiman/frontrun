"""Defect #21 investigation: ``detect_io=True`` vs Redis EVAL/EVALSHA atomicity.

Suspected defect: with ``detect_io=True`` the socket-IO detection path was
believed to interleave two clients' byte streams *inside* a single atomic
``EVALSHA``, fabricating a physically-impossible ``(success, success)``
over-admission for Lua-scripted rate limiters (a socket-path sibling of
defect #8, whose fix made the command-level path treat EVAL/EVALSHA/FCALL
as atomic).

Verdict after investigation: **NOT a frontrun defect**.  The ``detect_io=True``
path already respects single-command EVAL/EVALSHA atomicity — a rate limiter
whose check-then-write runs entirely server-side inside one Lua script is
deterministically green under ``detect_io=True`` (sync and async).  The
real-world red that motivated the suspicion (pyrate-limiter's ``RedisBucket``
via fastapi-limiter) turned out to be a TRUE positive: that library anchors
its Lua ``ZCOUNT`` window at a *client-side* timestamp captured before the
``EVALSHA``, so a caller holding a stale timestamp cannot observe a
newer-timestamped insert and both callers are admitted.  That interleaving
is reproducible against real Redis sequentially, with no frontrun involved —
EVALSHA atomicity does not cover the client-side capture→send window.

These tests pin the two sides of the correct behaviour:

1. A genuinely atomic single-EVAL/EVALSHA rate limiter (server-side ``INCR``
   check, no client-side state) never reds under ``detect_io=True``.
2. A non-atomic GET-check-then-INCR rate limiter (two separate commands)
   still reds under ``detect_io=True`` — EVAL atomicity handling must not
   over-suppress genuine multi-command TOCTOU races.

Running::

    REDIS_PORT=16399 frontrun python -m pytest tests/test_defect21_evalsha_io_atomicity.py -v
"""

from __future__ import annotations

import asyncio

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

import frontrun

pytestmark = pytest.mark.integration

LIMIT = 1

# Atomic check-then-conditional-admit rate limiter in a SINGLE Lua script.
# All state (the counter) and all decisions live server-side, so Redis's
# single-threaded atomic script execution makes double-admission impossible.
ATOMIC_LIMITER_SCRIPT = """
local c = redis.call('INCR', KEYS[1])
if c <= tonumber(ARGV[1]) then
  return 1
else
  return 0
end
"""

KEY = "defect21:limiter"


class TestAtomicEvalNoFalsePositive:
    """A single atomic EVAL/EVALSHA must not produce a fabricated race under detect_io=True."""

    @pytest.mark.parametrize("script_command", ["eval", "evalsha"])
    def test_sync_atomic_lua_green(self, redis_port: int, script_command: str) -> None:
        """Two threads, one atomic Lua command each: at most LIMIT successes, always green."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port)
                r.delete(KEY)
                self.script_sha = r.script_load(ATOMIC_LIMITER_SCRIPT) if script_command == "evalsha" else None
                r.close()
                self.results: list[int] = []

        def worker(state: State) -> None:
            r = redis_lib.Redis(port=port)
            if script_command == "evalsha":
                assert state.script_sha is not None
                res = r.evalsha(state.script_sha, 1, KEY, LIMIT)
            else:
                res = r.eval(ATOMIC_LIMITER_SCRIPT, 1, KEY, LIMIT)
            state.results.append(int(res))
            r.close()

        def invariant(state: State) -> bool:
            return sum(state.results) <= LIMIT

        result = frontrun.explore(
            setup=State,
            workers=[worker, worker],
            invariant=invariant,
            detect_io=True,
        )
        assert result.property_holds, (
            "FALSE POSITIVE: detect_io=True fabricated an over-admission for a "
            f"single atomic {script_command.upper()} — impossible on single-threaded Redis.\n" + str(result.explanation)
        )

    @pytest.mark.parametrize("script_command", ["eval", "evalsha"])
    def test_async_atomic_lua_green(self, redis_port: int, script_command: str) -> None:
        """Two asyncio tasks, one atomic Lua command each: always green under detect_io=True.

        Mirrors the fastapi-limiter scenario shape (async redis clients, one
        Lua script per task) minus pyrate-limiter's client-side timestamp,
        which is what actually races there.
        """
        import redis.asyncio as aioredis

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port)
                r.delete(KEY)
                self.script_sha = r.script_load(ATOMIC_LIMITER_SCRIPT) if script_command == "evalsha" else None
                r.close()
                self.results: list[int] = []

        async def worker(state: State) -> None:
            r = aioredis.Redis(port=port)
            if script_command == "evalsha":
                assert state.script_sha is not None
                res = await r.evalsha(state.script_sha, 1, KEY, LIMIT)
            else:
                res = await r.eval(ATOMIC_LIMITER_SCRIPT, 1, KEY, LIMIT)
            state.results.append(int(res))
            await r.aclose()

        def invariant(state: State) -> bool:
            return sum(state.results) <= LIMIT

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[worker, worker],
                invariant=invariant,
                detect_io=True,
            )
        )
        assert result.property_holds, (
            "FALSE POSITIVE: detect_io=True fabricated an over-admission for a "
            f"single atomic {script_command.upper()} — impossible on single-threaded Redis.\n" + str(result.explanation)
        )


class TestNonAtomicRaceStillDetected:
    """EVAL atomicity handling must not blind detect_io to real multi-command races."""

    def test_sync_get_then_incr_race_detected(self, redis_port: int) -> None:
        """GET-check-then-INCR (two commands, no Lua) is a real TOCTOU: must red."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port)
                r.delete(KEY)
                r.close()
                self.results: list[int] = []

        def worker(state: State) -> None:
            r = redis_lib.Redis(port=port)
            current = int(r.get(KEY) or 0)
            if current < LIMIT:
                r.incr(KEY)
                state.results.append(1)
            else:
                state.results.append(0)
            r.close()

        def invariant(state: State) -> bool:
            return sum(state.results) <= LIMIT

        result = frontrun.explore(
            setup=State,
            workers=[worker, worker],
            invariant=invariant,
            detect_io=True,
        )
        assert not result.property_holds, "OVER-SUPPRESSION: the non-atomic GET-then-INCR TOCTOU race was not detected"
        assert result.reproduction_attempts == 10
        assert result.reproduction_successes == 10, (
            f"real race must reproduce deterministically, got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_async_get_then_incr_race_detected(self, redis_port: int) -> None:
        """Async flavour of the same real TOCTOU race: must red."""
        import redis.asyncio as aioredis

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port)
                r.delete(KEY)
                r.close()
                self.results: list[int] = []

        async def worker(state: State) -> None:
            r = aioredis.Redis(port=port)
            current = int(await r.get(KEY) or 0)
            if current < LIMIT:
                await r.incr(KEY)
                state.results.append(1)
            else:
                state.results.append(0)
            await r.aclose()

        def invariant(state: State) -> bool:
            return sum(state.results) <= LIMIT

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[worker, worker],
                invariant=invariant,
                detect_io=True,
            )
        )
        assert not result.property_holds, (
            "OVER-SUPPRESSION: the non-atomic async GET-then-INCR TOCTOU race was not detected"
        )
        assert result.reproduction_attempts == 10
        assert result.reproduction_successes == 10, (
            f"real async race must reproduce deterministically, got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )
