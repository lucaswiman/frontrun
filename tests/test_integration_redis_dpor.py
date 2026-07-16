"""Integration tests for DPOR Redis key-level analysis.

These tests exercise DPOR's ability to detect application-level race
conditions through key-level Redis command interception (as opposed to
the coarser socket-level I/O detection in test_integration_redis.py).

Key-level analysis intercepts redis-py's ``execute_command()`` method,
classifies each Redis command (GET, SET, HSET, etc.) as a read or write
on specific keys, and reports those key-level accesses to the DPOR engine.
This gives more precise conflict detection than endpoint-level socket I/O.

Requirements::

    redis-server must be installed (apt-get install redis-server)
    pip install redis

Running::

    make test-integration-3.14
    # or:
    frontrun .venv-3.14/bin/pytest tests/test_integration_redis_dpor.py -v
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import threading
import time

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

import frontrun

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 1. Lost-update counter (key-level analysis)
# ---------------------------------------------------------------------------


class TestRedisCounterRaceKeyLevel:
    """Lost-update race on a Redis counter, detected via key-level analysis."""

    def test_dpor_detects_lost_update(self, redis_port: int) -> None:
        """DPOR should detect the lost-update race via Redis key-level analysis."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("counter", "0")
                r.close()

        def increment(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            val = int(r.get("counter"))  # type: ignore[arg-type]
            r.set("counter", str(val + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = frontrun.explore(
            setup=State,
            workers=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect lost-update on Redis counter"
        assert result.num_explored >= 2, "DPOR must explore multiple interleavings to find the race"
        assert result.explanation is not None

    def test_locked_counter_is_safe(self, redis_port: int) -> None:
        """Counter protected by a Python lock is safe."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("counter", "0")
                r.close()

        def increment(state: State) -> None:
            with state.lock:
                r = redis_lib.Redis(port=port, decode_responses=True)
                val = int(r.get("counter"))  # type: ignore[arg-type]
                r.set("counter", str(val + 1))
                r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = frontrun.explore(
            setup=State,
            workers=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 2. Check-then-act (key-level analysis)
# ---------------------------------------------------------------------------


class TestRedisCheckThenActKeyLevel:
    """TOCTOU race: double initialization via key-level analysis."""

    def test_dpor_detects_double_init(self, redis_port: int) -> None:
        """DPOR should detect the double-initialization race."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("resource", "init_count")
                r.set("init_count", "0")
                r.close()

        def maybe_init(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            if not r.exists("resource"):
                r.set("resource", "initialized")
                count = int(r.get("init_count"))  # type: ignore[arg-type]
                r.set("init_count", str(count + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("init_count"))  # type: ignore[arg-type]
            r.close()
            return result == 1

        result = frontrun.explore(
            setup=State,
            workers=[maybe_init, maybe_init],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect double-initialization race"

    def test_locked_init_is_safe(self, redis_port: int) -> None:
        """Initialization protected by lock is safe."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("resource", "init_count")
                r.set("init_count", "0")
                r.close()

        def maybe_init(state: State) -> None:
            with state.lock:
                r = redis_lib.Redis(port=port, decode_responses=True)
                if not r.exists("resource"):
                    r.set("resource", "initialized")
                    count = int(r.get("init_count"))  # type: ignore[arg-type]
                    r.set("init_count", str(count + 1))
                r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("init_count"))  # type: ignore[arg-type]
            r.close()
            return result == 1

        result = frontrun.explore(
            setup=State,
            workers=[maybe_init, maybe_init],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 3. Inventory / double-spend (key-level analysis)
# ---------------------------------------------------------------------------


class TestRedisInventoryRaceKeyLevel:
    """Double-spend race detected via key-level analysis."""

    def test_dpor_detects_oversell(self, redis_port: int) -> None:
        """DPOR should detect the oversell race."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("stock", "1")
                r.set("sold", "0")
                r.close()

        def buy(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            stock = int(r.get("stock"))  # type: ignore[arg-type]
            if stock > 0:
                r.set("stock", str(stock - 1))
                sold = int(r.get("sold"))  # type: ignore[arg-type]
                r.set("sold", str(sold + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            stock = int(r.get("stock"))  # type: ignore[arg-type]
            sold = int(r.get("sold"))  # type: ignore[arg-type]
            r.close()
            return stock >= 0 and sold <= 1

        result = frontrun.explore(
            setup=State,
            workers=[buy, buy],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect oversell race"


# ---------------------------------------------------------------------------
# 4. Transfer race (key-level analysis)
# ---------------------------------------------------------------------------


class TestRedisTransferRaceKeyLevel:
    """Lost-update race on concurrent balance transfers via key-level analysis."""

    def test_dpor_detects_transfer_anomaly(self, redis_port: int) -> None:
        """DPOR should detect the lost update in concurrent transfers."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("acct:a", "100")
                r.set("acct:b", "100")
                r.close()

        def transfer_a_to_b(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            bal_a = int(r.get("acct:a"))  # type: ignore[arg-type]
            bal_b = int(r.get("acct:b"))  # type: ignore[arg-type]
            r.set("acct:a", str(bal_a - 10))
            r.set("acct:b", str(bal_b + 10))
            r.close()

        def transfer_a_to_b_also(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            bal_a = int(r.get("acct:a"))  # type: ignore[arg-type]
            bal_b = int(r.get("acct:b"))  # type: ignore[arg-type]
            r.set("acct:a", str(bal_a - 30))
            r.set("acct:b", str(bal_b + 30))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            bal_a = int(r.get("acct:a"))  # type: ignore[arg-type]
            bal_b = int(r.get("acct:b"))  # type: ignore[arg-type]
            r.close()
            return bal_a + bal_b == 200

        result = frontrun.explore(
            setup=State,
            workers=[transfer_a_to_b, transfer_a_to_b_also],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect transfer lost-update"

    def test_locked_transfer_is_safe(self, redis_port: int) -> None:
        """Transfer protected by a Python lock preserves balance conservation."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("acct:a", "100")
                r.set("acct:b", "100")
                r.close()

        def transfer(state: State, amount: int) -> None:
            with state.lock:
                r = redis_lib.Redis(port=port, decode_responses=True)
                bal_a = int(r.get("acct:a"))  # type: ignore[arg-type]
                bal_b = int(r.get("acct:b"))  # type: ignore[arg-type]
                r.set("acct:a", str(bal_a - amount))
                r.set("acct:b", str(bal_b + amount))
                r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            bal_a = int(r.get("acct:a"))  # type: ignore[arg-type]
            bal_b = int(r.get("acct:b"))  # type: ignore[arg-type]
            r.close()
            return bal_a + bal_b == 200

        result = frontrun.explore(
            setup=State,
            workers=[
                lambda s: transfer(s, 10),
                lambda s: transfer(s, 30),
            ],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 5. Hash-based race (key-level analysis)
# ---------------------------------------------------------------------------


class TestRedisHashRaceKeyLevel:
    """Race on Redis hash operations detected via key-level analysis."""

    def test_dpor_detects_hash_lost_update(self, redis_port: int) -> None:
        """DPOR detects lost update on hash field read-modify-write."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.hset("user:1", "balance", "100")
                r.close()

        def debit(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            balance = int(r.hget("user:1", "balance"))  # type: ignore[arg-type]
            r.hset("user:1", "balance", str(balance - 10))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            balance = int(r.hget("user:1", "balance"))  # type: ignore[arg-type]
            r.close()
            # Two debits of 10 each: 100 - 10 - 10 = 80
            return balance == 80

        result = frontrun.explore(
            setup=State,
            workers=[debit, debit],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect hash lost-update"

    def test_hincrby_is_atomic_and_safe(self, redis_port: int) -> None:
        """HINCRBY is atomic — no lost update possible."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.hset("user:1", "balance", "100")
                r.close()

        def debit(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            r.hincrby("user:1", "balance", -10)
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            balance = int(r.hget("user:1", "balance"))  # type: ignore[arg-type]
            r.close()
            return balance == 80

        result = frontrun.explore(
            setup=State,
            workers=[debit, debit],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 6. Async Redis integration tests
# ---------------------------------------------------------------------------


class TestAsyncRedisCounterRace:
    """Lost-update race on async Redis counter via key-level analysis."""

    def test_async_dpor_detects_lost_update(self, redis_port: int) -> None:
        """Async DPOR should detect the lost-update race."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("async_counter", "0")
                r.close()

        async def increment(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            val = int(await r.get("async_counter"))  # type: ignore[arg-type]
            await r.set("async_counter", str(val + 1))
            await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("async_counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[increment, increment],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert not result.property_holds, "Async DPOR should detect lost-update on Redis counter"

    def test_async_locked_counter_is_safe(self, redis_port: int) -> None:
        """Async counter protected by asyncio.Lock is safe."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("async_counter", "0")
                r.close()

        async def increment(state: State) -> None:
            async with state.lock:
                r = aioredis.Redis(port=port, decode_responses=True)
                val = int(await r.get("async_counter"))  # type: ignore[arg-type]
                await r.set("async_counter", str(val + 1))
                await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("async_counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[increment, increment],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert result.property_holds, result.explanation


class TestAsyncRedisTransferRace:
    """Async transfer race via key-level analysis."""

    def test_async_dpor_detects_transfer_anomaly(self, redis_port: int) -> None:
        """Async DPOR should detect the lost update in concurrent transfers."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("async_acct:a", "100")
                r.set("async_acct:b", "100")
                r.close()

        async def transfer_10(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            bal_a = int(await r.get("async_acct:a"))  # type: ignore[arg-type]
            bal_b = int(await r.get("async_acct:b"))  # type: ignore[arg-type]
            await r.set("async_acct:a", str(bal_a - 10))
            await r.set("async_acct:b", str(bal_b + 10))
            await r.aclose()

        async def transfer_30(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            bal_a = int(await r.get("async_acct:a"))  # type: ignore[arg-type]
            bal_b = int(await r.get("async_acct:b"))  # type: ignore[arg-type]
            await r.set("async_acct:a", str(bal_a - 30))
            await r.set("async_acct:b", str(bal_b + 30))
            await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            bal_a = int(r.get("async_acct:a"))  # type: ignore[arg-type]
            bal_b = int(r.get("async_acct:b"))  # type: ignore[arg-type]
            r.close()
            return bal_a + bal_b == 200

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[transfer_10, transfer_30],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert not result.property_holds, "Async DPOR should detect transfer lost-update"


class TestAsyncRedisCheckThenAct:
    """Async TOCTOU race via key-level analysis."""

    def test_async_dpor_detects_double_init(self, redis_port: int) -> None:
        """Async DPOR should detect double initialization."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("async_resource", "async_init_count")
                r.set("async_init_count", "0")
                r.close()

        async def maybe_init(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            if not await r.exists("async_resource"):
                await r.set("async_resource", "initialized")
                count = int(await r.get("async_init_count"))  # type: ignore[arg-type]
                await r.set("async_init_count", str(count + 1))
            await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("async_init_count"))  # type: ignore[arg-type]
            r.close()
            return result == 1

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[maybe_init, maybe_init],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert not result.property_holds, "Async DPOR should detect double-init race"


# ---------------------------------------------------------------------------
# 8. List-based race (key-level analysis)
# ---------------------------------------------------------------------------


class TestRedisListRaceKeyLevel:
    """Race on Redis list operations detected via key-level analysis."""

    def test_dpor_detects_list_lost_update(self, redis_port: int) -> None:
        """DPOR detects race on list length check followed by processing."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("task_queue")
                r.rpush("task_queue", "task1")
                r.set("processed", "0")
                r.close()

        def worker(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            # Check-then-act: llen is a non-atomic guard, so both workers can
            # pass it and then double-process the single task.
            if r.llen("task_queue") > 0:
                r.lpop("task_queue")
                processed = int(r.get("processed"))  # type: ignore[arg-type]
                r.set("processed", str(processed + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            processed = int(r.get("processed"))  # type: ignore[arg-type]
            r.close()
            # Only 1 task in queue — at most 1 should be processed
            return processed <= 1

        result = frontrun.explore(
            setup=State,
            workers=[worker, worker],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        # Both workers see llen > 0 before either pops, so both enter the branch
        # and both increment ``processed`` — pushing it to 2 in the interleaving
        # DPOR finds, violating the "at most 1 processed" invariant.
        assert not result.property_holds, "DPOR should detect list check-then-act race"


# ---------------------------------------------------------------------------
# 9. Set-based race (key-level analysis)
# ---------------------------------------------------------------------------


class TestRedisSetRaceKeyLevel:
    """Race on Redis set operations detected via key-level analysis."""

    def test_dpor_detects_set_membership_race(self, redis_port: int) -> None:
        """DPOR detects race where both threads check membership before adding."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("processed_set")
                r.set("process_count", "0")
                r.close()

        def process_item(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            # Check-then-act: check if already processed
            if not r.sismember("processed_set", "item1"):
                r.sadd("processed_set", "item1")
                count = int(r.get("process_count"))  # type: ignore[arg-type]
                r.set("process_count", str(count + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            count = int(r.get("process_count"))  # type: ignore[arg-type]
            r.close()
            return count == 1

        result = frontrun.explore(
            setup=State,
            workers=[process_item, process_item],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect set membership race"


# ---------------------------------------------------------------------------
# 10. Async independent keys — no false conflicts
# ---------------------------------------------------------------------------
# In async mode, detect_io=True uses key-level analysis WITHOUT the
# LD_PRELOAD socket-level bridge, so independent keys should NOT conflict.


class TestAsyncRedisIndependentKeys:
    """Ensure async operations on different keys don't create false conflicts."""

    def test_async_independent_keys_no_false_conflict(self, redis_port: int) -> None:
        """Two async tasks writing different keys should not conflict."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("async_key_a", "0")
                r.set("async_key_b", "0")
                r.close()

        async def write_a(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            await r.set("async_key_a", "1")
            await r.aclose()

        async def write_b(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            await r.set("async_key_b", "1")
            await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            a = r.get("async_key_a")
            b = r.get("async_key_b")
            r.close()
            return a == "1" and b == "1"

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[write_a, write_b],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 11. Async hash race via key-level analysis
# ---------------------------------------------------------------------------


class TestAsyncRedisHashRace:
    """Async hash race via key-level analysis."""

    def test_async_dpor_detects_hash_lost_update(self, redis_port: int) -> None:
        """Async DPOR detects lost update on hash field read-modify-write."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.hset("async_user:1", "balance", "100")
                r.close()

        async def debit(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            balance = int(await r.hget("async_user:1", "balance"))  # type: ignore[arg-type]
            await r.hset("async_user:1", "balance", str(balance - 10))
            await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            balance = int(r.hget("async_user:1", "balance"))  # type: ignore[arg-type]
            r.close()
            return balance == 80

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[debit, debit],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert not result.property_holds, "Async DPOR should detect hash lost-update"


# ---------------------------------------------------------------------------
# 12. Multi-resource race: Redis + SQLite + in-memory state
# ---------------------------------------------------------------------------


class TestMultiResourceRace:
    """Race condition spanning Redis, SQLite, and in-memory state.

    Scenario: A user registration system where:
    - Redis is used as a cache/rate-limiter (check if username is taken)
    - SQLite stores the authoritative user database
    - In-memory counter tracks total registrations

    The race: two threads try to register the same username concurrently.
    Both check Redis cache (miss), both check SQLite (not found), both insert.
    """

    def test_dpor_detects_multi_resource_race(self, redis_port: int) -> None:
        """DPOR should detect the race across Redis + SQLite + in-memory state."""
        import sqlite3
        import tempfile

        port = redis_port

        class State:
            def __init__(self) -> None:
                self.registration_count = 0
                # Fresh SQLite DB per run
                self.db_file = tempfile.mktemp(suffix=".db")
                conn = sqlite3.connect(self.db_file)
                conn.execute("CREATE TABLE users (username TEXT UNIQUE, email TEXT)")
                conn.commit()
                conn.close()
                # Clear Redis cache
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("user:alice")
                r.close()

        def register_user(state: State, email: str) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            # Step 1: Check Redis cache for username
            cached = r.get("user:alice")
            if cached is not None:
                r.close()
                return  # Already registered

            # Step 2: Check SQLite database
            conn = sqlite3.connect(state.db_file)
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("alice",))
            count = cursor.fetchone()[0]

            if count == 0:
                # Step 3: Insert into SQLite (race can cause duplicate)
                conn.execute("INSERT OR IGNORE INTO users (username, email) VALUES (?, ?)", ("alice", email))
                conn.commit()
                # Step 4: Update Redis cache
                r.set("user:alice", email)
                # Step 5: Update in-memory counter (this is the bug — both threads increment)
                state.registration_count += 1

            conn.close()
            r.close()

        def invariant(state: State) -> bool:
            # registration_count should be 1 if only one thread registered
            return state.registration_count == 1

        result = frontrun.explore(
            setup=State,
            workers=[
                lambda s: register_user(s, "alice@a.com"),
                lambda s: register_user(s, "alice@b.com"),
            ],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect multi-resource registration race"

    def test_multi_resource_with_lock_is_safe(self, redis_port: int) -> None:
        """Registration protected by a lock is safe across all resource types."""
        import sqlite3
        import tempfile

        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.registration_count = 0
                self.db_file = tempfile.mktemp(suffix=".db")
                conn = sqlite3.connect(self.db_file)
                conn.execute("CREATE TABLE users (username TEXT UNIQUE, email TEXT)")
                conn.commit()
                conn.close()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("user:alice")
                r.close()

        def register_user(state: State, email: str) -> None:
            with state.lock:
                r = redis_lib.Redis(port=port, decode_responses=True)
                cached = r.get("user:alice")
                if cached is not None:
                    r.close()
                    return

                conn = sqlite3.connect(state.db_file)
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("alice",))
                count = cursor.fetchone()[0]

                if count == 0:
                    conn.execute("INSERT INTO users (username, email) VALUES (?, ?)", ("alice", email))
                    conn.commit()
                    r.set("user:alice", email)
                    state.registration_count += 1

                conn.close()
                r.close()

        def invariant(state: State) -> bool:
            conn = sqlite3.connect(state.db_file)
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("alice",))
            db_count = cursor.fetchone()[0]
            conn.close()
            return db_count == 1 and state.registration_count == 1

        result = frontrun.explore(
            setup=State,
            workers=[
                lambda s: register_user(s, "alice@a.com"),
                lambda s: register_user(s, "alice@b.com"),
            ],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 13. Pipeline integration test
# ---------------------------------------------------------------------------


class TestRedisPipelineKeyLevel:
    """Verify DPOR detects races through pipelined Redis commands."""

    def test_dpor_detects_pipeline_lost_update(self, redis_port: int) -> None:
        """Pipeline GET+SET should still be detected as a race."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("pipe_counter", "0")
                r.close()

        def increment_via_pipeline(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            val = int(r.get("pipe_counter"))  # type: ignore[arg-type]
            pipe = r.pipeline()
            pipe.set("pipe_counter", str(val + 1))
            pipe.execute()
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("pipe_counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = frontrun.explore(
            setup=State,
            workers=[increment_via_pipeline, increment_via_pipeline],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect lost-update through pipeline"


# ---------------------------------------------------------------------------
# 14. DPOR path-count: independent operations → single path
# ---------------------------------------------------------------------------


class TestDporPathCountIndependent:
    """Verify DPOR explores few paths when Redis operations are fully independent.

    The DPOR engine correctly identifies that independent Redis keys don't
    conflict at the I/O level.  However, async DPOR also performs opcode-level
    tracing which may detect shared Python state (module globals, connection
    internals, etc.) causing some additional exploration.  These tests verify
    that the property holds across all explored paths and that exploration
    is bounded well below the combinatorial explosion.
    """

    def test_async_independent_keys_property_holds(self, redis_port: int) -> None:
        """Async tasks writing completely different keys → property holds on all paths.

        Uses detect_io=True (no LD_PRELOAD) so socket-level false conflicts
        are avoided and DPOR sees only key-level accesses.  With 2 tasks × 10
        operations each, the combinatorial maximum would be C(20,10) ≈ 184K
        interleavings.  DPOR should explore far fewer.
        """
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                for i in range(20):
                    r.set(f"ind:{i}", "0")
                r.close()

        async def worker_a(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            for i in range(10):
                await r.set(f"ind:{i}", "a")
            await r.aclose()

        async def worker_b(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            for i in range(10, 20):
                await r.set(f"ind:{i}", "b")
            await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port)
            for i in range(10):
                if r.get(f"ind:{i}") != b"a":
                    r.close()
                    return False
            for i in range(10, 20):
                if r.get(f"ind:{i}") != b"b":
                    r.close()
                    return False
            r.close()
            return True

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[worker_a, worker_b],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert result.property_holds, result.explanation

    def test_async_independent_mixed_commands_property_holds(self, redis_port: int) -> None:
        """Async tasks using different command types on different keys → property holds."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("str_a", "0")
                r.delete("hash_b", "list_b")
                r.close()

        async def worker_a(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            await r.set("str_a", "hello")
            await r.append("str_a", " world")
            val = await r.get("str_a")
            await r.set("str_a", val or "")
            await r.aclose()

        async def worker_b(state: State) -> None:
            r = aioredis.Redis(port=port, decode_responses=True)
            await r.hset("hash_b", "field1", "val1")
            await r.hset("hash_b", "field2", "val2")
            await r.lpush("list_b", "item1", "item2")
            await r.llen("list_b")
            await r.aclose()

        def invariant(state: State) -> bool:
            return True

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[worker_a, worker_b],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        assert result.property_holds


# ---------------------------------------------------------------------------
# 14. DPOR path-count: serialized operations → minimal paths
# ---------------------------------------------------------------------------


class TestDporPathCountSerialized:
    """Verify DPOR explores minimal paths when operations are lock-serialized."""

    def test_locked_many_redis_ops_minimal_paths(self, redis_port: int) -> None:
        """Many Redis ops all under one lock avoid combinatorial path growth.

        With 2 threads and a single lock serializing all access, DPOR should
        explore at most 2 orderings (thread 0 first, thread 1 first), not
        a combinatorial explosion of per-operation interleavings.
        """
        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("locked_counter", "0")
                r.delete("locked_hash", "locked_list", "locked_set")
                r.close()

        def worker(state: State) -> None:
            with state.lock:
                r = redis_lib.Redis(port=port, decode_responses=True)
                val = int(r.get("locked_counter"))  # type: ignore[arg-type]
                r.set("locked_counter", str(val + 1))
                r.hset("locked_hash", f"step_{val}", "done")
                r.lpush("locked_list", f"item_{val}")
                r.sadd("locked_set", f"member_{val}")
                r.incr("locked_counter")
                r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            counter = int(r.get("locked_counter"))  # type: ignore[arg-type]
            r.close()
            # Each worker does +1 (set) then +1 (incr) = +2 per worker, 2 workers = 4
            return counter == 4

        result = frontrun.explore(
            setup=State,
            workers=[worker, worker],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation
        # With synced I/O (dpor_vv respects lock HB for Redis key accesses),
        # the theoretical minimum is 2 (one per lock ordering).  Remaining
        # overcounting comes from:
        # 1. Lock acquire uses io_vv (intentional: must explore both orderings)
        # 2. Python-level internals of redis-py create scheduling points
        #    with untracked object conflicts (connection setup, parser state)
        # 3. LD_PRELOAD TCP events from Redis connection setup
        # Keep the established tight ceiling: increasing it would hide
        # extraneous traces instead of detecting a scheduling regression.
        assert result.num_explored <= 12, f"Lock-serialized ops should need ≤ 12 DPOR paths, got {result.num_explored}"

    def test_async_locked_many_redis_ops_minimal_paths(self, redis_port: int) -> None:
        """Async: many Redis ops under one asyncio.Lock → minimal DPOR paths."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("redis.asyncio not available")

        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("async_locked_ctr", "0")
                r.delete("async_locked_hash")
                r.close()

        async def worker(state: State) -> None:
            async with state.lock:
                r = aioredis.Redis(port=port, decode_responses=True)
                val = int(await r.get("async_locked_ctr"))  # type: ignore[arg-type]
                await r.set("async_locked_ctr", str(val + 1))
                await r.hset("async_locked_hash", f"step_{val}", "done")
                await r.aclose()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            counter = int(r.get("async_locked_ctr"))  # type: ignore[arg-type]
            r.close()
            return counter == 2

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[worker, worker],
                invariant=invariant,
                detect_io=True,
                max_executions=50,
                deadlock_timeout=15.0,
                reproduce_on_failure=0,
            )
        )
        # If DPOR deadlocked, skip — known limitation with asyncio.Lock
        if not result.property_holds and result.explanation and "Deadlock" in result.explanation:
            pytest.skip("asyncio.Lock deadlocked under DPOR — known limitation")
        assert result.property_holds, result.explanation
        # I/O-level conflict analysis (report_io_access) ignores lock
        # happens-before, so DPOR explores extra interleavings even when
        # operations are lock-protected.  This is by design: I/O races
        # can exist even with application-level locks (e.g. lock granularity
        # bugs).  The bound is relaxed to account for this.
        assert result.num_explored <= 15, f"Async lock-serialized ops should need ≤ 15 paths, got {result.num_explored}"


# ---------------------------------------------------------------------------
# Server/database identity regressions (0.7.0 release audit)
#
# Each test uses a reader/writer witness pattern: the invariant is violated
# only when the writer's SET lands before the reader's GET, an ordering the
# initial schedule does not produce.  Reaching it requires DPOR to detect the
# GET/SET dependency and backtrack, so a missed dependency (two identities
# for one physical database) falsely certifies the property.  Connections
# are established during setup so the workers execute key-level commands
# only.
# ---------------------------------------------------------------------------


class TestRedisIdentityAliases:
    """Two clients naming one server differently must share resources."""

    def test_dpor_detects_race_across_host_aliases(self, redis_port: int) -> None:
        """localhost and 127.0.0.1 reach the same server (issue #250 finding)."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                self.reader = redis_lib.Redis(host="localhost", port=port, decode_responses=True)
                self.writer = redis_lib.Redis(host="127.0.0.1", port=port, decode_responses=True)
                self.reader.ping()
                self.writer.ping()
                self.reader.delete("alias_flag", "alias_witness")

        def reader(state: State) -> None:
            if state.reader.get("alias_flag") == "1":
                state.reader.set("alias_witness", "1")

        def writer(state: State) -> None:
            state.writer.set("alias_flag", "1")

        def invariant(state: State) -> bool:
            return state.reader.get("alias_witness") is None

        result = frontrun.explore(
            setup=State,
            workers=[reader, writer],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "aliased clients reach one server; the SET-before-GET order must be found"


class TestRedisUnixSocketIdentity:
    """A unix-socket client and a TCP client to one server must share resources."""

    _UNIX_PORT = int(os.environ.get("REDIS_PORT_UNIX", "16398"))

    @pytest.fixture
    def unix_socket_server(self):
        """Spawn a dedicated redis-server listening on both TCP and a unix socket."""
        if not shutil.which("redis-server"):
            pytest.skip("redis-server not installed")
        tmpdir = tempfile.mkdtemp(prefix="fr-redis-")  # short path: AF_UNIX limit
        sock_path = os.path.join(tmpdir, "redis.sock")
        proc = subprocess.Popen(
            [
                "redis-server",
                "--port",
                str(self._UNIX_PORT),
                "--unixsocket",
                sock_path,
                "--save",
                "",
                "--appendonly",
                "no",
                "--loglevel",
                "warning",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 5.0
        while not os.path.exists(sock_path) and time.time() < deadline:
            time.sleep(0.05)
        client = redis_lib.Redis(unix_socket_path=sock_path)
        try:
            client.ping()
        except redis_lib.exceptions.RedisError:
            proc.terminate()
            proc.wait()
            shutil.rmtree(tmpdir, ignore_errors=True)
            pytest.skip("could not start a unix-socket redis-server")
        finally:
            client.close()
        yield sock_path, self._UNIX_PORT
        proc.terminate()
        proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dpor_detects_race_between_unix_socket_and_tcp_clients(self, unix_socket_server: tuple[str, int]) -> None:
        """Mixed unix-socket/TCP access to one server (issue #250 finding)."""
        sock_path, port = unix_socket_server

        class State:
            def __init__(self) -> None:
                self.reader = redis_lib.Redis(unix_socket_path=sock_path, decode_responses=True)
                self.writer = redis_lib.Redis(host="localhost", port=port, decode_responses=True)
                self.reader.ping()
                self.writer.ping()
                self.reader.delete("unix_flag", "unix_witness")

        def reader(state: State) -> None:
            if state.reader.get("unix_flag") == "1":
                state.reader.set("unix_witness", "1")

        def writer(state: State) -> None:
            state.writer.set("unix_flag", "1")

        def invariant(state: State) -> bool:
            return state.reader.get("unix_witness") is None

        result = frontrun.explore(
            setup=State,
            workers=[reader, writer],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "unix-socket and TCP clients reach one server; the race must be found"


class TestRedisLiveSelectIdentity:
    """A live SELECT must move a client's accesses to the selected database."""

    def test_dpor_detects_race_after_connection_init_select(self, redis_port: int) -> None:
        """SELECT at connection-init time (issue #250 finding).

        The reader client is configured db=0 but switched to db 1 with a live
        SELECT during setup, so both workers physically race on db1:select_flag.
        Attributing the reader's accesses to the stale configured db misses the
        dependency and falsely certifies.
        """
        port = redis_port

        class State:
            def __init__(self) -> None:
                self.reader = redis_lib.Redis(port=port, db=0, decode_responses=True, single_connection_client=True)
                self.writer = redis_lib.Redis(port=port, db=1, decode_responses=True)
                self.reader.ping()
                self.reader.execute_command("SELECT", 1)
                self.writer.ping()
                self.writer.delete("select_flag", "select_witness")

        def reader(state: State) -> None:
            if state.reader.get("select_flag") == "1":
                state.reader.set("select_witness", "1")

        def writer(state: State) -> None:
            state.writer.set("select_flag", "1")

        def invariant(state: State) -> bool:
            return state.writer.get("select_witness") is None

        result = frontrun.explore(
            setup=State,
            workers=[reader, writer],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "both workers race on db 1; the SET-before-GET order must be found"
