"""Tests for DPOR detection of Redis-based race conditions.

These tests connect to a real Redis server and exercise DPOR's ability
to detect application-level race conditions that manifest through
network I/O (TCP socket calls to Redis).

The race conditions are in the APPLICATION code, not in Redis itself.
Individual Redis commands (GET, SET, EXISTS) are atomic, but compound
read-modify-write sequences are not — two threads can read a stale
value and overwrite each other's updates.

DPOR detects these races through the LD_PRELOAD I/O interception
library, which intercepts libc-level send/recv calls to the Redis
socket and reports them as I/O accesses to the DPOR engine.

Requirements::

    redis-server must be installed (apt-get install redis-server)
    pip install redis

Running::

    make test-integration-3.10
    # or directly:
    frontrun .venv-3.10/bin/pytest tests/test_integration_redis.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

from frontrun.dpor import explore_dpor

# Use a non-default port to avoid colliding with a user's Redis.
# CI can override via REDIS_PORT env var (e.g. when using a service container).
_REDIS_PORT = int(os.environ.get("REDIS_PORT", "16399"))


@pytest.fixture(scope="module")
def redis_port():
    """Provide a Redis port, starting a server if one isn't already listening."""
    # Try connecting to an existing Redis (e.g. CI service container).
    r = redis_lib.Redis(port=_REDIS_PORT)
    try:
        r.ping()
        r.close()
        yield _REDIS_PORT
        return
    except redis_lib.ConnectionError:
        r.close()

    # No Redis found — start our own.
    if not shutil.which("redis-server"):
        pytest.skip("redis-server not installed and no Redis listening on port " + str(_REDIS_PORT))

    proc = subprocess.Popen(
        [
            "redis-server",
            "--port",
            str(_REDIS_PORT),
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
    time.sleep(0.5)
    r = redis_lib.Redis(port=_REDIS_PORT)
    try:
        r.ping()
    except redis_lib.ConnectionError:
        proc.kill()
        pytest.skip("Could not start redis-server")
    finally:
        r.close()
    yield _REDIS_PORT
    proc.terminate()
    proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# 1. Lost-update counter
# ---------------------------------------------------------------------------
# Two threads each GET a counter, increment in Python, and SET back.
# The compound GET→increment→SET is not atomic, so one increment can
# be lost when both threads read before either writes.


class TestRedisCounterRace:
    """Lost-update race on a Redis counter."""

    def test_dpor_detects_lost_update(self, redis_port: int) -> None:
        """DPOR should detect the lost-update race via I/O interdiction."""
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

        result = explore_dpor(
            setup=State,
            threads=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect lost-update on Redis counter"
        assert result.explanation is not None

    def test_locked_counter_is_safe(self, redis_port: int) -> None:
        """Counter protected by a Python lock across the full read-modify-write."""
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

        result = explore_dpor(
            setup=State,
            threads=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation

    def test_naive_threading_race_rate(self, redis_port: int) -> None:
        """Verify the race manifests intermittently with plain threads.

        Threads start with a random 0-5ms offset to model realistic
        request arrival timing (like orm_race.py's demo_naive_threading).
        """
        import random
        import time

        port = redis_port
        trials = 500
        failures = 0
        rng = random.Random(42)

        for _ in range(trials):
            r = redis_lib.Redis(port=port, decode_responses=True)
            r.set("counter", "0")
            r.close()

            def increment() -> None:
                conn = redis_lib.Redis(port=port, decode_responses=True)
                val = int(conn.get("counter"))  # type: ignore[arg-type]
                conn.set("counter", str(val + 1))
                conn.close()

            t1 = threading.Thread(target=increment)
            t2 = threading.Thread(target=increment)
            t1.start()
            # Random offset models realistic request arrival timing.
            time.sleep(rng.uniform(0, 0.005))
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

            r = redis_lib.Redis(port=port, decode_responses=True)
            if int(r.get("counter")) != 2:  # type: ignore[arg-type]
                failures += 1
            r.close()

        rate = failures / trials * 100
        # The race should be observable but not dominant
        assert failures > 0, f"Race never triggered in {trials} trials — test may not be realistic"


# ---------------------------------------------------------------------------
# 2. Check-then-act: double initialization
# ---------------------------------------------------------------------------
# Two threads check if a key exists, and if not, initialize it and bump
# an init counter.  Both can see the key missing and both initialize,
# causing init_count to be 2 instead of 1.


class TestRedisCheckThenAct:
    """TOCTOU race: double initialization of a Redis key."""

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

        result = explore_dpor(
            setup=State,
            threads=[maybe_init, maybe_init],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect double-initialization race"
        assert result.explanation is not None

    def test_setnx_prevents_double_init(self, redis_port: int) -> None:
        """Using SET NX (atomic check-and-set) prevents double init."""
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
                # Atomic check-and-set: only one thread succeeds
                was_set = r.set("resource", "initialized", nx=True)
                if was_set:
                    count = int(r.get("init_count"))  # type: ignore[arg-type]
                    r.set("init_count", str(count + 1))
                r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("init_count"))  # type: ignore[arg-type]
            r.close()
            return result == 1

        result = explore_dpor(
            setup=State,
            threads=[maybe_init, maybe_init],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 3. Inventory / double-spend
# ---------------------------------------------------------------------------
# Two threads check if there is enough stock, then decrement.  Both
# can see sufficient stock and both buy, driving the count negative.


class TestRedisInventoryRace:
    """Double-spend race on a Redis-backed inventory counter."""

    def test_dpor_detects_oversell(self, redis_port: int) -> None:
        """DPOR should detect the oversell race where stock goes negative."""
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
            # Only 1 item was available — at most 1 should be sold
            return stock >= 0 and sold <= 1

        result = explore_dpor(
            setup=State,
            threads=[buy, buy],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect oversell race"
        assert result.explanation is not None

    def test_locked_buy_is_safe(self, redis_port: int) -> None:
        """Inventory buy protected by a Python lock."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("stock", "1")
                r.set("sold", "0")
                r.close()

        def buy(state: State) -> None:
            with state.lock:
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

        result = explore_dpor(
            setup=State,
            threads=[buy, buy],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation

    def test_naive_threading_race_rate(self, redis_port: int) -> None:
        """Verify the oversell race manifests intermittently with plain threads."""
        import random
        import time

        port = redis_port
        trials = 500
        failures = 0
        rng = random.Random(42)

        for _ in range(trials):
            r = redis_lib.Redis(port=port, decode_responses=True)
            r.set("stock", "1")
            r.set("sold", "0")
            r.close()

            def buy() -> None:
                conn = redis_lib.Redis(port=port, decode_responses=True)
                stock = int(conn.get("stock"))  # type: ignore[arg-type]
                if stock > 0:
                    conn.set("stock", str(stock - 1))
                    sold = int(conn.get("sold"))  # type: ignore[arg-type]
                    conn.set("sold", str(sold + 1))
                conn.close()

            t1 = threading.Thread(target=buy)
            t2 = threading.Thread(target=buy)
            t1.start()
            time.sleep(rng.uniform(0, 0.005))
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

            r = redis_lib.Redis(port=port, decode_responses=True)
            stock = int(r.get("stock"))  # type: ignore[arg-type]
            sold = int(r.get("sold"))  # type: ignore[arg-type]
            r.close()
            if stock < 0 or sold > 1:
                failures += 1

        rate = failures / trials * 100
        assert failures > 0, f"Race never triggered in {trials} trials"


# ---------------------------------------------------------------------------
# 4. Transfer race — conservation of total balance
# ---------------------------------------------------------------------------
# Two threads transfer different amounts from account A to account B.
# Both read both balances, compute new values, and write back.  The
# second writer overwrites the first's update, violating conservation.


class TestRedisTransferRace:
    """Lost-update race on concurrent Redis-backed balance transfers."""

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
            # Total balance must be conserved: 100 + 100 = 200
            return bal_a + bal_b == 200

        result = explore_dpor(
            setup=State,
            threads=[transfer_a_to_b, transfer_a_to_b_also],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect transfer lost-update"
        assert result.explanation is not None

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

        result = explore_dpor(
            setup=State,
            threads=[
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
