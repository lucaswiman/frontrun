"""Integration tests for Celery-style task race detection with DPOR.

Celery tasks that race on shared state (Redis, SQL) don't need Celery-specific
support -- frontrun's existing Redis key-level and SQL cursor interception
detects the actual races.  These tests demonstrate this by running task-like
functions as threads under frontrun's DPOR explorer.

Three scenarios are covered:

1. **Redis backend race** -- check-then-set on a Redis key (simulating
   result backend races between concurrent tasks).
2. **SQL backend race** -- read-modify-write on a SQLite row (simulating
   a counter increment task with a lost update).
3. **Task result overwrite** -- two tasks writing results to the same Redis
   key (write-write race on Celery's result backend).

Requirements::

    redis-server must be installed (apt-get install redis-server)
    pip install redis

Running::

    make test-integration-3.14
    # or:
    frontrun .venv-3.14/bin/pytest tests/test_integration_celery.py -v
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

from frontrun import explore

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 1. Redis backend race: check-then-set (simulating result backend)
# ---------------------------------------------------------------------------


class TestCeleryRedisBackendRace:
    """Two 'Celery tasks' do check-then-set on a Redis key.

    Simulates a result backend race where two tasks check whether a
    result exists, then write their own result.  Without synchronization
    one task's result can be silently overwritten.
    """

    def test_dpor_detects_check_then_set_race(self, redis_port: int) -> None:
        """DPOR should detect the check-then-set race on a shared result key."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("celery-result-abc123", "celery-result-write-count")
                r.set("celery-result-write-count", "0")
                r.close()

        def task_store_result(state: State, task_id: int = 0) -> None:
            """Simulates a Celery task storing its result with a check-then-set pattern."""
            r = redis_lib.Redis(port=port, decode_responses=True)
            existing = r.get("celery-result-abc123")
            if existing is None:
                r.set("celery-result-abc123", f"result-from-task-{task_id}")
                r.incr("celery-result-write-count")
            r.close()

        def task_a(state: State) -> None:
            task_store_result(state, task_id=0)

        def task_b(state: State) -> None:
            task_store_result(state, task_id=1)

        def invariant(state: State) -> bool:
            """At most one task should have written a result."""
            r = redis_lib.Redis(port=port, decode_responses=True)
            write_count = int(r.get("celery-result-write-count"))  # type: ignore[arg-type]
            r.close()
            return write_count <= 1

        result = explore(
            setup=State,
            workers=[task_a, task_b],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect check-then-set race on Redis result key"
        assert result.num_explored >= 2, "DPOR must explore multiple interleavings"


# ---------------------------------------------------------------------------
# 2. SQL backend race: read-modify-write (simulating counter increment)
# ---------------------------------------------------------------------------


class TestCelerySQLBackendRace:
    """Two 'Celery tasks' do read-modify-write on a SQLite counter row.

    Simulates a common Celery pattern where tasks increment a shared counter
    in a database.  Without proper locking, the classic lost-update race
    occurs: both tasks read the same value, increment, and write back,
    losing one increment.
    """

    def test_dpor_detects_lost_update(self) -> None:
        """DPOR should detect the lost-update race on a SQL counter."""
        db_dir = tempfile.mkdtemp(prefix="frontrun_celery_test_")
        db_path = os.path.join(db_dir, "celery_test.db")

        class State:
            def __init__(self) -> None:
                conn = sqlite3.connect(db_path)
                conn.execute("CREATE TABLE IF NOT EXISTS task_counter (id INTEGER PRIMARY KEY, value INTEGER)")
                conn.execute("DELETE FROM task_counter")
                conn.execute("INSERT INTO task_counter (id, value) VALUES (1, 0)")
                conn.commit()
                conn.close()

        def increment_task(state: State) -> None:
            """Simulates a Celery task that increments a shared counter."""
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT value FROM task_counter WHERE id = 1")
            current = cursor.fetchone()[0]
            conn.execute("UPDATE task_counter SET value = ? WHERE id = 1", (current + 1,))
            conn.commit()
            conn.close()

        def invariant(state: State) -> bool:
            """After two increments, counter should be 2."""
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT value FROM task_counter WHERE id = 1")
            value = cursor.fetchone()[0]
            conn.close()
            return value == 2

        result = explore(
            setup=State,
            workers=[increment_task, increment_task],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect lost-update race on SQL counter"
        assert result.num_explored >= 2, "DPOR must explore multiple interleavings"

    def test_locked_increment_is_safe(self) -> None:
        """Counter protected by a Python lock is safe."""
        db_dir = tempfile.mkdtemp(prefix="frontrun_celery_test_")
        db_path = os.path.join(db_dir, "celery_test.db")

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                conn = sqlite3.connect(db_path)
                conn.execute("CREATE TABLE IF NOT EXISTS task_counter (id INTEGER PRIMARY KEY, value INTEGER)")
                conn.execute("DELETE FROM task_counter")
                conn.execute("INSERT INTO task_counter (id, value) VALUES (1, 0)")
                conn.commit()
                conn.close()

        def increment_task(state: State) -> None:
            """Simulates a Celery task that increments a shared counter, with locking."""
            with state.lock:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT value FROM task_counter WHERE id = 1")
                current = cursor.fetchone()[0]
                conn.execute("UPDATE task_counter SET value = ? WHERE id = 1", (current + 1,))
                conn.commit()
                conn.close()

        def invariant(state: State) -> bool:
            """After two increments, counter should be 2."""
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT value FROM task_counter WHERE id = 1")
            value = cursor.fetchone()[0]
            conn.close()
            return value == 2

        result = explore(
            setup=State,
            workers=[increment_task, increment_task],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 3. Task result overwrite: write-write race on Redis
# ---------------------------------------------------------------------------


class TestCeleryResultOverwrite:
    """Two tasks writing results to the same Redis key (write-write race).

    Simulates the scenario where two Celery task executions (e.g. retries
    or duplicate deliveries) both write their result to the same key.
    The final value depends on interleaving -- a write-write race.
    """

    def test_dpor_detects_write_write_race(self, redis_port: int) -> None:
        """DPOR should detect the write-write race on a shared result key."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("celery-task-result-xyz")
                r.close()
                self.written_by: list[int | None] = [None, None]

        def store_result(state: State, task_id: int = 0) -> None:
            """Simulates a Celery task writing its result to the backend."""
            r = redis_lib.Redis(port=port, decode_responses=True)
            r.set("celery-task-result-xyz", f"result-{task_id}")
            state.written_by[task_id] = task_id
            r.close()

        def task_a(state: State) -> None:
            store_result(state, task_id=0)

        def task_b(state: State) -> None:
            store_result(state, task_id=1)

        def invariant(state: State) -> bool:
            """The stored result should be from task_b (last declared thread).

            In sequential execution [task_a, task_b], task_b writes last
            so the value is 'result-1'.  DPOR explores the reversed
            interleaving where task_a writes after task_b, producing
            'result-0' instead -- a write-write race.
            """
            r = redis_lib.Redis(port=port, decode_responses=True)
            val = r.get("celery-task-result-xyz")
            r.close()
            return val == "result-1"

        result = explore(
            setup=State,
            workers=[task_a, task_b],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect write-write race on Redis result key"
        assert result.num_explored >= 2, "DPOR must explore multiple interleavings"
