"""Regression tests for timeout/deadlock surfacing in explore_async_random (F6).

``AsyncShuffler.run`` swallowed the ``TimeoutError`` that ``run_all`` raises
after cancelling tasks mid-flight, so ``run_with_schedule`` returned a
partially-mutated state and ``explore_async_random`` evaluated the invariant
on it.  A hung/deadlocked run could thus become a false "invariant violation"
counterexample, or a real deadlock could be silently dropped.

A deadlock or scheduler timeout must instead be surfaced as its own outcome,
and the invariant must NOT be evaluated against a cancelled/timed-out run's
state.
"""

from __future__ import annotations

import asyncio

import pytest

from frontrun.async_shuffler import explore_async_random


def test_deadlock_is_surfaced_not_false_invariant() -> None:
    """A lock-order-inversion deadlock must be reported as a deadlock.

    Two tasks acquire two locks in opposite order.  Some interleavings
    deadlock.  The invariant below would be *satisfied* by a partial
    (cancelled) run where neither task finished — so swallowing the timeout
    and checking the invariant would wrongly report property_holds=True and
    silently drop the deadlock.
    """

    class State:
        def __init__(self) -> None:
            self.lock_a = asyncio.Lock()
            self.lock_b = asyncio.Lock()
            self.done = 0

    async def task_ab(state: State) -> None:
        async with state.lock_a:
            await asyncio.sleep(0)
            async with state.lock_b:
                state.done += 1

    async def task_ba(state: State) -> None:
        async with state.lock_b:
            await asyncio.sleep(0)
            async with state.lock_a:
                state.done += 1

    result = asyncio.run(
        explore_async_random(
            setup=State,
            tasks=[task_ab, task_ba],
            # Invariant a partial/cancelled run trivially satisfies.
            invariant=lambda s: s.done <= 2,
            max_attempts=40,
            timeout_per_run=2.0,
            deadlock_timeout=0.5,
            seed=1234,
        )
    )

    assert not result.property_holds, "deadlock should be surfaced, not scored as a pass"
    assert result.explanation is not None
    assert "deadlock" in result.explanation.lower(), result.explanation


def test_detect_sql_reports_table_accesses() -> None:
    """F8: detect_sql=True in the random async shuffler must actually report.

    Previously AwaitScheduler._setup_task_context installed no IO reporter or
    DPOR context, so _report_sql_access always returned False and the
    documented table-conflict reporting never happened — a silent no-op.
    """
    aiosqlite = pytest.importorskip("aiosqlite")

    reported: list[tuple[str, str]] = []

    class State:
        def __init__(self) -> None:
            self.reported = reported

    async def worker(state: State) -> None:
        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
            await conn.execute("INSERT INTO t VALUES (1, 0)")
            from frontrun._io_detection import get_io_reporter

            # The shuffler must have installed a reporter for this task.
            reporter = get_io_reporter()
            if reporter is not None:
                state.reported.append(("reporter-present", "ok"))
            await conn.execute("SELECT * FROM t WHERE id = 1")

    import asyncio

    asyncio.run(
        explore_async_random(
            setup=State,
            tasks=[worker, worker],
            invariant=lambda s: True,
            max_attempts=2,
            timeout_per_run=3.0,
            detect_sql=True,
            seed=7,
        )
    )

    assert ("reporter-present", "ok") in reported, (
        "detect_sql=True must install an IO reporter so SQL accesses are reported"
    )
