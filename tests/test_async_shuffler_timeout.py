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
from typing import Any

import pytest

from frontrun.async_scheduler import SchedulerTimeoutError
from frontrun.async_shuffler import explore_async_random, run_with_schedule


def test_public_run_with_schedule_rejects_deadlocked_state() -> None:
    """Exact replay must not return state after the scheduler aborted.

    ``AsyncShuffler.run`` records a scheduler timeout on the runner so random
    exploration can classify it, but the public exact-schedule helper must
    surface that failure.  Returning the state after the abort presents the
    scheduler's cleanup/free-run as if the requested schedule completed.
    """

    class State:
        def __init__(self) -> None:
            self.lock_a = asyncio.Lock()
            self.lock_b = asyncio.Lock()

    async def task_ab(state: State) -> None:
        async with state.lock_a:
            await asyncio.sleep(0)
            async with state.lock_b:
                pass

    async def task_ba(state: State) -> None:
        async with state.lock_b:
            await asyncio.sleep(0)
            async with state.lock_a:
                pass

    async def replay() -> None:
        with pytest.raises(SchedulerTimeoutError, match="[Dd]eadlock"):
            await run_with_schedule(
                [0, 1] * 20,
                State,
                [task_ab, task_ba],
                timeout=1.0,
                deadlock_timeout=0.05,
            )

    asyncio.run(replay())


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


def test_lock_deadlock_with_no_task_in_pause_is_detected() -> None:
    """A deadlock where every task is blocked on a real asyncio lock — none of
    them inside the scheduler's pause() — must still be detected as a deadlock.

    The pause-path detection (all-waiting check + per-wait deadlock_timeout)
    only sees tasks blocked *inside* pause().  With the alternating schedule
    below, both tasks are granted straight into their second lock acquisition
    and block on stock ``asyncio.Lock`` futures instead: no task ever waits in
    pause(), so no pause timeout fires, ``scheduler.had_error`` stays False,
    and the run surfaces as a bare wall-clock timeout — indistinguishable from
    a slow-but-correct run, which the exploration loop rightly skips as
    inconclusive.  A genuine lock-order-inversion deadlock is then silently
    dropped (whether ``test_deadlock_is_surfaced_not_false_invariant`` above
    catches this depends on which schedule flavors the seed happens to
    generate — Python-version-dependent).
    """
    from frontrun.async_shuffler import _patch_async_runtime, _run_with_schedule_status

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

    async def run() -> tuple[Any, Any]:
        # Alternate grants so each task takes its first lock, then both block
        # acquiring the other's — a deadlock formed entirely outside pause().
        with _patch_async_runtime(detect_sql=False):
            return await _run_with_schedule_status(
                [0, 1] * 20, State, [task_ab, task_ba], timeout=2.0, deadlock_timeout=0.5
            )

    state, runner = asyncio.run(run())
    assert state.done == 0  # genuinely deadlocked: neither task finished
    assert runner.timed_out
    assert runner.scheduler.had_error, "lock-blocked deadlock must set the scheduler error, not just time out"
    assert "deadlock" in str(runner.scheduler._error).lower()


def test_slow_but_correct_run_cannot_return_passing_proof() -> None:
    """A slow-but-correct run that merely exceeds timeout_per_run must NOT be
    reported as a deadlock counterexample.

    The sync bytecode explorer treats a plain timeout as inconclusive (skips
    it) and only surfaces a genuinely-detected deadlock.  explore_async_random
    conflated the two, reporting *any* run over timeout_per_run as
    property_holds=False "Deadlock detected" — a false counterexample for
    correct-but-slow code.  Here the tasks are pure CPU work (no await), so the
    scheduler never detects a deadlock; the run only exceeds the wall-clock
    timeout.
    """
    import time

    class State:
        def __init__(self) -> None:
            self.value = 0

    async def slow_task(state: State) -> None:
        # Correct, no deadlock: a pure CPU stretch that exceeds the tiny
        # per-run timeout.  The scheduler never detects a deadlock.
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            pass
        state.value += 1

    result = asyncio.run(
        explore_async_random(
            setup=State,
            tasks=[slow_task, slow_task],
            invariant=lambda s: True,  # can never be violated
            max_attempts=2,
            timeout_per_run=0.1,
            deadlock_timeout=5.0,
            seed=1,
        )
    )

    assert not result.property_holds
    assert result.counterexample is None
    assert result.explanation is not None
    assert "inconclusive" in result.explanation.lower()
    assert "deadlock" not in result.explanation.lower()


def test_slow_unmanaged_await_is_not_a_false_deadlock() -> None:
    """No-progress polling cannot prove an ordinary wall timer is deadlocked."""

    async def worker(state: dict[str, bool]) -> None:
        await asyncio.sleep(0.05)
        state["done"] = True

    result = asyncio.run(
        explore_async_random(
            setup=lambda: {"done": False},
            tasks=[worker],
            invariant=lambda state: state["done"],
            max_attempts=1,
            max_ops=10,
            timeout_per_run=0.2,
            deadlock_timeout=0.01,
            patch_sleep=False,
            seed=1,
        )
    )

    assert result.property_holds, result.explanation


def test_max_ops_truncation_cannot_return_passing_proof() -> None:
    """If every sampled schedule is truncated, no invariant was checked."""

    async def long_task(_state: object) -> None:
        for _ in range(20):
            await asyncio.sleep(0)

    result = asyncio.run(
        explore_async_random(
            setup=object,
            tasks=[long_task],
            invariant=lambda _state: True,
            max_attempts=2,
            max_ops=1,
            timeout_per_run=1.0,
            deadlock_timeout=1.0,
            seed=1,
        )
    )

    assert not result.property_holds
    assert result.counterexample is None
    assert result.explanation is not None
    assert "inconclusive" in result.explanation.lower()
    assert "max_ops" in result.explanation


def test_uncaught_wait_for_timeout_is_task_crash_not_deadlock() -> None:
    class State:
        pass

    async def worker(state: State) -> None:
        await asyncio.wait_for(asyncio.Event().wait(), timeout=0.01)

    result = asyncio.run(
        explore_async_random(
            setup=State,
            tasks=[worker],
            invariant=lambda s: True,
            max_attempts=1,
            timeout_per_run=1.0,
            deadlock_timeout=0.2,
            clock="virtual",
            seed=1,
        )
    )

    assert not result.property_holds
    assert result.explanation is not None
    assert "Task crash" in result.explanation
    assert "TimeoutError" in result.explanation
    assert "Deadlock detected" not in result.explanation


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
