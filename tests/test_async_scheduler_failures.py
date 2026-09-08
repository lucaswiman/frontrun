"""Async scheduler failures must surface without certifying partial executions."""

from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest

import frontrun
import frontrun.async_shuffler as async_shuffler
from frontrun.async_scheduler import InterleavedLoop, SchedulerTimeoutError
from frontrun.async_shuffler import explore_async_random, run_with_schedule


class _LockInversion:
    def __init__(self) -> None:
        self.lock_a = asyncio.Lock()
        self.lock_b = asyncio.Lock()
        self.done = 0

    async def take_ab(self) -> None:
        async with self.lock_a:
            await asyncio.sleep(0)
            async with self.lock_b:
                self.done += 1

    async def take_ba(self) -> None:
        async with self.lock_b:
            await asyncio.sleep(0)
            async with self.lock_a:
                self.done += 1


def test_public_run_with_schedule_rejects_deadlocked_state() -> None:
    """Exact replay must not return state after the scheduler aborted.

    ``AsyncShuffler.run`` records a scheduler timeout on the runner so random
    exploration can classify it, but the public exact-schedule helper must
    surface that failure.  Returning the state after the abort presents the
    scheduler's cleanup/free-run as if the requested schedule completed.
    """

    async def replay() -> None:
        with pytest.raises(SchedulerTimeoutError, match="[Dd]eadlock"):
            await run_with_schedule(
                [0, 1] * 20,
                _LockInversion,
                [_LockInversion.take_ab, _LockInversion.take_ba],
                timeout=1.0,
                deadlock_timeout=0.05,
            )

    asyncio.run(replay())


def test_deadlock_on_unmanaged_locks_is_not_scored_as_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lock-order-inversion deadlock on *unmanaged* primitives must never be
    scored as a pass — but the random shuffler reports it as inconclusive, not
    a fabricated counterexample.

    Two tasks acquire two stock ``asyncio.Lock`` objects in opposite order and
    deadlock, blocking *outside* ``pause()``.  The invariant below is trivially
    satisfied by the partial (cancelled) state, so the essential guarantee is
    that the run is NOT scored ``property_holds=True``.

    The shuffler cannot *constructively* prove this deadlock: it is invisible to
    the sound all-waiting-in-pause() detector, and a bounded no-progress window
    cannot distinguish a permanent deadlock from a slow-but-completing unmanaged
    await (the exact ambiguity that made the pause watchdog fabricate a
    "Deadlock detected" counterexample for correct-but-slow code — see
    ``test_peer_waiting_on_slow_unmanaged_await_is_not_a_false_deadlock``).  So
    ``explore_async_random`` (``detect_external_deadlock=False``) honestly
    reports *inconclusive* rather than a false FAIL.  Sound deadlock detection
    on unmanaged primitives is the DPOR path's job, not the random shuffler's.
    """

    # Exercise one known deadlocking schedule instead of relying on a random
    # sample to hit it repeatedly and paying timeout_per_run for every hit.
    monkeypatch.setattr(async_shuffler, "random_round_robin_schedule", lambda *_args: [0, 1] * 20)
    result = asyncio.run(
        explore_async_random(
            setup=_LockInversion,
            tasks=[_LockInversion.take_ab, _LockInversion.take_ba],
            # Invariant a partial/cancelled run trivially satisfies.
            invariant=lambda s: s.done <= 2,
            max_attempts=1,
            timeout_per_run=0.2,
            deadlock_timeout=0.05,
            seed=1234,
        )
    )

    # The core soundness guarantee: a deadlocked run is never a false pass.
    assert result.property_holds is not True, "deadlock must not be scored as a pass"
    # And the shuffler must not fabricate a deadlock it cannot prove: the honest
    # verdict is inconclusive (property_holds=None, no counterexample).
    assert result.property_holds is None, result.explanation
    assert result.counterexample is None
    assert result.explanation is not None
    assert "inconclusive" in result.explanation.lower(), result.explanation


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

    async def run() -> tuple[Any, Any]:
        # Alternate grants so each task takes its first lock, then both block
        # acquiring the other's — a deadlock formed entirely outside pause().
        with _patch_async_runtime(detect_sql=False):
            return await _run_with_schedule_status(
                [0, 1] * 20,
                _LockInversion,
                [_LockInversion.take_ab, _LockInversion.take_ba],
                timeout=2.0,
                deadlock_timeout=0.5,
            )

    state, runner = asyncio.run(run())
    assert state.done == 0  # genuinely deadlocked: neither task finished
    assert runner.timed_out
    assert runner.scheduler.had_error, "lock-blocked deadlock must set the scheduler error, not just time out"
    assert "deadlock" in str(runner.scheduler._error).lower()


def test_slow_but_correct_run_cannot_return_passing_proof(recwarn: pytest.WarningsRecorder) -> None:
    """A slow-but-correct run that merely exceeds timeout_per_run must NOT be
    reported as a deadlock counterexample.

    The sync bytecode explorer treats a plain timeout as inconclusive (skips
    it) and only surfaces a genuinely-detected deadlock.  explore_async_random
    conflated the two, reporting *any* run over timeout_per_run as
    property_holds=False "Deadlock detected" — a false counterexample for
    correct-but-slow code. Here the tasks wait on an unmanaged wall timer that
    exceeds the per-run timeout.
    """

    class State:
        def __init__(self) -> None:
            self.value = 0

    async def slow_task(state: State) -> None:
        await asyncio.sleep(0.5)
        state.value += 1

    result = asyncio.run(
        explore_async_random(
            setup=State,
            tasks=[slow_task, slow_task],
            invariant=lambda s: True,  # can never be violated
            max_attempts=2,
            timeout_per_run=0.1,
            deadlock_timeout=5.0,
            patch_sleep=False,
            seed=1,
        )
    )

    assert not result.property_holds
    assert result.counterexample is None
    assert result.explanation is not None
    assert "inconclusive" in result.explanation.lower()
    assert "deadlock" not in result.explanation.lower()
    gc.collect()
    assert not [warning for warning in recwarn if "was never awaited" in str(warning.message)]


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


def test_peer_waiting_on_slow_unmanaged_await_is_not_a_false_deadlock() -> None:
    """A task blocked in pause() waiting for a peer that is merely slow on an
    unmanaged awaitable must NOT be reported as a deadlock counterexample.

    ``test_slow_unmanaged_await_is_not_a_false_deadlock`` covers the single-task
    case, where no task ever waits in pause().  With two tasks the failure mode
    is different: ``fast_task`` reaches ``pause()`` where the schedule wants
    ``slow_task`` first, but ``slow_task`` is off on an *unmanaged* Future that
    resolves via a real timer AFTER ``deadlock_timeout`` yet well BEFORE
    ``timeout_per_run``.  The pause watchdog (``_handle_timeout``) fired and set
    ``scheduler._error`` unconditionally — bypassing ``detect_external_deadlock``
    — so the run was scored ``property_holds=False`` "Deadlock detected".  That
    is a fabricated counterexample for a slow-but-correct run: a plain
    unmanaged stall must reach the overall timeout (inconclusive) or complete,
    never become a fail.
    """

    class State:
        def __init__(self) -> None:
            self.done: list[str] = []

    async def slow_task(state: State) -> None:
        await asyncio.sleep(0)  # a controlled pause point
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        # Resolve after 0.3s real time: > deadlock_timeout (0.05) but well
        # under timeout_per_run (3.0).  Unmanaged by the scheduler.
        loop.call_later(0.3, lambda: fut.done() or fut.set_result(None))
        await fut
        await asyncio.sleep(0)
        state.done.append("slow")

    async def fast_task(state: State) -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state.done.append("fast")

    result = asyncio.run(
        explore_async_random(
            setup=State,
            tasks=[slow_task, fast_task],
            invariant=lambda s: True,  # can never be violated
            max_attempts=4,
            max_ops=8,
            timeout_per_run=3.0,
            deadlock_timeout=0.05,
            patch_sleep=False,
            seed=1234,
        )
    )

    assert result.property_holds is not False, result.explanation
    if result.explanation is not None:
        assert "deadlock" not in result.explanation.lower(), result.explanation


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


class _AllWaitingLoop(InterleavedLoop):
    """A loop whose tasks all block in pause(), forcing all-waiting deadlock."""

    def should_proceed(self, task_id, marker=None):  # type: ignore[no-untyped-def]
        # Never let anyone proceed → every task blocks in pause().
        return False


def test_run_all_propagates_scheduler_error() -> None:
    """``run_all`` must raise the scheduler's ``_error`` after draining tasks."""

    async def worker() -> None:
        await loop.pause("w")

    loop = _AllWaitingLoop(deadlock_timeout=0.2)

    async def main() -> None:
        await loop.run_all([worker, worker], timeout=5.0)

    with pytest.raises(SchedulerTimeoutError) as excinfo:
        asyncio.run(main())

    # The surfaced error must be the scheduler's own deadlock error, not a
    # generic "tasks did not complete" overall-timeout.
    assert loop._error is not None
    assert excinfo.value is loop._error or str(loop._error) in str(excinfo.value)


async def _self_cancel(state: list[str]) -> None:
    state.append("partial")
    raise asyncio.CancelledError("worker cancelled itself")


@pytest.mark.parametrize("strategy", ["dpor", "random"])
def test_self_cancelled_worker_is_not_a_successful_exploration(strategy: str) -> None:
    """A cancelled worker leaves partial state and cannot prove the property."""
    options: dict[str, object]
    if strategy == "dpor":
        options = {"max_executions": 1, "reproduce_on_failure": 0, "detect_io": False}
    else:
        options = {"max_attempts": 1, "max_ops": 2, "seed": 1}

    result = asyncio.run(
        frontrun.explore(
            setup=list,
            workers=[_self_cancel],
            invariant=lambda state: state == ["partial"],
            strategy=strategy,
            **options,
        )
    )

    assert not result.property_holds
    assert result.explanation is not None
    assert "cancel" in result.explanation.lower()


def test_run_with_schedule_propagates_worker_cancellation() -> None:
    """Exact replay must not return state from a cancelled worker."""

    async def replay() -> None:
        with pytest.raises(asyncio.CancelledError, match="worker cancelled itself"):
            await run_with_schedule([0, 0], list, [_self_cancel])

    asyncio.run(replay())
