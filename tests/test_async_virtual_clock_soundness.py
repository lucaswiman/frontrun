"""Async virtual-clock soundness regressions from the pre-release audit.

Each test pins a way the async virtual clock could produce a wrong result:
an autojump racing an in-flight wake (clock overshoots on a *successful*
wait), timeout-guarded lock cycles scored as deadlocks (the classic
timeout-avoidance pattern), ``Timeout.reschedule()`` double-counting virtual
time already elapsed in the block, ``CancelledError`` subclasses escaping the
expired-timeout conversion, and (simulating 3.14t contextvar inheritance) the
patched wrappers following a scheduler into a foreign thread's event loop.
"""

from __future__ import annotations

import asyncio
import math
import sys
import threading
import time

import pytest

import frontrun


class _WaitForState:
    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self.fut: asyncio.Future[int] | None = None
        self.value: int | None = None
        self.elapsed: float | None = None


class _BareFutureTimeoutState:
    def __init__(self) -> None:
        self.timed_out = False
        self.elapsed: float | None = None


def test_async_wait_for_coroutine_awaiting_bare_future_autojumps() -> None:
    async def worker(state: _BareFutureTimeoutState) -> None:
        future = asyncio.get_running_loop().create_future()

        async def inner() -> None:
            await future

        started = time.monotonic()
        try:
            await asyncio.wait_for(inner(), 1.0)
        except TimeoutError:
            state.timed_out = True
            state.elapsed = time.monotonic() - started

    result = asyncio.run(
        frontrun.explore(
            setup=_BareFutureTimeoutState,
            workers=[worker],
            invariant=lambda state: state.timed_out and state.elapsed == pytest.approx(1.0),
            strategy="dpor",
            clock="virtual",
            max_executions=1,
            timeout_per_run=0.5,
            deadlock_timeout=0.1,
            reproduce_on_failure=0,
            detect_io=False,
        )
    )

    assert result.property_holds, result.explanation


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires 3.11+")
def test_async_timeout_around_bare_future_autojumps() -> None:
    async def worker(state: _BareFutureTimeoutState) -> None:
        future = asyncio.get_running_loop().create_future()
        started = time.monotonic()
        try:
            async with asyncio.timeout(1.0):
                await future
        except TimeoutError:
            state.timed_out = True
            state.elapsed = time.monotonic() - started

    result = asyncio.run(
        frontrun.explore(
            setup=_BareFutureTimeoutState,
            workers=[worker],
            invariant=lambda state: state.timed_out and state.elapsed == pytest.approx(1.0),
            strategy="dpor",
            clock="virtual",
            max_executions=1,
            timeout_per_run=0.5,
            deadlock_timeout=0.1,
            reproduce_on_failure=0,
            detect_io=False,
        )
    )

    assert result.property_holds, result.explanation


def test_async_wait_for_success_does_not_autojump_to_full_timeout() -> None:
    # The waiter parks (engine-blocked) in wait_for on a bare future; the
    # setter resolves it and finishes. The waiter's wake callback is then
    # still in the loop's ready queue while nothing is engine-runnable — the
    # autojump must NOT fire the 10s timeout deadline before that wake runs.
    # A successful instant wait observing the full timeout in virtual time is
    # a false counterexample (and offsets every later deadline in the run).
    async def waiter(s: _WaitForState) -> None:
        s.fut = asyncio.get_running_loop().create_future()
        s.ready.set()
        start = time.monotonic()
        s.value = await asyncio.wait_for(s.fut, timeout=10.0)
        s.elapsed = time.monotonic() - start

    async def setter(s: _WaitForState) -> None:
        await s.ready.wait()
        fut = s.fut
        assert fut is not None
        fut.set_result(42)

    result = asyncio.run(
        frontrun.explore(
            setup=_WaitForState,
            workers=[waiter, setter],
            invariant=lambda s: s.value == 42 and s.elapsed is not None and s.elapsed < 5.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


class _SequentialTimedWaitState:
    def __init__(self) -> None:
        self.fut1: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.timed_out = False


def test_async_second_timed_wait_after_completed_wait_for_autojumps() -> None:
    # The waiter's first wait_for is resolved by the setter, which then
    # finishes: nothing is engine-runnable while the 5s deadline is still
    # registered, so a deferred autojump is armed. The waiter's recovered
    # wake takes a real turn (invalidating that jump) and re-parks in a
    # *second* timed wait without yielding to the loop. The stale jump must
    # hand the owed clock advance to a fresh one for the 1s deadline —
    # dropping it stalls a correct program to timeout_per_run wall time and
    # reports an inconclusive failure.
    async def waiter(s: _SequentialTimedWaitState) -> None:
        try:
            await asyncio.wait_for(s.fut1, timeout=5.0)
        except TimeoutError:
            pass
        try:
            await asyncio.wait_for(asyncio.get_running_loop().create_future(), timeout=1.0)
        except TimeoutError:
            s.timed_out = True

    async def setter(s: _SequentialTimedWaitState) -> None:
        if not s.fut1.done():
            s.fut1.set_result(None)

    started = time.monotonic()
    result = asyncio.run(
        frontrun.explore(
            setup=_SequentialTimedWaitState,
            workers=[waiter, setter],
            invariant=lambda s: s.timed_out,
            strategy="dpor",
            clock="virtual",
            timeout_per_run=5.0,
            deadlock_timeout=1.0,
            reproduce_on_failure=0,
        )
    )
    elapsed = time.monotonic() - started
    assert result.property_holds, result.explanation
    # Both timed waits resolve in virtual time; the stalled-schedule shape of
    # this regression burned timeout_per_run wall seconds per execution.
    assert elapsed < 4.0


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires 3.11+")
def test_async_timeout_guarded_abba_lock_cycle_is_not_a_deadlock() -> None:
    # Classic timeout-based deadlock avoidance: both tasks guard their inner
    # acquire with asyncio.timeout and recover from TimeoutError. The real
    # program provably cannot deadlock (the virtual timeout cancels the inner
    # acquire, releasing the outer lock), so reporting DeadlockError is a
    # false counterexample.  The sync side documents exactly this rule: a
    # timed acquire must not register a wait-for-graph edge.
    class State:
        def __init__(self) -> None:
            self.a = asyncio.Lock()
            self.b = asyncio.Lock()
            self.completed = 0
            self.recovered = 0

    async def w_ab(s: State) -> None:
        async with s.a:
            try:
                async with asyncio.timeout(1.0):
                    async with s.b:
                        s.completed += 1
            except TimeoutError:
                s.recovered += 1

    async def w_ba(s: State) -> None:
        async with s.b:
            try:
                async with asyncio.timeout(1.0):
                    async with s.a:
                        s.completed += 1
            except TimeoutError:
                s.recovered += 1

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[w_ab, w_ba],
            invariant=lambda s: s.completed + s.recovered == 2,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires 3.11+")
def test_async_timeout_reschedule_when_is_a_noop_after_virtual_sleep() -> None:
    # reschedule(cm.when()) is a no-op per asyncio semantics.  The loop clock
    # is pinned to real time while the block consumes *virtual* time, so a
    # naive "remaining = when - loop.time()" translation re-adds the 5 virtual
    # seconds already spent and fires the timeout at t=15 instead of t=10.
    class State:
        def __init__(self) -> None:
            self.elapsed: float | None = None

    async def worker(s: State) -> None:
        start = time.monotonic()
        try:
            async with asyncio.timeout(10.0) as cm:
                await asyncio.sleep(5.0)
                cm.reschedule(cm.when())  # no-op: same absolute deadline
                await asyncio.sleep(20.0)
        except TimeoutError:
            s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.elapsed is not None and 9.5 <= s.elapsed <= 10.5,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout_at requires 3.11+")
def test_async_timeout_at_from_cm_when_preserves_exact_virtual_deadline() -> None:
    # asyncio.timeout_at(cm.when()) hands the loop-time value straight to the
    # patched timeout_at.  The loop clock stays real while the block consumes
    # virtual time, so round-tripping that value through loop.time() smears
    # nondeterministic wall drift into the virtual deadline; the exact
    # provenance carried by cm.when() must be recovered instead, as
    # Timeout.reschedule() already does.
    class State:
        def __init__(self) -> None:
            self.elapsed: float | None = None

    async def worker(s: State) -> None:
        start = time.monotonic()
        async with asyncio.timeout(1.0) as cm:
            pass
        try:
            async with asyncio.timeout_at(cm.when()):
                await asyncio.get_running_loop().create_future()
        except TimeoutError:
            s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.elapsed == 1.0,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires 3.11+")
def test_async_timeout_reschedule_extension_uses_virtual_deadline() -> None:
    # cm.reschedule(cm.when() + 5): extend the deadline by 5s → fires at t=15.
    class State:
        def __init__(self) -> None:
            self.elapsed: float | None = None

    async def worker(s: State) -> None:
        start = time.monotonic()
        try:
            async with asyncio.timeout(10.0) as cm:
                await asyncio.sleep(5.0)
                when = cm.when()
                assert when is not None
                cm.reschedule(when + 5.0)
                await asyncio.sleep(20.0)
        except TimeoutError:
            s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.elapsed is not None and 14.5 <= s.elapsed <= 15.5,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires 3.11+")
def test_async_timeout_reschedule_from_fresh_loop_time_uses_current_virtual_time() -> None:
    # An application commonly enables a disabled timeout with
    # loop.time() + delay. After virtual time has elapsed, that expression is
    # relative to *now*, not to the context's entry instant.
    class State:
        def __init__(self) -> None:
            self.elapsed: float | None = None
            self.reached_halfway = False

    async def worker(s: State) -> None:
        start = time.monotonic()
        try:
            async with asyncio.timeout(None) as cm:
                await asyncio.sleep(5.0)
                cm.reschedule(asyncio.get_running_loop().time() + 1.0)
                await asyncio.sleep(0.5)
                s.reached_halfway = True
                await asyncio.sleep(1.0)
        except TimeoutError:
            s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.reached_halfway and s.elapsed is not None and 5.9 <= s.elapsed <= 6.1,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_sleep_nan_matches_asyncio_value_error() -> None:
    class State:
        def __init__(self) -> None:
            self.error: BaseException | None = None

    async def worker(s: State) -> None:
        try:
            await asyncio.sleep(math.nan)
        except BaseException as exc:  # noqa: BLE001 - compare the public exception contract
            s.error = exc

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: isinstance(s.error, ValueError) and "NaN" in str(s.error),
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_async_sleep_infinity_stays_cancellable_by_finite_timeout() -> None:
    class State:
        def __init__(self) -> None:
            self.elapsed: float | None = None

    async def worker(s: State) -> None:
        start = time.monotonic()
        try:
            await asyncio.wait_for(asyncio.sleep(math.inf), timeout=1.0)
        except TimeoutError:
            s.elapsed = time.monotonic() - start

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.elapsed is not None and s.elapsed == pytest.approx(1.0),
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires 3.11+")
def test_async_timeout_expiry_converts_cancelled_error_subclasses() -> None:
    # Real asyncio.Timeout.__aexit__ converts any CancelledError *subclass*
    # into TimeoutError once expired; the virtual context previously used an
    # exact `is` check, letting a subclass escape as cancellation (Task crash).
    class MyCancel(asyncio.CancelledError):
        pass

    class State:
        def __init__(self) -> None:
            self.timed_out = False

    async def worker(s: State) -> None:
        try:
            async with asyncio.timeout(1.0):
                try:
                    await asyncio.sleep(10.0)
                except asyncio.CancelledError:
                    raise MyCancel() from None
        except TimeoutError:
            s.timed_out = True

    result = asyncio.run(
        frontrun.explore(
            setup=State,
            workers=[worker],
            invariant=lambda s: s.timed_out,
            clock="virtual",
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds, result.explanation


def test_patched_async_sleep_ignores_scheduler_inherited_by_foreign_thread() -> None:
    # On 3.14t, threading.Thread starts with a copy of the caller's context,
    # so a thread spawned by explored code inherits _scheduler_var /
    # _task_id_var.  asyncio.sleep in that thread's own loop must NOT route
    # through the exploration scheduler (cross-loop condition waits, PyO3
    # engine calls from a second OS thread) — it must behave as unmanaged.
    # Simulated portably by installing a scheduler whose loop-thread id is a
    # different thread's ident.
    from frontrun._async_autopause import _scheduler_var, _task_id_var
    from frontrun._async_virtual_timeouts import _patch_asyncio_sleep, _unpatch_asyncio_sleep
    from frontrun._virtual_clock import VirtualClock

    class ForeignScheduler:
        virtual_clock = VirtualClock()

        def __init__(self) -> None:
            self._event_loop_thread_id = threading.get_ident() - 1  # never this thread

        async def sleep_until(self, task_id: int, deadline: float) -> None:  # pragma: no cover
            raise AssertionError("foreign-thread sleep must not reach the exploration scheduler")

    elapsed_box: dict[str, float] = {}

    async def inner() -> None:
        start = time.perf_counter()
        await asyncio.sleep(0.2)
        elapsed_box["elapsed"] = time.perf_counter() - start

    _patch_asyncio_sleep()
    try:
        token_s = _scheduler_var.set(ForeignScheduler())
        token_t = _task_id_var.set(3)
        try:
            asyncio.run(inner())
        finally:
            _scheduler_var.reset(token_s)
            _task_id_var.reset(token_t)
    finally:
        _unpatch_asyncio_sleep()
    # Unmanaged semantics: the real delay is preserved (neither skipped as a
    # cooperative yield nor virtualised through the foreign scheduler).
    assert elapsed_box["elapsed"] >= 0.15


@pytest.mark.skipif(sys.version_info < (3, 11), reason="asyncio.timeout requires 3.11+")
def test_async_random_stock_primitive_under_timeout_is_not_scored_as_deadlock() -> None:
    # The random runtime patches only sleep/timeouts/time — not asyncio.Event.
    # A task suspended on a *stock* Event inside asyncio.timeout is invisible
    # to the schedule (head stall); the other task then dies in pause()'s
    # watchdog and the run was scored "Deadlock detected" — even though the
    # real program provably recovers via the timeout at virtual t=0.05.
    # A pending virtual deadline must be advanced before declaring the stall.
    from frontrun import explore_async_random

    class State:
        def __init__(self) -> None:
            self.stock_event = asyncio.Event()
            self.timed_out = False

    async def blocked_under_timeout(s: State) -> None:
        try:
            async with asyncio.timeout(0.05):
                await s.stock_event.wait()
        except TimeoutError:
            s.timed_out = True

    async def bystander(s: State) -> None:
        for _ in range(3):
            await asyncio.sleep(0)

    async def _run() -> None:
        result = await explore_async_random(
            setup=State,
            tasks=[blocked_under_timeout, bystander],
            invariant=lambda s: s.timed_out,
            max_attempts=5,
            max_ops=40,
            seed=7,
            deadlock_timeout=2.0,
            timeout_per_run=15.0,
            clock="virtual",
        )
        assert result.property_holds, result.explanation

    asyncio.run(_run())
