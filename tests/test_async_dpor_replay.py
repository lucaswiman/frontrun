"""Regression tests for async DPOR counterexample replay (findings F3, F5).

F3: ``_ReplayAsyncScheduler`` lacked the ``engine`` / ``execution`` /
``_lock_blocked`` attributes that the patched ``_CooperativeAsyncLock``
references, so every lock-involving counterexample raised an AttributeError
during replay.  That error was swallowed by an over-broad
``except (TimeoutError, Exception)``, so reproduction_successes was always 0.

F5: cooperative-lock global state (wait-for graph, lock owners, held locks)
was cleared only between exploration iterations, never between the up-to-10
replay attempts, so stale edges / id() reuse could cause spurious
DeadlockError or phantom ownership in later replays.
"""

from __future__ import annotations

import asyncio

from frontrun.cli import require_active


def test_lock_counterexample_reproduces() -> None:
    """A counterexample whose tasks use asyncio.Lock must reproduce.

    The increment acquires a lock only around the *read*, then releases and
    writes after an await — a lost update is still possible.  DPOR finds the
    failing schedule; replaying it must reproduce the violation rather than
    crash in lock plumbing and report 0 reproductions.
    """
    require_active("test_async_dpor_lock_replay")
    import frontrun

    class Counter:
        def __init__(self) -> None:
            self.value = 0
            self.lock = asyncio.Lock()

        async def increment(self) -> None:
            async with self.lock:
                temp = self.value
            await asyncio.sleep(0)
            self.value = temp + 1

    async def _do_increment(c: Counter) -> None:
        await c.increment()

    result = asyncio.run(
        frontrun.explore(
            setup=Counter,
            workers=[_do_increment, _do_increment],
            invariant=lambda c: c.value == 2,
            strategy="dpor",
            deadlock_timeout=2.0,
            timeout_per_run=3.0,
            reproduce_on_failure=10,
        )
    )

    assert not result.property_holds, "lost update should be found"
    assert result.reproduction_attempts == 10, result.reproduction_attempts
    # The core F3 bug: replay crashed in lock plumbing → 0 successes.
    assert result.reproduction_successes > 0, (
        f"counterexample must reproduce, got {result.reproduction_successes}/{result.reproduction_attempts}"
    )


def test_lock_blocked_holder_counterexample_reproduces() -> None:
    """A counterexample where a task blocks on a lock held across an await must reproduce.

    ``AsyncDporScheduler.should_proceed`` applies a lock-blocked holder override:
    when the engine picks a task blocked on a contended ``asyncio.Lock`` held
    across an await, the scheduler silently runs the *holder* so it can release
    the lock.  ``_ReplayAsyncScheduler`` populates ``_lock_blocked`` (the patched
    lock-acquire path writes it) but never *read* it, so replaying a recorded
    schedule that contains an engine pick of a lock-blocked task deadlocked: the
    picked task is stuck inside the real ``lock.acquire()`` and the holder is
    denied.  Every replay attempt then hung for ``deadlock_timeout`` and was
    scored as a failed reproduction, so a genuine deterministic counterexample
    reported ``reproduction_successes == 0``.

    ``deadlock_timeout`` is kept short so the *pre-fix* hang fails in bounded
    time rather than stalling the suite.
    """
    require_active("test_async_dpor_lock_blocked_replay")
    import frontrun

    class S:
        def __init__(self) -> None:
            self.lock = asyncio.Lock()
            self.was_held = False

    async def t0(s: S) -> None:
        async with s.lock:  # holds the lock ACROSS an await
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    async def t1(s: S) -> None:
        await asyncio.sleep(0)
        if s.lock.locked():
            s.was_held = True
        async with s.lock:  # blocks on the held lock → engine picks t1, holder t0 must run
            await asyncio.sleep(0)

    result = asyncio.run(
        frontrun.explore(
            setup=S,
            workers=[t0, t1],
            invariant=lambda s: not s.was_held,
            strategy="dpor",
            deadlock_timeout=1.0,
            timeout_per_run=3.0,
            reproduce_on_failure=5,
        )
    )

    assert not result.property_holds, "lock-held-across-await violation should be found"
    assert result.reproduction_attempts == 5, result.reproduction_attempts
    # The bug: replay of the recorded schedule deadlocks because
    # _ReplayAsyncScheduler.should_proceed lacks the _lock_blocked holder
    # override, so every attempt hangs deadlock_timeout → 0 successes.
    assert result.reproduction_successes > 0, (
        f"lock-blocked counterexample must reproduce, "
        f"got {result.reproduction_successes}/{result.reproduction_attempts}"
    )
