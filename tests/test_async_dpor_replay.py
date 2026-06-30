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
