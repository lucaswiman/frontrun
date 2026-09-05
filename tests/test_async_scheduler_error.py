"""Regression tests for scheduler-error propagation (finding F1).

When the async scheduler detects a deadlock/timeout in ``pause()`` it sets
``self._error`` and notifies all waiters.  Every subsequent ``pause()`` then
short-circuits (``if self._finished or self._error: return``), so all tasks
free-run to completion and ``run_all`` previously returned normally — the
scheduler error was silently swallowed and the run was scored as a valid
exploration.  ``run_all`` must instead surface ``self._error`` so the
exploration loop can classify the run as a deadlock/timeout rather than a
normal completion.
"""

from __future__ import annotations

import asyncio

import pytest

from frontrun.async_scheduler import InterleavedLoop, SchedulerTimeoutError


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
