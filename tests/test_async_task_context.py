"""Tests for task-aware DPOR context resolution in async exploration.

Async DPOR runs all tasks on a single event-loop thread, so the per-thread
``threading.local`` storage in ``_io_detection`` cannot distinguish tasks.
These tests verify that the DPOR thread-id / context and the SQL transaction
state are resolved per *task* (via contextvars), not per OS thread.

Regression coverage for findings F2 and F4.
"""

from __future__ import annotations

import asyncio

from frontrun import _io_detection
from frontrun._async_autopause import _task_id_var


def test_dpor_thread_id_is_task_aware() -> None:
    """``get_dpor_thread_id`` must reflect the currently-running task.

    F2: each task's ``_setup_task_context`` previously called
    ``set_dpor_thread_id`` once at start on shared threading.local storage,
    so after all tasks start the value was permanently the last task's id.
    With a contextvar-backed store, reading it inside a task returns that
    task's id.
    """
    observed: dict[str, int | None] = {}

    async def task(task_id: int) -> None:
        _task_id_var.set(task_id)
        _io_detection.set_dpor_thread_id_task(task_id)
        # Yield so the other task runs.  Under the old (threading.local) bug
        # this clobbered the shared thread id; the task-aware contextvar
        # setter keeps each task's id isolated.
        await asyncio.sleep(0)
        observed[f"task{task_id}"] = _io_detection.get_dpor_thread_id()

    async def main() -> None:
        await asyncio.gather(task(0), task(1))

    asyncio.run(main())

    assert observed["task0"] == 0, observed
    assert observed["task1"] == 1, observed


def test_dpor_context_is_task_aware() -> None:
    """``get_dpor_context`` returns ``(scheduler, current_task_id)`` per task."""
    sentinel_scheduler = object()
    observed: dict[str, tuple[object, int] | None] = {}

    async def task(task_id: int) -> None:
        _task_id_var.set(task_id)
        _io_detection.set_dpor_scheduler_task(sentinel_scheduler)
        _io_detection.set_dpor_thread_id_task(task_id)
        await asyncio.sleep(0)
        observed[f"task{task_id}"] = _io_detection.get_dpor_context()

    async def main() -> None:
        await asyncio.gather(task(0), task(1))

    asyncio.run(main())

    assert observed["task0"] == (sentinel_scheduler, 0), observed
    assert observed["task1"] == (sentinel_scheduler, 1), observed


def test_transaction_state_is_task_aware() -> None:
    """SQL transaction state must be isolated per async task.

    F4: ``_in_transaction`` / ``_tx_buffer`` lived on threading.local shared
    by all tasks on the event-loop thread, so interleaved transactions
    corrupted each other.
    """
    from frontrun import _sql_transactions

    reported: list[tuple[int, str, str]] = []

    async def task(task_id: int) -> None:
        _task_id_var.set(task_id)
        # Install a per-task transaction store (as AsyncDporScheduler does).
        _io_detection.set_tx_store_task()
        # Isolate from any leaked DPOR scheduler context — this test only
        # exercises transaction buffering, not row-lock release.
        _io_detection.set_dpor_scheduler_task(None)

        def reporter(res_id: str, kind: str) -> None:
            reported.append((task_id, res_id, kind))

        # Begin a transaction and buffer a write.
        _sql_transactions._handle_tx_op(reporter, _sql_transactions.TxOp.BEGIN)
        await asyncio.sleep(0)  # let the other task interleave its BEGIN
        _sql_transactions._report_or_buffer(reporter, f"sql:t{task_id}", "write")
        await asyncio.sleep(0)
        # Commit: must flush exactly this task's buffered write, nothing else.
        _sql_transactions._handle_tx_op(reporter, _sql_transactions.TxOp.COMMIT)

    async def main() -> None:
        await asyncio.gather(task(0), task(1))

    asyncio.run(main())

    # Each task should have flushed exactly its own buffered write on COMMIT.
    assert (0, "sql:t0", "write") in reported, reported
    assert (1, "sql:t1", "write") in reported, reported
    # No task should have flushed the other task's write under its own id.
    assert (0, "sql:t1", "write") not in reported, reported
    assert (1, "sql:t0", "write") not in reported, reported
    # Exactly two writes total (no double-flush from shared-buffer corruption).
    writes = [r for r in reported if r[2] == "write"]
    assert len(writes) == 2, reported
