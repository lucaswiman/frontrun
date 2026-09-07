"""Tests for task-aware DPOR context resolution in async exploration.

Async DPOR runs all tasks on a single event-loop thread, so the per-thread
``threading.local`` storage in ``_io_detection`` cannot distinguish tasks.
These tests verify that the DPOR thread-id / context and the SQL transaction
state are resolved per *task* (via contextvars), not per OS thread.

Regression coverage for findings F2 and F4.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from frontrun import _io_detection
from frontrun._async_autopause import _in_scheduler_pause, _scheduler_var, _task_id_var
from frontrun._async_dpor_replay import _ReplayAsyncScheduler
from frontrun.async_dpor import AsyncDporScheduler
from frontrun.async_scheduler import InterleavedLoop
from frontrun.async_shuffler import AwaitScheduler


def test_dpor_context_is_task_aware() -> None:
    """``get_dpor_context`` returns ``(scheduler, current_task_id)`` per task."""
    sentinel_scheduler = object()
    observed: dict[str, tuple[int | None, tuple[object, int] | None]] = {}

    async def task(task_id: int) -> None:
        _task_id_var.set(task_id)
        _io_detection.set_dpor_scheduler_task(sentinel_scheduler)
        _io_detection.set_dpor_thread_id_task(task_id)
        await asyncio.sleep(0)
        observed[f"task{task_id}"] = (_io_detection.get_dpor_thread_id(), _io_detection.get_dpor_context())

    async def main() -> None:
        await asyncio.gather(task(0), task(1))

    asyncio.run(main())

    assert observed["task0"] == (0, (sentinel_scheduler, 0)), observed
    assert observed["task1"] == (1, (sentinel_scheduler, 1)), observed


@pytest.mark.parametrize("scheduler_type", [AwaitScheduler, _ReplayAsyncScheduler, AsyncDporScheduler])
def test_sql_reporter_belongs_to_each_task(
    scheduler_type: type[AwaitScheduler] | type[_ReplayAsyncScheduler] | type[AsyncDporScheduler],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copied cleared context must not suppress the next scheduler's reporter."""
    from frontrun._dpor import PyDporEngine

    scheduler = (
        AsyncDporScheduler(engine := PyDporEngine(2), engine.begin_execution(), 2, detect_sql=True)
        if scheduler_type is AsyncDporScheduler
        else scheduler_type([], 2, detect_sql=True)
    )
    parent_reporter = object()
    monkeypatch.setattr(_io_detection._io_tls, "io_reporter", parent_reporter, raising=False)

    async def worker(task_id: int) -> None:
        scheduler._setup_task_context(task_id)
        try:
            reporter = _io_detection.get_io_reporter()
            assert reporter is not None
            await asyncio.sleep(0)
            assert _io_detection.get_io_reporter() is reporter
        finally:
            scheduler._cleanup_task_context(task_id)
            scheduler._tasks_done.add(task_id)

    async def scenario() -> None:
        token = _io_detection._io_reporter_var.set(None)
        try:
            await asyncio.gather(worker(0), worker(1))
            assert _io_detection._io_tls.io_reporter is parent_reporter
        finally:
            _io_detection._io_reporter_var.reset(token)

    asyncio.run(scenario())


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

    assert sorted(r for r in reported if r[2] == "write") == [(0, "sql:t0", "write"), (1, "sql:t1", "write")]


def test_async_random_and_replay_record_their_event_loop_thread() -> None:
    parent = threading.get_ident()
    observed: list[tuple[int, int, int]] = []

    def construct_schedulers() -> None:
        current = threading.get_ident()
        random_scheduler = AwaitScheduler([], 1)
        replay_scheduler = _ReplayAsyncScheduler([], 1)
        observed.append((current, random_scheduler._event_loop_thread_id, replay_scheduler._event_loop_thread_id))

    thread = threading.Thread(target=construct_schedulers)
    thread.start()
    thread.join()

    assert len(observed) == 1
    current, random_thread, replay_thread = observed[0]
    assert current != parent
    assert random_thread == current
    assert replay_thread == current


class _PauseRecorder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def pause(self, task_id: int) -> None:
        from frontrun._async_autopause import _in_scheduler_pause

        self.calls.append(task_id)
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            await asyncio.sleep(0)
        finally:
            _in_scheduler_pause.set(depth)


def test_wrap_auto_paused_tasks_inserts_scheduler_pause() -> None:
    from frontrun._async_autopause import wrap_auto_paused_tasks

    recorder = _PauseRecorder()
    events: list[str] = []

    async def task() -> None:
        events.append("before")
        await asyncio.sleep(0)
        events.append("after")

    wrapped = wrap_auto_paused_tasks({7: task}, recorder)
    asyncio.run(wrapped[7]())

    assert events == ["before", "after"]
    assert recorder.calls
    assert set(recorder.calls) == {7}


def test_replay_task_context_is_installed_and_cleared() -> None:
    async def scenario() -> None:
        scheduler = _ReplayAsyncScheduler([0], 1)
        scheduler._setup_task_context(0)
        try:
            assert _scheduler_var.get() is scheduler
            assert _task_id_var.get() == 0
        finally:
            scheduler._cleanup_task_context(0)
        assert _scheduler_var.get() is None
        assert _task_id_var.get() is None

    asyncio.run(scenario())


def test_replay_pause_suppresses_nested_autopause(monkeypatch: pytest.MonkeyPatch) -> None:
    depths: list[int] = []

    async def record_pause(self: InterleavedLoop, task_id: int, marker: object = None) -> None:
        depths.append(_in_scheduler_pause.get())
        await asyncio.sleep(0)
        depths.append(_in_scheduler_pause.get())

    monkeypatch.setattr(InterleavedLoop, "pause", record_pause)

    async def scenario() -> None:
        await _ReplayAsyncScheduler([0], 1).pause(0)
        assert _in_scheduler_pause.get() == 0

    asyncio.run(scenario())
    assert depths == [1, 1]


def test_dpor_finishes_only_after_all_engine_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    from frontrun._dpor import PyDporEngine

    async def scenario() -> None:
        engine = PyDporEngine(2)
        scheduler = AsyncDporScheduler(engine, engine.begin_execution(), 2)
        scheduler._num_tasks = 0  # run_all has not initialized its task count yet.
        monkeypatch.setattr(scheduler, "_schedule_next", lambda: None)
        scheduler._current_task = 0
        await scheduler._mark_done(0)
        assert not scheduler._finished
        scheduler._current_task = 1
        await scheduler._mark_done(1)
        assert scheduler._finished

    asyncio.run(scenario())
