"""Replay scheduler for async DPOR counterexample reproduction.

Async mirror of the sync ``_dpor_runtime/replay.py``: ``_ReplayAsyncScheduler``
drives the tasks along a *recorded* schedule (the ``schedule_trace`` of a
failing exploration run) instead of asking the DPOR engine at each step, so a
counterexample can be replayed deterministically.

It shares the condition-gated park/wake skeleton with the exploration scheduler
via :class:`~frontrun.async_scheduler._AsyncSchedulerBase`; only the
next-task decision differs (walking the recorded schedule through ``_advance``
rather than the engine's ``_schedule_next``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from frontrun import _async_cooperative
from frontrun._async_autopause import _in_scheduler_pause, _scheduler_var, _task_id_var
from frontrun._async_cooperative import (
    _real_asyncio_condition,
    _real_asyncio_sleep,
    _release_task_async_locks,
)
from frontrun._deadlock import DeadlockError, format_cycle
from frontrun._dpor_core import (
    ReplayEngine,
    ReplayExecution,
    RowLockRegistry,
    advance_and_dispatch,
    advance_replay_index,
    extend_replay_schedule,
    wake_sync_id,
)
from frontrun._virtual_clock import DeadlineCoordinator, VirtualClock, WakeEvent
from frontrun.async_scheduler import SchedulerTimeoutError, _AsyncSchedulerBase, frontrun_wait_for

__all__ = ["_ReplayAsyncScheduler"]


class _ReplayAsyncScheduler(_AsyncSchedulerBase):
    """Replay a fixed schedule for async counterexample reproduction."""

    #: Watchdog error prefix distinguishes replay deadlocks from exploration.
    _deadlock_prefix = "Replay deadlock"

    def __init__(
        self,
        schedule: list[int],
        num_tasks: int,
        *,
        deadlock_timeout: float = 5.0,
        virtual_clock: VirtualClock | None = None,
        clock_actor_id: int | None = None,
        detect_sql: bool = False,
        detect_redis: bool = False,
    ) -> None:
        super().__init__(deadlock_timeout=deadlock_timeout)
        self._condition = _real_asyncio_condition()
        self._replay_schedule = list(schedule)
        # Start at index 1 because schedule[0] is consumed as the initial _current_task.
        self._replay_index = 1 if schedule else 0
        self._replay_max_ops = len(self._replay_schedule) * 10 + 10_000
        self._num_replay_tasks = num_tasks
        # Virtual clock replay state (see AsyncDporScheduler): recorded
        # clock-actor entries in the schedule become clock advances.
        self.virtual_clock = virtual_clock
        self._clock_actor_id = clock_actor_id
        self._deadlines = DeadlineCoordinator()
        self._sleepers: dict[int, float] = {}
        # Recorded actor entries reached before any deadline was registered
        # (drift): the owed advance is performed at the next registration.
        self._pending_clock_advances = 0
        self._event_blocked: set[int] = set()  # pyright: ignore[reportIncompatibleVariableOverride]
        self._detect_sql = detect_sql
        self._detect_redis = detect_redis
        self._row_lock_registry = RowLockRegistry()
        self._active_row_locks = self._row_lock_registry._active_row_locks
        self._row_lock_waiters: dict[str, list[tuple[int, asyncio.Future[None]]]] = {}
        self._current_task: int | None = None
        self._current_task_consumed = False
        if schedule:
            first = schedule[0]
            if clock_actor_id is not None and first == clock_actor_id:
                # A leading actor entry cannot advance anything yet (no
                # deadlines registered — a stale wakeup-tree step); skip it
                # without owing an advance.
                self._current_task = None
                self._replay_index = 0
                self._advance()
                self._pending_clock_advances = 0
            else:
                self._current_task = first
        # Stubs so the patched cooperative asyncio.Lock can call
        # engine.report_sync / execution.block_thread without crashing during
        # replay. _lock_blocked mirrors the DPOR scheduler's attribute so the
        # same lock-acquire code path works unmodified.
        self.engine: Any = ReplayEngine()
        self.execution: Any = ReplayExecution()
        self._lock_blocked: dict[int, int] = {}

    def _extend_schedule(self) -> bool:
        return extend_replay_schedule(
            self._replay_schedule,
            self._replay_index,
            self._replay_max_ops,
            self._num_replay_tasks,
            self._tasks_done | self._event_blocked,
        )

    def _on_clock_sleep(self, event: WakeEvent) -> None:
        """Sleep-arm of a replay clock advance: drop the sleeper (no engine)."""
        self._sleepers.pop(event.actor_id, None)

    def _on_clock_timeout(self, event: WakeEvent) -> None:
        """Timeout-arm of a replay clock advance: fire the token, scrub blocked sets."""
        fire = getattr(event.token, "fire", None)
        if fire is not None:
            fire()
        self._lock_blocked.pop(event.actor_id, None)
        self._event_blocked.discard(event.actor_id)

    def _replay_advance_clock(self, target: float | None = None) -> None:
        """Advance the virtual clock during replay and wake due deadlines."""
        clock = self.virtual_clock
        if clock is None:
            return
        advance_and_dispatch(
            self._deadlines, clock, target, on_sleep=self._on_clock_sleep, on_timeout=self._on_clock_timeout
        )

    def add_timeout_deadline(self, task_id: int, deadline: float, token: object) -> None:
        self._deadlines.add_timeout(task_id, deadline, token)
        if self._pending_clock_advances > 0:
            self._pending_clock_advances -= 1
            self._replay_advance_clock()

    def remove_timeout_deadline(self, task_id: int, token: object) -> None:
        self._deadlines.cancel(task_id, token)

    def _advance(self) -> None:
        """Advance ``_replay_index`` and ``_current_task`` to the next live actor."""
        while True:
            self._replay_index, next_actor = advance_replay_index(
                self._replay_schedule,
                self._replay_index,
                self._extend_schedule,
                self._tasks_done | self._event_blocked,
            )
            if next_actor is not None and self._clock_actor_id is not None and next_actor == self._clock_actor_id:
                # Recorded clock-actor step: advance the clock and keep going.
                if self._deadlines.has_pending():
                    self._replay_advance_clock()
                else:
                    # Drift: the sleeper has not registered yet — owe the
                    # advance and perform it on registration (sleep_until).
                    self._pending_clock_advances += 1
                continue
            break
        self._current_task = next_actor
        self._current_task_consumed = False
        if next_actor is None:
            self._finished = True

    async def sleep_until(self, task_id: int, deadline: float | None = None, *, duration: float | None = None) -> None:
        """Replay counterpart of ``AsyncDporScheduler.sleep_until``."""
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            await _real_asyncio_sleep(0)
            self._progress += 1
            async with self._condition:
                if deadline is None:
                    if duration is None or self.virtual_clock is None:
                        raise TypeError("sleep_until needs either deadline= or duration= (with a virtual clock)")
                    deadline = self.virtual_clock.now() + duration
                if self._finished or self._error:
                    return
                self._sleepers[task_id] = deadline
                self._deadlines.add_sleep(task_id, deadline, wake_sync_id(task_id))
                if self._pending_clock_advances > 0:
                    # Replay owed us an actor step that arrived before this
                    # registration (drift): perform it now.
                    self._pending_clock_advances -= 1
                    self._replay_advance_clock()
                if self._current_task == task_id:
                    self._advance()
                self._condition.notify_all()
                try:
                    while task_id in self._sleepers:
                        if self._finished or self._error:
                            return
                        alive = [t for t in range(self._num_replay_tasks) if t not in self._tasks_done]
                        if alive and all(t in self._sleepers for t in alive):
                            # Every live task is asleep: only time can move.
                            self._replay_advance_clock()
                            self._condition.notify_all()
                            continue
                        try:
                            await frontrun_wait_for(self._condition.wait(), timeout=self.deadlock_timeout)
                        except asyncio.TimeoutError:
                            self._error = SchedulerTimeoutError(
                                f"Replay deadlock: task {task_id} sleeping until t={deadline} was never woken"
                            )
                            self._condition.notify_all()
                            return
                    while not (self._finished or self._error) and self._current_task != task_id:
                        if self._current_task in self._event_blocked:
                            self._advance()
                            self._condition.notify_all()
                            continue
                        try:
                            await frontrun_wait_for(self._condition.wait(), timeout=self.deadlock_timeout)
                        except asyncio.TimeoutError:
                            self._error = SchedulerTimeoutError(
                                f"Replay deadlock: task {task_id} woke from sleep but was never scheduled"
                            )
                            self._condition.notify_all()
                            return
                    if not (self._finished or self._error):
                        self._current_task_consumed = True
                finally:
                    self._sleepers.pop(task_id, None)
                    self._deadlines.cancel_sleep(task_id)
        finally:
            _in_scheduler_pause.set(depth)

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        # Schedule-drift safety net: if replay scheduled a sleeping task,
        # time must pass before it can move — jump to its deadline.
        cur = self._current_task
        if cur in self._event_blocked:
            self._advance()
            self._condition.notify_all()
            cur = self._current_task
        if cur is not None and self.virtual_clock is not None and cur in self._sleepers:
            self._replay_advance_clock(self._sleepers[cur])
            self._condition.notify_all()
        if self._current_task is None:
            self._finished = True
            return True
        if self._current_task == task_id:
            return True
        # Mirror AsyncDporScheduler.should_proceed: if the currently-scheduled
        # task is blocked on a lock held by task_id, let task_id proceed so it
        # can release the lock.  The recorded schedule contains the engine's
        # raw pick of the blocked task (the exploration-time holder override is
        # not written to the trace), so without this override the picked task is
        # stuck inside the real lock.acquire() and replay deadlocks.
        if self._current_task in self._lock_blocked:
            holder = self._lock_blocked[self._current_task]
            if holder == task_id:
                return True
        return False

    # -- _AsyncSchedulerBase scheduling hooks ----------------------------

    def _should_kick(self, task_id: int) -> bool:
        return self._current_task == task_id or self._current_task in self._event_blocked

    def _perform_kick(self, task_id: int) -> None:
        self._advance()

    def _recover_stalled_schedule(self) -> bool:
        if self._current_task in self._event_blocked:
            self._advance()
            self._condition.notify_all()
            return True
        return False

    def _on_error_set(self) -> None:
        for waiters in self._row_lock_waiters.values():
            for _task_id, future in waiters:
                if not future.done():
                    future.set_result(None)

    async def acquire_row_locks_async(self, task_id: int, resource_ids: list[str]) -> list[str]:
        graph = _async_cooperative._async_wait_graph
        acquired: list[str] = []
        for res_id in resource_ids:
            lock_id = self._row_lock_registry._row_lock_int_id(res_id)
            while (holder := self._active_row_locks.get(res_id)) is not None and holder != task_id:
                if graph is not None:
                    cycle = graph.add_waiting(task_id, lock_id, kind="row_lock")
                    if cycle is not None:
                        graph.remove_waiting(task_id, lock_id, kind="row_lock")
                        desc = format_cycle(cycle, self._row_lock_registry.id_to_resource())
                        error = DeadlockError(f"Row-lock deadlock detected: {desc}", desc)
                        await self._report_error(error)
                        raise error
                future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._row_lock_waiters.setdefault(res_id, []).append((task_id, future))
                self._event_blocked.add(task_id)
                self._lock_blocked[task_id] = holder
                depth = _in_scheduler_pause.get()
                _in_scheduler_pause.set(depth + 1)
                try:
                    await self.kick_stalled_schedule(task_id)
                    await future
                    self._event_blocked.discard(task_id)
                    self._lock_blocked.pop(task_id, None)
                    if self._error is not None:
                        raise self._error
                    await self.wait_until_scheduled_after_block(task_id, "SQL row lock")
                    if self._error is not None:
                        raise self._error
                finally:
                    if graph is not None:
                        graph.remove_waiting(task_id, lock_id, kind="row_lock")
                    waiters = self._row_lock_waiters.get(res_id)
                    if waiters is not None:
                        waiters[:] = [entry for entry in waiters if entry[1] is not future]
                        if not waiters:
                            self._row_lock_waiters.pop(res_id, None)
                    self._event_blocked.discard(task_id)
                    self._lock_blocked.pop(task_id, None)
                    _in_scheduler_pause.set(depth)
            self._row_lock_registry.record_acquire(task_id, res_id, graph)
            acquired.append(res_id)
        return acquired

    def release_row_locks(self, task_id: int, resources: list[str] | None = None) -> None:
        graph = _async_cooperative._async_wait_graph
        for res_id, _lock_id in self._row_lock_registry.pop(task_id, graph, resources):
            for _waiter, future in self._row_lock_waiters.get(res_id, []):
                if not future.done():
                    future.set_result(None)

    def report_and_wait(self, _frame: Any, _task_id: int) -> bool:
        """Match the SQL interception scheduler port during replay.

        Await-point scheduling already controls replay; this synchronous hook
        exists so SQL interception can complete the same boundary handshake it
        uses during exploration.
        """
        return True

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        self._current_task_consumed = True

    def on_task_yielded(self, task_id: int) -> None:
        if self._finished or self._error:
            return
        if self._current_task == task_id and self._current_task_consumed:
            self._advance()
            self._notify_waiters_soon()

    def _setup_task_context(self, task_id: Any) -> None:
        _scheduler_var.set(self)
        _task_id_var.set(task_id)
        if self._detect_sql or self._detect_redis:
            from frontrun._io_detection import (
                set_dpor_scheduler_task,
                set_dpor_thread_id_task,
                set_io_reporter,
                set_tx_store_task,
            )

            set_dpor_scheduler_task(self)
            set_dpor_thread_id_task(task_id)
            set_io_reporter(lambda _resource_id, _kind: None)
            store = set_tx_store_task()
            store._in_transaction = False
            store._is_autobegin = False
            store._tx_buffer = []
            store._tx_savepoints = {}

    def _cleanup_task_context(self, task_id: Any) -> None:
        # Release any asyncio.Lock objects still held by this task (e.g. the
        # task crashed or was cancelled without release()).  Without this,
        # stale holding edges / lock owners leak into later replay attempts
        # and cause spurious DeadlockError or phantom ownership.
        _release_task_async_locks(task_id)
        self.release_row_locks(task_id)
        _scheduler_var.set(None)
        _task_id_var.set(None)
        if self._detect_sql or self._detect_redis:
            from frontrun._io_detection import set_dpor_scheduler_task, set_dpor_thread_id_task, set_io_reporter

            set_dpor_scheduler_task(None)
            set_dpor_thread_id_task(None)
            if len(self._tasks_done) + 1 >= self._num_replay_tasks:
                set_io_reporter(None)

    def finish_task(self, task_id: int) -> None:
        self._tasks_done.add(task_id)

    async def _mark_done(self, task_id: Any) -> None:
        """Mark a task as finished and update _current_task if needed."""
        async with self._condition:
            self._tasks_done.add(task_id)
            if self._current_task == task_id:
                # Advance to the next scheduled task so other tasks can proceed.
                self._advance()
            self._condition.notify_all()
