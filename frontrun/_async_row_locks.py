"""Shared modeled row-lock waiting protocol for async exploration and replay."""

from __future__ import annotations

import asyncio
from typing import Any

from frontrun import _async_cooperative
from frontrun._async_autopause import _in_scheduler_pause
from frontrun._deadlock import DeadlockError, format_cycle
from frontrun._dpor_core import RowLockRegistry


class AsyncRowLockProtocol:
    """Own row-lock graph edges, parked waiters, wakeup, and cancellation cleanup."""

    def __init__(self, host: Any, registry: RowLockRegistry) -> None:
        self._host = host
        self._registry = registry
        self.waiters: dict[str, list[tuple[int, asyncio.Future[None]]]] = {}

    async def acquire(self, task_id: int, resource_ids: list[str]) -> list[str]:
        graph = _async_cooperative._async_wait_graph
        acquired: list[str] = []
        newly_acquired: list[str] = []
        try:
            for res_id in resource_ids:
                lock_id = self._registry._row_lock_int_id(res_id)
                while (holder := self._registry.active_lock_owner(res_id)) is not None and holder != task_id:
                    if graph is not None and not self._host._deadlines.in_timed_wait(task_id):
                        cycle = graph.add_waiting(task_id, lock_id, kind="row_lock")
                        if cycle is not None:
                            graph.remove_waiting(task_id, lock_id, kind="row_lock")
                            desc = format_cycle(cycle, self._registry.id_to_resource())
                            error = DeadlockError(f"Row-lock deadlock detected: {desc}", desc)
                            await self._host._report_error(error)
                            raise error

                    future = asyncio.get_running_loop().create_future()
                    self.waiters.setdefault(res_id, []).append((task_id, future))
                    self._host._event_blocked.add(task_id)
                    self._host._lock_blocked[task_id] = holder
                    self._host.execution.block_thread(task_id)
                    depth = _in_scheduler_pause.get()
                    _in_scheduler_pause.set(depth + 1)
                    unblocked = False
                    try:
                        await self._host.kick_stalled_schedule(task_id)
                        await future
                        self._host.execution.unblock_thread(task_id)
                        unblocked = True
                        self._host._event_blocked.discard(task_id)
                        self._host._lock_blocked.pop(task_id, None)
                        if self._host._error is not None:
                            raise self._host._error
                        await self._host.wait_until_scheduled_after_block(task_id, "SQL row lock")
                        if self._host._error is not None:
                            raise self._host._error
                    finally:
                        if graph is not None:
                            graph.remove_waiting(task_id, lock_id, kind="row_lock")
                        waiters = self.waiters.get(res_id)
                        if waiters is not None:
                            waiters[:] = [entry for entry in waiters if entry[1] is not future]
                            if not waiters:
                                self.waiters.pop(res_id, None)
                        self._host._event_blocked.discard(task_id)
                        self._host._lock_blocked.pop(task_id, None)
                        if not unblocked:
                            self._host.execution.unblock_thread(task_id)
                        _in_scheduler_pause.set(depth)

                if self._registry.active_lock_owner(res_id) is None:
                    newly_acquired.append(res_id)
                self._registry.record_acquire(task_id, res_id, graph)
                acquired.append(res_id)
            return acquired
        except BaseException:
            self.release(task_id, newly_acquired)
            raise

    def release(self, task_id: int, resources: list[str] | None = None) -> None:
        graph = _async_cooperative._async_wait_graph
        for res_id, _lock_id in self._registry.pop(task_id, graph, resources):
            for _waiter, future in self.waiters.get(res_id, []):
                if not future.done():
                    future.set_result(None)

    def wake_all(self) -> None:
        for waiters in self.waiters.values():
            for _task_id, future in waiters:
                if not future.done():
                    future.set_result(None)
