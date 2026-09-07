from __future__ import annotations

import asyncio
from typing import Any

import pytest

from frontrun._async_cooperative import _AsyncWaiters, _CooperativeAsyncQueue


class _CountingWaiters(_AsyncWaiters):
    def __init__(self) -> None:
        super().__init__()
        self.oldest_pops = 0

    def popitem(self, last: bool = True) -> tuple[Any, int]:
        assert last is False
        self.oldest_pops += 1
        return super().popitem(last=last)


@pytest.mark.asyncio
async def test_queue_pop_waiter_skips_cancelled_in_constant_time_fifo_steps() -> None:
    queue = _CooperativeAsyncQueue()
    cancelled = [asyncio.get_running_loop().create_future() for _ in range(32)]
    for future in cancelled:
        future.cancel()
    pending = asyncio.get_running_loop().create_future()
    waiters = _CountingWaiters()
    waiters.update((future, task_id) for task_id, future in enumerate([*cancelled, pending]))

    assert queue._pop_waiter(waiters) == (32, pending)
    assert waiters.oldest_pops == 33
    assert not waiters
