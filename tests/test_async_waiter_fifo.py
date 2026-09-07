from __future__ import annotations

from collections import OrderedDict
from typing import Any

from frontrun._async_cooperative import _CooperativeAsyncQueue


class _Future:
    def __init__(self, done: bool = False) -> None:
        self._done = done

    def done(self) -> bool:
        return self._done


class _CountingWaiters(OrderedDict[Any, int]):
    def __init__(self) -> None:
        super().__init__()
        self.oldest_pops = 0

    def popitem(self, last: bool = True) -> tuple[Any, int]:
        assert last is False
        self.oldest_pops += 1
        return super().popitem(last=last)


def test_queue_pop_waiter_skips_cancelled_in_constant_time_fifo_steps() -> None:
    queue = _CooperativeAsyncQueue()
    cancelled = [_Future(done=True) for _ in range(32)]
    pending = _Future()
    waiters = _CountingWaiters()
    waiters.update((future, task_id) for task_id, future in enumerate([*cancelled, pending]))

    assert queue._pop_waiter(waiters) == (32, pending)
    assert waiters.oldest_pops == 33
    assert not waiters
