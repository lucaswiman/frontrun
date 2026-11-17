"""Defect #22: DPOR never physically executed the reversed order of two
critical sections on the same lock.

Two threads each do ``with lock: ...``.  The engine detected the
acquire/acquire race and inserted a wakeup for the second thread, but the
acquire event was attributed to the scheduling step *after* the one in
which it physically happened (``_dpor_tls._last_path_id`` named the next
transition the scheduler had just picked, not the step the reporting thread
was executing).  The wakeup therefore landed one step too late: the woken
thread found the lock already held, blocked, and the original order
replayed while the engine counted a fresh execution.  ``num_explored`` was
right (N!), ``property_holds`` was wrong (``True`` for an invariant only the
reversed order violates).

These tests pin the physical behaviour, not the execution count: the
reversed acquisition order must actually run.
"""

from __future__ import annotations

import threading

import pytest

import frontrun

pytestmark = pytest.mark.timeout(120)


class TestTwoCriticalSectionsReversed:
    def test_reversed_order_is_physically_executed(self) -> None:
        orders: list[tuple[int, ...]] = []

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.x = 0
                self.order: list[int] = []

        def first(s: State) -> None:
            with s.lock:
                s.order.append(0)
                s.x = 1

        def second(s: State) -> None:
            with s.lock:
                s.order.append(1)
                s.x = 2

        def invariant(s: State) -> bool:
            orders.append(tuple(s.order))
            # Only holds when `second` ran its critical section last.
            return s.x == 2

        result = frontrun.explore(
            setup=State,
            workers=[first, second],
            invariant=invariant,
            max_executions=50,
            preemption_bound=None,
            stop_on_first=False,
            detect_io=False,
            total_timeout=60.0,
        )

        assert (1, 0) in orders, f"reversed critical-section order never executed; physical orders: {orders}"
        assert (0, 1) in orders, f"original order never executed; physical orders: {orders}"
        assert result.property_holds is False, (
            f"x == 2 holds only for order (0, 1); DPOR explored {result.num_explored} executions "
            f"with physical orders {orders} and still certified it"
        )


class TestReadOutsideLockWriteUnderLock:
    """A read outside the lock racing a pop-and-advance under the lock.

    ``assign_head`` snapshots ``head`` without the lock, then writes
    ``items[head]`` under it.  ``pop_head`` pops ``items[head]`` and advances
    ``head`` under the lock.  The two serializable outcomes are
    ``('x', ['b'])`` (assign first) and ``('a', ['x'])`` (pop first).  The
    racy outcome ``('a', ['b', 'x'])`` -- stale key written after the pop --
    needs ``pop_head``'s critical section to run *between* the snapshot and
    ``assign_head``'s acquire, i.e. the reversed acquisition order.
    """

    def test_stale_key_write_is_found(self) -> None:
        outcomes: list[tuple[object, ...]] = []

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.items: dict[int, str] = {1: "a", 2: "b"}
                self.head = 1
                self.popped: str | None = None

        def assign_head(s: State) -> None:
            key = s.head
            with s.lock:
                s.items[key] = "x"

        def pop_head(s: State) -> None:
            with s.lock:
                key = s.head
                s.popped = s.items.pop(key)
                s.head = key + 1

        serializable = {("x", ("b",)), ("a", ("x",))}

        def invariant(s: State) -> bool:
            outcome = (s.popped, tuple(s.items[k] for k in sorted(s.items)))
            outcomes.append(outcome)
            return outcome in serializable

        result = frontrun.explore(
            setup=State,
            workers=[assign_head, pop_head],
            invariant=invariant,
            max_executions=100,
            preemption_bound=None,
            stop_on_first=False,
            detect_io=False,
            total_timeout=60.0,
        )

        assert result.property_holds is False, (
            f"stale-key write never observed; explored {result.num_explored} executions with outcomes {outcomes}"
        )
        assert ("a", ("b", "x")) in outcomes, outcomes
