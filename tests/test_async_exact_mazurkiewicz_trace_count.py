"""Async DPOR Mazurkiewicz trace counts for simple concurrent programs.

Async counterpart of ``test_exact_mazurkiewicz_trace_count.py``.  Verifies
that ``explore(strategy="dpor")`` on async tasks explores exactly the
theoretically predicted number of Mazurkiewicz traces for the same families
of concurrent programs, translated to async tasks with
``await asyncio.sleep(0)`` as yield points.

Categories (mirroring the sync tests):
1. N tasks mutating independent state   → 1 trace
2. Two tasks racing on N shared vars    → 2^N traces
3. N tasks serialized by a single lock  → N! traces
4. (2) but with a lock                  → 2 traces
"""

from __future__ import annotations

import asyncio
import math

import pytest

import frontrun
from frontrun.cli import require_active


class _Slot:
    """A single mutable slot, isolated as its own Python object."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# Case 1: N tasks mutating independent state
# ---------------------------------------------------------------------------


class TestAsyncIndependentState:
    """N async tasks where task i only touches slot i.

    All cross-task operation pairs access disjoint objects, so every
    pair is independent.  Every linearization belongs to the same
    Mazurkiewicz trace → 1 trace.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_independent_writes(self, n: int) -> None:
        require_active("test_async_exact_independent_writes")

        class State:
            def __init__(self) -> None:
                self.slots = [_Slot() for _ in range(n)]

        def make_task(i: int):  # noqa: ANN202
            async def task_fn(s: State) -> None:
                await asyncio.sleep(0)
                s.slots[i].value = i + 1
                await asyncio.sleep(0)

            return task_fn

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[make_task(i) for i in range(n)],
                invariant=lambda s: all(s.slots[i].value == i + 1 for i in range(len(s.slots))),
                max_executions=1000,
                preemption_bound=None,
                stop_on_first=False,
                deadlock_timeout=5.0,
                total_timeout=60.0,
                strategy="dpor",
                detect_io=False,
            )
        )

        assert result.property_holds, f"Invariant should hold for independent writes (N={n})"
        assert result.num_explored == 1, (
            f"N={n}: Expected exactly 1 Mazurkiewicz trace for independent writes, got {result.num_explored}"
        )


# ---------------------------------------------------------------------------
# Case 2: Two tasks racing on N shared variables (same access order)
# ---------------------------------------------------------------------------


class TestAsyncTwoTasksSharedState:
    """Two async tasks each writing to the same N variables in the same order.

    Each variable v_i is a separate _Slot.  Each task writes to each
    variable in sequence with an await between each write.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_two_tasks_n_shared_vars(self, n: int) -> None:
        """Two tasks writing to N shared variables → 2^N Mazurkiewicz traces."""
        require_active("test_async_exact_two_tasks_shared")

        class State:
            def __init__(self) -> None:
                self.vars = [_Slot() for _ in range(n)]

        def make_task(tid: int):  # noqa: ANN202
            async def task_fn(s: State) -> None:
                for v in s.vars:
                    v.value = tid
                    await asyncio.sleep(0)

            return task_fn

        expected = 2**n
        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[make_task(0), make_task(1)],
                invariant=lambda s: True,
                max_executions=max(expected * 10, 1000),
                preemption_bound=None,
                stop_on_first=False,
                deadlock_timeout=5.0,
                total_timeout=60.0,
                strategy="dpor",
                detect_io=False,
            )
        )

        assert result.num_explored == expected, (
            f"N={n}: Expected exactly {expected} Mazurkiewicz traces (2^{n}), got {result.num_explored}"
        )


# ---------------------------------------------------------------------------
# Case 3: N tasks serialized by a single lock
# ---------------------------------------------------------------------------


class TestAsyncNTasksWithLock:
    """N async tasks competing for a single asyncio.Lock, then updating
    shared state.  The lock serializes all critical sections → N! traces.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_n_tasks_single_lock(self, n: int) -> None:
        """N tasks with single asyncio.Lock → N! Mazurkiewicz traces."""
        require_active("test_async_exact_n_tasks_lock")

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()
                self.shared = _Slot()

        def make_task(tid: int):  # noqa: ANN202
            async def task_fn(s: State) -> None:
                async with s.lock:
                    s.shared.value = tid

            return task_fn

        expected = math.factorial(n)
        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[make_task(i) for i in range(n)],
                invariant=lambda s: True,
                max_executions=max(expected * 10, 1000),
                preemption_bound=None,
                stop_on_first=False,
                deadlock_timeout=5.0,
                total_timeout=60.0,
                strategy="dpor",
                detect_io=False,
            )
        )

        assert result.num_explored == expected, (
            f"N={n}: Expected exactly {expected} Mazurkiewicz traces ({n}!), got {result.num_explored}"
        )


# ---------------------------------------------------------------------------
# Case 4: Two tasks, N shared vars, with a single lock (case 2 + lock)
# ---------------------------------------------------------------------------


class TestAsyncTwoTasksSharedStateWithLock:
    """Two async tasks each writing to N shared variables, protected by a
    single asyncio.Lock over the entire write sequence → 2 traces.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_two_tasks_locked_n_vars(self, n: int) -> None:
        """Two tasks with single lock over N var writes → 2 traces."""
        require_active("test_async_exact_two_tasks_locked")

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()
                self.vars = [_Slot() for _ in range(n)]

        def make_task(tid: int):  # noqa: ANN202
            async def task_fn(s: State) -> None:
                async with s.lock:
                    for v in s.vars:
                        v.value = tid

            return task_fn

        result = asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[make_task(0), make_task(1)],
                invariant=lambda s: True,
                max_executions=100,
                preemption_bound=None,
                stop_on_first=False,
                deadlock_timeout=5.0,
                total_timeout=60.0,
                strategy="dpor",
                detect_io=False,
            )
        )

        assert result.num_explored == 2, (
            f"N={n}: Expected exactly 2 Mazurkiewicz traces (lock serializes), got {result.num_explored}"
        )
