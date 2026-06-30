"""Regression test for free-threaded waiter re-blocking on cooperative locks.

Before the fix, a waiter that woke up and then lost the real lock race did not
report ``lock_wait`` again. DPOR kept scheduling that still-blocked thread,
which made the exact N! lock-order count nondeterministically under- or
over-explore on free-threaded Python.
"""

from __future__ import annotations

import math
import sysconfig
import threading

import pytest

import frontrun

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
SEARCHES = ["bit-reversal:42", "stride", "stride:3", "conflict-first"]


class _Slot:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.shared = _Slot()


def _make_thread(tid: int):  # noqa: ANN202
    def thread_fn(s: _State) -> None:
        with s.lock:
            s.shared.value = tid

    return thread_fn


@pytest.mark.skipif(not FREE_THREADED, reason="Regression is specific to free-threaded Python")
def test_waiters_reblock_after_losing_lock_race() -> None:
    expected = math.factorial(3)
    failures: list[tuple[int, str, int]] = []

    for iteration in range(8):
        for search in SEARCHES:
            result = frontrun.explore(
                setup=_State,
                workers=[_make_thread(i) for i in range(3)],
                invariant=lambda s: True,
                max_executions=1000,
                preemption_bound=None,
                stop_on_first=False,
                detect_io=False,
                total_timeout=30.0,
                search=search,
            )
            if result.num_explored != expected:
                failures.append((iteration, search, result.num_explored))

    assert not failures, f"Unexpected trace counts for single-lock permutations: {failures}"
