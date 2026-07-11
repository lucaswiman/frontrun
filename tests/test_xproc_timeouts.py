"""Liveness and timeout diagnosis for the cross-process coordinators.

Pre-release audit regressions:

- The DPOR relay-join budget must bound *lack of progress*, not total
  iteration wall time: ``deadlock_timeout`` bounds silence between steps, so a
  healthy iteration may legitimately run far longer than any fixed multiple of
  it and must still return ``ok=True``.
- A genuine stall (unmodeled DB-level blocking, or a statement slower than
  ``deadlock_timeout``) must surface as ``failure_kind="timeout"`` with the
  raise-``deadlock_timeout`` advice — in BOTH coordinators — not as a
  misleading "worker disconnected" / "worker connection failed" worker_error.
- The exhaustive coordinator must bound scheduling steps per run so a
  nonterminating worker (``while True`` around scheduled statements) cannot
  hang ``explore()`` forever.

Workers run as in-process threads (ThreadLauncher) over a real AF_UNIX socket,
matching the other xproc functional tests.
"""

from __future__ import annotations

import threading
import time

from frontrun._dpor_runtime.xproc.coordinator import CrossProcessCoordinator
from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher


def _join_worker_threads() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith("xproc-worker-"):
            thread.join(timeout=10.0)


# ---------------------------------------------------------------------------
# DPOR coordinator: relay liveness must be progress-based
# ---------------------------------------------------------------------------


def test_dpor_healthy_long_iteration_is_not_killed_by_join_budget() -> None:
    # Each step stays below the recv timeout, while the total run exceeds the
    # injected relay backstop. Every frame must reset the no-progress budget.
    steps, step_time = 6, 0.03

    def healthy(proxy) -> None:
        for _ in range(steps):
            if not proxy.report_and_wait(None, 0):
                return
            time.sleep(step_time)

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=0.2, max_executions=1)
    coord._relay_no_progress_budget = 0.12
    result = coord.explore(
        worker_set=ThreadLauncher([healthy, healthy]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert result.ok, f"healthy long iteration reported as failure: {result.failure!r} ({result.failure_kind})"
    assert result.failure is None


def test_dpor_genuine_stall_is_diagnosed_as_timeout_not_worker_error() -> None:
    # Regression: a worker stalled mid-step longer than the relay budget
    # (simulating unmodeled DB-level blocking) tripped the relay join budget /
    # recv timeout and was reported as failure_kind="worker_error" ("worker
    # connection failed" or "worker disconnected"), masking _evaluate's
    # documented failure_kind="timeout" diagnosis and its raise-deadlock_timeout
    # advice.
    release = threading.Event()

    def staller(proxy) -> None:
        if not proxy.report_and_wait(None, 0):
            return
        release.wait(15.0)  # unmodeled stall while holding the turn (> 2*0.5+10 budget)
        proxy.report_and_wait(None, 0)

    def waiter(proxy) -> None:
        for _ in range(3):
            if not proxy.report_and_wait(None, 0):
                return

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=0.5, max_executions=1)
    try:
        result = coord.explore(
            worker_set=ThreadLauncher([staller, waiter]),
            setup=lambda: None,
            invariant=lambda: True,
        )
    finally:
        release.set()  # let the stalled worker unwind promptly
        _join_worker_threads()
    assert not result.ok
    assert result.failure_kind == "timeout", f"got {result.failure_kind!r}: {result.failure!r}"
    assert "deadlock_timeout" in (result.failure or "")


def test_dpor_worker_disconnect_still_reported_as_worker_error() -> None:
    # Control for the stall diagnosis: an actual disconnect (socket EOF) must
    # keep surfacing as a worker_error, not be re-labelled a timeout.
    def disconnect(proxy) -> None:
        proxy._sock.close()

    coord = DporCrossProcessCoordinator(num_workers=1, deadlock_timeout=0.5)
    result = coord.explore(
        worker_set=ThreadLauncher([disconnect]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "disconnected" in (result.failure or "")


# ---------------------------------------------------------------------------
# Exhaustive coordinator: alive-but-slow is a timeout, not a disconnect
# ---------------------------------------------------------------------------


def test_exhaustive_slow_worker_is_diagnosed_as_timeout() -> None:
    # Regression: _advance collapsed recv TimeoutError and OSError/EOF into
    # "worker disconnected or timed out" -> failure_kind="worker_error", so the
    # documented "timeout" kind was unreachable and a slow-but-alive worker was
    # reported as disconnected.
    release = threading.Event()

    def slow(proxy) -> None:
        if not proxy.report_and_wait(None, 0):
            return
        release.wait(5.0)  # statement slower than deadlock_timeout=1.0
        proxy.report_and_wait(None, 0)

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=1.0)
    try:
        result = coord.explore(
            worker_set=ThreadLauncher([slow]),
            setup=lambda: None,
            invariant=lambda: True,
            max_iterations=2,
        )
    finally:
        release.set()
        _join_worker_threads()
    assert not result.ok
    assert result.failure_kind == "timeout", f"got {result.failure_kind!r}: {result.failure!r}"
    assert "deadlock_timeout" in (result.failure or "")


def test_exhaustive_disconnected_worker_still_reported_as_worker_error() -> None:
    # Control: a worker that dies (socket EOF) is a worker_error, not a timeout.
    def die(proxy) -> None:
        if not proxy.report_and_wait(None, 0):
            return
        proxy._sock.close()

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=1.0)
    result = coord.explore(
        worker_set=ThreadLauncher([die]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=2,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "disconnected" in (result.failure or "")


# ---------------------------------------------------------------------------
# Exhaustive coordinator: per-run step bound
# ---------------------------------------------------------------------------


def test_exhaustive_step_cap_aborts_nonterminating_worker() -> None:
    # Regression: a nonterminating worker (while True around scheduled
    # statements) hung explore() forever — frames keep arriving so the per-recv
    # deadlock_timeout never fires, and max_iterations only bounds *completed*
    # iterations. A per-run step cap must abort the run with a clear error.
    def spinner(proxy) -> None:
        while proxy.report_and_wait(None, 0):
            pass

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=2.0, max_steps_per_run=50)
    result = coord.explore(
        worker_set=ThreadLauncher([spinner]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=4,
    )
    assert not result.ok
    assert result.failure_kind == "step_limit"
    assert not result.exhausted
    assert "max_steps_per_run" in (result.failure or "")


def test_exhaustive_step_cap_default_does_not_limit_normal_workloads() -> None:
    # The default cap is generous: an ordinary short workload is unaffected.
    def quick(proxy) -> None:
        for _ in range(3):
            if not proxy.report_and_wait(None, 0):
                return

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=2.0)
    result = coord.explore(
        worker_set=ThreadLauncher([quick]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=4,
    )
    assert result.ok, f"unexpected failure: {result.failure!r}"
