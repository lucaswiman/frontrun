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

import socket
import threading
import time

import pytest

from frontrun._dpor_runtime.xproc.coordinator import CrossProcessCoordinator
from frontrun._dpor_runtime.xproc.dpor_coordinator import (
    DporCrossProcessCoordinator,
    _accept_hello_before_total_deadline,
    _TotalTimeoutExpiredError,
)
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher


def _join_worker_threads() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith("xproc-worker-"):
            thread.join(timeout=10.0)


def test_total_timeout_shares_budget_between_connect_and_hello(tmp_path) -> None:
    socket_path = str(tmp_path / "xproc.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(1)
    listener.settimeout(1.0)
    release = threading.Event()

    def connect_without_hello() -> None:
        time.sleep(0.2)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        try:
            release.wait(1.0)
        finally:
            client.close()

    client_thread = threading.Thread(target=connect_without_hello)
    client_thread.start()
    start = time.monotonic()
    total_timeout = 0.3
    try:
        with pytest.raises(_TotalTimeoutExpiredError):
            _accept_hello_before_total_deadline(
                listener,
                object(),  # type: ignore[arg-type] - liveness is irrelevant to this socket-level regression
                [],
                connect_budget=1.0,
                total_deadline=start + total_timeout,
                total_timeout=total_timeout,
            )
    finally:
        release.set()
        client_thread.join(timeout=2.0)
        listener.close()

    elapsed = time.monotonic() - start
    assert elapsed < 0.42, f"connect + HELLO consumed separate timeout budgets ({elapsed:.3f}s)"


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


def test_dpor_total_timeout_bounds_a_single_long_execution() -> None:
    # total_timeout is a documented search bound, but it was only checked
    # between executions (dpor_exploration_iter's loop top). A single long
    # execution whose worker keeps sending frames defeats both liveness guards
    # (each frame resets the recv timeout AND bumps the relay heartbeat), so
    # explore() overran total_timeout by orders of magnitude — until the
    # engine's max_branches step cap, not the user's time budget, ended it.
    steps, step_time = 1500, 0.01  # ~15s of healthy frames if never aborted

    def chatty(proxy) -> None:
        for _ in range(steps):
            if not proxy.report_and_wait(None, 0):
                return
            time.sleep(step_time)

    coord = DporCrossProcessCoordinator(num_workers=1, deadlock_timeout=2.0, total_timeout=0.5)
    start = time.monotonic()
    try:
        result = coord.explore(
            worker_set=ThreadLauncher([chatty]),
            setup=lambda: None,
            invariant=lambda: True,
        )
    finally:
        _join_worker_threads()
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"total_timeout=0.5 was not honored mid-execution (returned after {elapsed:.1f}s)"
    # The in-flight execution was truncated, so the search must not claim
    # exhaustion — and nothing failed, so the truncated run is not a failure.
    assert result.exhausted is False
    assert result.ok, f"truncation misreported as failure: {result.failure!r} ({result.failure_kind})"


class _NeverConnectingWorkerSet:
    """Launcher whose handles stay alive but never connect to the coordinator."""

    def launch(self, targets):  # noqa: ARG002
        return [object()]

    def join(self, handles, timeout):  # noqa: ARG002
        return []


def test_dpor_total_timeout_bounds_initial_worker_connection() -> None:
    # total_timeout is an end-to-end search bound, including the initial
    # process launch / HELLO handshake.  The coordinator previously passed its
    # much larger connect budget to accept_hello_live(), so a worker that never
    # connected overran total_timeout before the first execution even began.
    coord = DporCrossProcessCoordinator(num_workers=1, deadlock_timeout=0.1, total_timeout=0.02)
    coord._connect_budget = 0.3
    start = time.monotonic()
    result = coord.explore(
        worker_set=_NeverConnectingWorkerSet(),
        setup=lambda: None,
        invariant=lambda: True,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 0.15, f"total_timeout=0.02 was not honored during worker connection ({elapsed:.3f}s)"
    assert result.ok
    assert result.iterations == 0
    assert result.exhausted is False


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


class _ProbeThreadLauncher(ThreadLauncher):
    """ThreadLauncher plus a scripted LivenessProbe.

    ``state['detail']`` (set from a worker body mid-run) makes the fleet-wide
    probe report an abnormal process exit, exactly as MpLauncher /
    SubprocessLauncher do when one child dies without closing its socket.
    """

    def __init__(self, bodies, state: dict) -> None:
        super().__init__(bodies)
        self._state = state

    def any_exited(self, handles) -> bool:  # noqa: ARG002 - fleet-wide fake
        return self._state.get("detail") is not None

    def all_exited(self, handles) -> bool:  # noqa: ARG002
        return False

    def diagnose(self, handles) -> str | None:  # noqa: ARG002
        return self._state.get("detail")


def test_exhaustive_timeout_diagnosis_does_not_blame_the_slow_worker_for_anothers_exit() -> None:
    # Regression: any_exited()/diagnose() are fleet-wide, but the timeout-
    # diagnosis loop recorded the exit under the *timed-out* worker's id, so
    # the top line read "worker 0 failed: worker process exited during the run
    # (worker 1: ...)" — claiming the merely-slow worker 0 exited when the
    # embedded detail names worker 1 as the process that actually died.
    release = threading.Event()
    probe_state: dict = {}

    def slow(proxy) -> None:
        if not proxy.report_and_wait(None, 0):
            return
        # Simulate a sibling process dying abnormally mid-run, then overrun
        # deadlock_timeout ourselves.
        probe_state["detail"] = "worker 1: ModuleNotFoundError: No module named 'missing_dep'"
        release.wait(5.0)
        proxy.report_and_wait(None, 0)

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=1.0)
    try:
        result = coord.explore(
            worker_set=_ProbeThreadLauncher([slow], probe_state),
            setup=lambda: None,
            invariant=lambda: True,
            max_iterations=2,
        )
    finally:
        release.set()
        _join_worker_threads()
    assert not result.ok
    assert result.failure_kind == "worker_error"
    failure = result.failure or ""
    # The real culprit from diagnose() must be preserved...
    assert "worker 1: ModuleNotFoundError" in failure
    # ...and the timed-out worker must not be claimed to have exited.
    assert "worker 0 failed: worker process exited" not in failure, failure


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


def test_exhaustive_step_cap_accepts_run_finishing_at_exact_limit() -> None:
    """The cap bounds additional steps, not clean completion after the last one."""

    def exactly_three_steps(proxy) -> None:
        for _ in range(3):
            assert proxy.report_and_wait(None, 0)

    result = CrossProcessCoordinator(num_workers=1, deadlock_timeout=2.0, max_steps_per_run=3).explore(
        worker_set=ThreadLauncher([exactly_three_steps]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=1,
    )

    assert result.ok, f"finite run at the exact step limit was rejected: {result.failure!r}"
    assert result.failure_kind is None
