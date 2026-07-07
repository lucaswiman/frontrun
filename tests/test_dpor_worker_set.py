"""Unit tests for the backend-agnostic worker-launch seam.

``ThreadWorkerSet`` is the in-process (OS-thread) implementation of the
``WorkerSet`` port defined in ``frontrun._dpor_core.worker``. Cross-process
launchers implement the same launch-and-join contract with process handles.
"""

from __future__ import annotations

import threading
import time

import pytest

from frontrun._dpor_core.worker import WorkerSet, WorkerTarget
from frontrun._dpor_runtime.worker_set import ThreadWorkerSet
from frontrun._threaded_runner import _POST_TIMEOUT_CLEANUP_JOIN_SECONDS


def _targets(funcs):
    return [WorkerTarget(worker_id=i, func=f, args=()) for i, f in enumerate(funcs)]


def test_thread_worker_set_satisfies_protocol() -> None:
    assert isinstance(ThreadWorkerSet(), WorkerSet)


def test_runs_every_target_to_completion() -> None:
    seen: list[int] = []
    lock = threading.Lock()

    def make(n: int):
        def body() -> None:
            with lock:
                seen.append(n)

        return body

    ws = ThreadWorkerSet(name_prefix="test")
    ws.run(_targets([make(0), make(1), make(2)]), timeout=5.0)

    assert sorted(seen) == [0, 1, 2]


def test_launch_and_join_return_live_handles() -> None:
    release = threading.Event()
    ws = ThreadWorkerSet()
    handles = ws.launch(_targets([lambda: release.wait(timeout=5.0)]))
    alive = ws.join(handles, timeout=0.1)
    release.set()
    for thread in alive:
        thread.join(timeout=2.0)
    assert len(alive) == 1


def test_target_receives_args() -> None:
    received: dict[int, tuple] = {}
    lock = threading.Lock()

    def record(worker_id: int, value: int) -> None:
        with lock:
            received[worker_id] = (worker_id, value)

    targets = [WorkerTarget(worker_id=i, func=record, args=(i, i * 10)) for i in range(3)]
    ThreadWorkerSet().run(targets, timeout=5.0)

    assert received == {0: (0, 0), 1: (1, 10), 2: (2, 20)}


def test_teardown_runs_after_join() -> None:
    order: list[str] = []
    lock = threading.Lock()

    def body() -> None:
        with lock:
            order.append("work")

    def teardown() -> None:
        order.append("teardown")

    ThreadWorkerSet().run(_targets([body, body]), timeout=5.0, teardown=teardown)

    assert order.count("work") == 2
    assert order[-1] == "teardown"


def test_on_timeout_called_with_alive_workers() -> None:
    release = threading.Event()
    timed_out: list[object] = []
    join_timeouts: list[float] = []

    class SpyWorkerSet(ThreadWorkerSet):
        def join(self, handles: object, timeout: float) -> list[threading.Thread]:
            join_timeouts.append(timeout)
            return super().join(handles, timeout)

    def slow() -> None:
        release.wait(timeout=5.0)
        time.sleep(0.05)

    def fast() -> None:
        pass

    def on_timeout(alive: list) -> None:
        timed_out.extend(alive)
        release.set()  # let the slow worker exit so the test cleans up

    SpyWorkerSet().run(
        _targets([fast, slow]),
        timeout=0.3,
        on_timeout=on_timeout,
    )

    assert len(timed_out) == 1
    assert join_timeouts == [0.3, _POST_TIMEOUT_CLEANUP_JOIN_SECONDS]
    assert not timed_out[0].is_alive()


def test_threads_exposed_for_inspection() -> None:
    store: list[threading.Thread] = []
    ws = ThreadWorkerSet(thread_store=store)
    ws.run(_targets([lambda: time.sleep(0)]), timeout=5.0)
    assert len(store) == 1
    assert ws.threads is store


def test_run_invokes_teardown_when_launch_fails() -> None:
    # teardown is the runner's opcode-tracer uninstall.  If launch() raises
    # (e.g. thread.start() hits "can't start new thread" under exhaustion, or a
    # target has func=None), teardown must still run — otherwise the
    # process-wide sys.settrace/sys.monitoring tracer leaks and mis-traces
    # every subsequent execution.
    torn_down: list[bool] = []
    bad_targets = [WorkerTarget(worker_id=0, func=None, args=())]

    with pytest.raises(TypeError):
        ThreadWorkerSet().run(
            bad_targets,
            timeout=1.0,
            teardown=lambda: torn_down.append(True),
        )

    assert torn_down == [True], "teardown must run even when launch() fails"
