"""Unit tests for the backend-agnostic worker-launch seam.

``ThreadWorkerSet`` is the in-process (OS-thread) implementation of the
``WorkerSet`` port defined in ``frontrun._dpor_core.worker``. The planned
cross-process backend will supply a subprocess-based implementation of the
same Protocol; these tests pin the launch-and-join contract both must honour.
"""

from __future__ import annotations

import threading
import time

from frontrun._dpor_core.worker import WorkerSet, WorkerTarget
from frontrun._dpor_runtime.worker_set import ThreadWorkerSet


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
    ws.run(_targets([make(0), make(1), make(2)]), lambda t: t.func(), timeout=5.0)

    assert sorted(seen) == [0, 1, 2]


def test_run_one_receives_worker_id_and_args() -> None:
    received: dict[int, tuple] = {}
    lock = threading.Lock()

    targets = [WorkerTarget(worker_id=i, func=(lambda: None), args=(i, i * 10)) for i in range(3)]

    def run_one(t: WorkerTarget) -> None:
        with lock:
            received[t.worker_id] = t.args

    ThreadWorkerSet().run(targets, run_one, timeout=5.0)

    assert received == {0: (0, 0), 1: (1, 10), 2: (2, 20)}


def test_teardown_runs_after_join() -> None:
    order: list[str] = []
    lock = threading.Lock()

    def body() -> None:
        with lock:
            order.append("work")

    def teardown() -> None:
        order.append("teardown")

    ThreadWorkerSet().run(_targets([body, body]), lambda t: t.func(), timeout=5.0, teardown=teardown)

    assert order.count("work") == 2
    assert order[-1] == "teardown"


def test_on_timeout_called_with_alive_workers() -> None:
    release = threading.Event()
    timed_out: list[object] = []

    def slow() -> None:
        release.wait(timeout=5.0)

    def fast() -> None:
        pass

    def on_timeout(alive: list) -> None:
        timed_out.extend(alive)
        release.set()  # let the slow worker exit so the test cleans up

    ThreadWorkerSet().run(
        _targets([fast, slow]),
        lambda t: t.func(),
        timeout=0.3,
        on_timeout=on_timeout,
    )

    assert len(timed_out) == 1


def test_threads_exposed_for_inspection() -> None:
    store: list[threading.Thread] = []
    ws = ThreadWorkerSet(thread_store=store)
    ws.run(_targets([lambda: time.sleep(0)]), lambda t: t.func(), timeout=5.0)
    assert len(store) == 1
    assert ws.threads is store
