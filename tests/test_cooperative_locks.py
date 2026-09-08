"""Cooperative lock acquisition contracts and DPOR scheduling regressions."""

from __future__ import annotations

import _thread
import math
import sysconfig
import threading
from collections import deque
from collections.abc import Callable

import pytest

import frontrun
from frontrun._cooperative import CooperativeLock, CooperativeRLock

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
SEARCHES = ["bit-reversal:42", "stride", "stride:3", "conflict-first"]


@pytest.mark.parametrize("lock_factory", [CooperativeLock, CooperativeRLock])
@pytest.mark.parametrize(
    "acquire",
    [
        lambda lock: lock.acquire(blocking=False, timeout=0),
        lambda lock: lock.acquire(timeout=-2),
    ],
)
def test_invalid_acquire_argument_combinations_raise(
    lock_factory: Callable[[], CooperativeLock | CooperativeRLock],
    acquire: Callable[[CooperativeLock | CooperativeRLock], bool],
) -> None:
    """Cooperative wrappers reject every combination rejected by CPython."""
    with pytest.raises(ValueError):
        acquire(lock_factory())


@pytest.mark.parametrize("lock_factory", [CooperativeLock, CooperativeRLock])
def test_default_timeout_is_valid_for_nonblocking_lock_acquire(
    lock_factory: Callable[[], CooperativeLock | CooperativeRLock],
) -> None:
    """CPython accepts its -1 timeout sentinel with blocking=False."""
    assert lock_factory().acquire(blocking=False, timeout=-1)


def test_semaphore_rejects_nonblocking_timeout() -> None:
    from frontrun._cooperative import CooperativeSemaphore

    with pytest.raises(ValueError):
        CooperativeSemaphore().acquire(blocking=False, timeout=0)


class _TwoLocks:
    def __init__(self) -> None:
        self.lock_a = threading.Lock()
        self.lock_b = threading.Lock()
        self.completed = 0


def _worker(state: _TwoLocks, first, second) -> None:
    # Retry loop with timeout-based deadlock avoidance.
    for _ in range(50):
        first.acquire()
        try:
            if second.acquire(timeout=0.05):
                try:
                    state.completed += 1
                finally:
                    second.release()
                return
        finally:
            first.release()


def test_timeout_acquire_does_not_falsely_deadlock():
    """The retry-with-timeout pattern must not be reported as a deadlock."""
    result = frontrun.explore_random(
        setup=_TwoLocks,
        threads=[
            lambda s: _worker(s, s.lock_a, s.lock_b),
            lambda s: _worker(s, s.lock_b, s.lock_a),
        ],
        invariant=lambda s: True,
        max_attempts=60,
        max_ops=4000,
        seed=7,
        deadlock_timeout=2.0,
    )

    assert result.property_holds, (
        f"Timeout-based deadlock-avoidance code was falsely reported as a deadlock: {result.explanation}"
    )


class MiniEngine:
    """Minimal model of statemachine's SyncEngine.processing_loop()."""

    def __init__(self, lock, acquire_kwargs=None) -> None:
        self._lock = lock
        self._acquire_kwargs = {"blocking": False} if acquire_kwargs is None else acquire_kwargs
        self.q = deque()
        self.processed = []

    def send(self, item) -> None:
        self.q.append(item)
        if not self._lock.acquire(**self._acquire_kwargs):
            return
        try:
            while self.q:
                self.processed.append(self.q.popleft())
        finally:
            self._lock.release()


def _make_state(lock_factory, acquire_kwargs=None):
    class State:
        def __init__(self):
            self.engine = MiniEngine(lock_factory(), acquire_kwargs)

    return State


def _worker_a(s):
    s.engine.send("a")


def _worker_b(s):
    s.engine.send("b")


def _invariant(s):
    # Every enqueued event must eventually be processed.
    return len(s.engine.processed) == 2


def _explore(lock_factory, acquire_kwargs=None):
    return frontrun.explore(
        setup=_make_state(lock_factory, acquire_kwargs),
        workers=[_worker_a, _worker_b],
        invariant=_invariant,
        detect_io=False,
        reproduce_on_failure=10,
    )


@pytest.mark.parametrize(
    ("lock_factory", "acquire_kwargs"),
    [
        pytest.param(_thread.allocate_lock, {"blocking": False}, id="native-trylock"),
        pytest.param(CooperativeLock, {"blocking": False}, id="cooperative-trylock"),
        pytest.param(threading.Lock, {"blocking": False}, id="patched-threading-trylock"),
        pytest.param(CooperativeLock, {"timeout": 0}, id="lock-zero-timeout"),
        pytest.param(CooperativeRLock, {"timeout": 0}, id="rlock-zero-timeout"),
    ],
)
def test_trylock_finds_and_replays_lost_wakeup(lock_factory, acquire_kwargs):
    # A failed trylock must remain observable to DPOR (defect #18).
    result = _explore(lock_factory, acquire_kwargs)
    assert not result.property_holds, "DPOR missed the queue-drainer lost wakeup"
    assert result.reproduction_successes >= 8


def test_trylock_retry_timeout_is_inconclusive_not_false_proof():
    """A scheduler-starved busy retry must not be certified as a proof."""

    class State:
        def __init__(self):
            self.lock = threading.Lock()
            self.count = 0

    def worker(s):
        while not s.lock.acquire(blocking=False):
            pass
        try:
            s.count += 1
        finally:
            s.lock.release()

    result = frontrun.explore(
        setup=State,
        workers=[worker, worker],
        invariant=lambda s: s.count == 2,
        detect_io=False,
        reproduce_on_failure=10,
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "inconclusive" in result.explanation.lower()


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
