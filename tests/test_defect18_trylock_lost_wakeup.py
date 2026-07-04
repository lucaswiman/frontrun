"""Defect #18: CooperativeLock masked lost-wakeup races behind acquire(blocking=False).

A failed non-blocking acquire (trylock) is observable behavior: the caller
takes a different branch instead of waiting.  The classic victim is the
"producer assumes the lock holder will drain the queue" pattern used by
python-statemachine's ``SyncEngine.processing_loop()``:

    q.append(x)
    if not lock.acquire(blocking=False):
        return                      # assume holder will process x
    while q:
        process(q.popleft())
    lock.release()                  # <- x enqueued after final check is LOST

With a real C-level lock, DPOR sees every acquire/release as an access on
the lock object and finds the race.  ``CooperativeLock`` reported nothing
for a failed trylock and never explored the (release, attempt) reversal, so
the same program under the pytest plugin's ``patch_locks()`` (which any
library doing ``from threading import Lock`` at import time picks up) went
GREEN — a false negative that caused python-statemachine to be mis-triaged
as clean.

The fix reports ``lock_attempt_ok`` / ``lock_attempt_fail`` sync events for
non-blocking acquires and adds trylock-aware release backtracking in the
Rust engine: for each release of lock L, any other thread's trylock of L
later in the trace gets a wakeup inserted just before the release, so the
failure branch is explored.

These tests assert that the real lock and the cooperative lock now AGREE:
both find the lost-wakeup race and reproduce it deterministically.
"""

import _thread
import threading
from collections import deque

import frontrun
from frontrun._cooperative import CooperativeLock


class MiniEngine:
    """Minimal model of statemachine's SyncEngine.processing_loop()."""

    def __init__(self, lock) -> None:
        self._lock = lock
        self.q = deque()
        self.processed = []

    def send(self, item) -> None:
        self.q.append(item)
        if not self._lock.acquire(blocking=False):
            return  # assume the current holder will drain our item
        try:
            while self.q:
                self.processed.append(self.q.popleft())
        finally:
            self._lock.release()


def _make_state(lock_factory):
    class State:
        def __init__(self):
            self.engine = MiniEngine(lock_factory())

    return State


def _worker_a(s):
    s.engine.send("a")


def _worker_b(s):
    s.engine.send("b")


def _invariant(s):
    # Every enqueued event must eventually be processed.
    return len(s.engine.processed) == 2


def _explore(lock_factory):
    return frontrun.explore(
        setup=_make_state(lock_factory),
        workers=[_worker_a, _worker_b],
        invariant=_invariant,
        detect_io=False,
        reproduce_on_failure=10,
    )


def test_real_lock_finds_lost_wakeup():
    """Baseline: with a raw C-level lock the race is found and replayed."""
    result = _explore(_thread.allocate_lock)
    assert not result.property_holds, "DPOR missed the lost-wakeup race with a real lock"
    assert result.reproduction_successes >= 8, (
        f"lost-wakeup counterexample replayed only "
        f"{result.reproduction_successes}/{result.reproduction_attempts}"
    )


def test_cooperative_lock_finds_lost_wakeup():
    """Regression: CooperativeLock must not mask the trylock lost-wakeup race."""
    result = _explore(CooperativeLock)
    assert not result.property_holds, (
        "CooperativeLock masked the lost-wakeup race (defect #18 false negative): "
        f"explored {result.num_explored} interleavings, all green"
    )
    assert result.reproduction_successes >= 8, (
        f"lost-wakeup counterexample replayed only "
        f"{result.reproduction_successes}/{result.reproduction_attempts}"
    )


def test_threading_lock_agrees_with_real_lock():
    """Hermeticity: whatever threading.Lock currently is (the pytest plugin
    patches it to CooperativeLock at configure time), explore() must reach
    the same verdict as with the real lock."""
    result = _explore(threading.Lock)
    assert not result.property_holds, (
        f"threading.Lock ({type(threading.Lock()).__name__}) masked the lost-wakeup "
        "race that the real C-level lock finds"
    )
    assert result.reproduction_successes >= 8


def test_trylock_success_branch_still_green_when_correct():
    """A correct trylock consumer (retries until acquired) must stay green."""

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
    assert result.property_holds, result.explanation
