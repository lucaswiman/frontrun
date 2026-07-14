"""Finding 7: cooperative Lock/RLock must honor acquire(timeout=...).

Classic timeout-based deadlock-avoidance pattern:

    acquire A; try acquire B with a timeout; if it fails, release A and retry.

This program *cannot* deadlock, because a timed acquire gives up.  The
cooperative locks previously ignored the timeout in the managed slow path
(no deadline; the WaitForGraph wait edge was registered unconditionally), so
the transient A/B/A/B cycle was reported as a fatal DeadlockError.

After the fix:
  * a timed acquire returns False after its deadline (and removes its wait
    edge), and
  * a timed waiter's transient cycle does not raise DeadlockError.
"""

import threading

import frontrun


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
