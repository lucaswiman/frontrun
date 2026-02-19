"""
Concurrency tests for tenacity using frontrun bytecode exploration.

Bug-finding tests target areas WITHOUT proper synchronization.
Safe-area tests target areas WITH locks or inherent thread-safety.

Repository: https://github.com/jd/tenacity (commit 0bdf1d9)
"""

import os
import signal
import sys
from contextlib import contextmanager

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_test_dir, "..", "external_repos", "tenacity")
sys.path.insert(0, os.path.abspath(_repo_root))

from frontrun.bytecode import explore_interleavings  # noqa: E402


@contextmanager
def timeout_minutes(minutes=10):
    def _handler(signum, frame):
        raise TimeoutError(f"Test timed out after {minutes} minute(s)")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(minutes * 60))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def print_result(name, result):
    if result.property_holds:
        print(f"  [{name}] SAFE: invariant held across {result.num_explored} interleavings")
    else:
        print(f"  [{name}] RACE FOUND after {result.num_explored} interleavings!")
    return result


# ===========================================================================
# BUG-FINDING TESTS (areas that should race)
# ===========================================================================


# --- B1: statistics["idle_for"] += sleep lost update ---
# Two threads sharing the same Retrying object's statistics dict.
# The `next_action` closure (line 423) does `self.statistics["idle_for"] += sleep`
# which is a non-atomic read-modify-write.

class StatsIdleForState:
    def __init__(self):
        self.stats = {"idle_for": 0.0, "attempt_number": 1, "start_time": 0.0}
        self.increments_per_thread = 5

    def thread1(self):
        for _ in range(self.increments_per_thread):
            self.stats["idle_for"] += 1.0

    def thread2(self):
        for _ in range(self.increments_per_thread):
            self.stats["idle_for"] += 1.0


def _idle_for_invariant(s):
    expected = s.increments_per_thread * 2 * 1.0
    return s.stats["idle_for"] == expected


def test_b1_stats_idle_for(max_attempts=500, max_ops=200):
    """statistics['idle_for'] += sleep: lost update race."""
    with timeout_minutes(5):
        return print_result("stats idle_for", explore_interleavings(
            setup=lambda: StatsIdleForState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_idle_for_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B2: statistics["attempt_number"] += 1 lost update ---

class StatsAttemptNumberState:
    def __init__(self):
        self.stats = {"idle_for": 0.0, "attempt_number": 1, "start_time": 0.0}
        self.increments_per_thread = 5

    def thread1(self):
        for _ in range(self.increments_per_thread):
            self.stats["attempt_number"] += 1

    def thread2(self):
        for _ in range(self.increments_per_thread):
            self.stats["attempt_number"] += 1


def _attempt_number_invariant(s):
    expected = 1 + s.increments_per_thread * 2
    return s.stats["attempt_number"] == expected


def test_b2_stats_attempt_number(max_attempts=500, max_ops=200):
    """statistics['attempt_number'] += 1: lost update race."""
    with timeout_minutes(5):
        return print_result("stats attempt_number", explore_interleavings(
            setup=lambda: StatsAttemptNumberState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_attempt_number_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B3: wrapped_f.statistics orphaned reference race ---
# In BaseRetrying.wraps(), `wrapped_f.statistics = copy.statistics` (line 330)
# is executed by each call. Two concurrent calls each overwrite .statistics.
# If thread1 reads .statistics AFTER its own call but BEFORE thread2's write,
# and thread2 reads AFTER its own call, they see different dict objects.

class WrappedFStatsRaceState:
    def __init__(self):
        from tenacity import Retrying, stop_after_attempt, wait_none

        self.retrying = Retrying(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            sleep=lambda x: None,
        )

        @self.retrying.wraps
        def my_func(x):
            return x * 2

        self.my_func = my_func
        self.stats_refs = [None, None]

    def thread1(self):
        self.my_func(1)
        self.stats_refs[0] = self.my_func.statistics

    def thread2(self):
        self.my_func(2)
        self.stats_refs[1] = self.my_func.statistics


def _wrapped_f_stats_race_invariant(s):
    ref0, ref1 = s.stats_refs
    if ref0 is None or ref1 is None:
        return True
    # In a race-free world, both captures would see the same dict.
    # With a race, thread1 captures before thread2 overwrites, so
    # they hold references to different dict objects.
    return ref0 is ref1


def test_b3_wrapped_f_stats(max_attempts=500, max_ops=300):
    """wrapped_f.statistics: concurrent calls produce orphaned references."""
    with timeout_minutes(5):
        return print_result("wrapped_f.statistics race", explore_interleavings(
            setup=lambda: WrappedFStatsRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_wrapped_f_stats_race_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B4: IterState shared mutation ---
# When a single Retrying object is used from multiple threads directly
# (not via wraps which copies), iter_state is on threading.local so it
# should be safe. But the statistics dict is also on threading.local.
# However, if someone shares a single *copy* across threads, the
# _local is the same object - let's test that.

class SharedCopyStatsState:
    def __init__(self):
        from tenacity import Retrying, stop_after_attempt, retry_if_result, wait_none

        self.retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_none(),
            retry=retry_if_result(lambda r: r is None),
            sleep=lambda x: None,
        )
        # Create a single copy shared by both threads
        self.shared_copy = self.retrying.copy()
        self.results = [None, None]

    def thread1(self):
        # Using the shared copy directly
        self.results[0] = self.shared_copy(lambda: 42)

    def thread2(self):
        self.results[1] = self.shared_copy(lambda: 99)


def _shared_copy_invariant(s):
    # Both should succeed since neither returns None
    if s.results[0] != 42 or s.results[1] != 99:
        return False
    # The statistics should reflect the LAST call's attempt_number=1
    # With the race, attempt_number could be corrupted
    stats = s.shared_copy.statistics
    return stats.get("attempt_number", 0) >= 1


def test_b4_shared_copy_stats(max_attempts=500, max_ops=300):
    """Shared Retrying copy: statistics corruption from concurrent __call__."""
    with timeout_minutes(5):
        return print_result("shared copy stats", explore_interleavings(
            setup=lambda: SharedCopyStatsState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_shared_copy_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B5: RetryCallState.idle_for += sleep lost update ---
# If two threads somehow share a RetryCallState (unusual but possible
# with custom callback patterns), idle_for += is non-atomic.

class IdleForLostUpdateState:
    def __init__(self):
        from tenacity import RetryCallState, Retrying, stop_after_attempt, wait_none
        self.retrying = Retrying(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            sleep=lambda x: None,
        )
        self.rcs = RetryCallState(
            retry_object=self.retrying, fn=None, args=(), kwargs={}
        )
        self.increments_per_thread = 5

    def thread1(self):
        for _ in range(self.increments_per_thread):
            self.rcs.idle_for += 1.0

    def thread2(self):
        for _ in range(self.increments_per_thread):
            self.rcs.idle_for += 1.0


def _idle_for_lost_update_invariant(s):
    expected = s.increments_per_thread * 2 * 1.0
    return s.rcs.idle_for == expected


def test_b5_idle_for_lost_update(max_attempts=500, max_ops=200):
    """RetryCallState.idle_for += sleep: lost update."""
    with timeout_minutes(5):
        return print_result("idle_for lost update", explore_interleavings(
            setup=lambda: IdleForLostUpdateState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_idle_for_lost_update_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B6: RetryCallState.attempt_number += 1 lost update ---

class AttemptNumberLostUpdateState:
    def __init__(self):
        from tenacity import RetryCallState, Retrying, stop_after_attempt, wait_none
        self.retrying = Retrying(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            sleep=lambda x: None,
        )
        self.rcs = RetryCallState(
            retry_object=self.retrying, fn=None, args=(), kwargs={}
        )
        self.increments_per_thread = 5

    def thread1(self):
        for _ in range(self.increments_per_thread):
            self.rcs.attempt_number += 1

    def thread2(self):
        for _ in range(self.increments_per_thread):
            self.rcs.attempt_number += 1


def _attempt_number_lost_update_invariant(s):
    expected = 1 + s.increments_per_thread * 2
    return s.rcs.attempt_number == expected


def test_b6_attempt_number_lost_update(max_attempts=500, max_ops=200):
    """RetryCallState.attempt_number += 1: lost update."""
    with timeout_minutes(5):
        return print_result("attempt_number lost update", explore_interleavings(
            setup=lambda: AttemptNumberLostUpdateState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_attempt_number_lost_update_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# ===========================================================================
# SAFE-AREA TESTS (areas that should NOT race)
# ===========================================================================


# --- S1: threading.local isolates statistics per thread ---
# BaseRetrying uses threading.local() for statistics, so each thread
# should have its own independent copy.

class ThreadLocalStatsState:
    def __init__(self):
        from tenacity import Retrying, stop_after_attempt, wait_none

        self.retrying = Retrying(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            sleep=lambda x: None,
        )
        self.t1_stats = None
        self.t2_stats = None

    def thread1(self):
        # Access statistics through the property (uses threading.local)
        self.retrying.begin()
        self.retrying.statistics["attempt_number"] += 1
        self.retrying.statistics["attempt_number"] += 1
        self.t1_stats = dict(self.retrying.statistics)

    def thread2(self):
        self.retrying.begin()
        self.retrying.statistics["attempt_number"] += 1
        self.t2_stats = dict(self.retrying.statistics)


def _thread_local_stats_invariant(s):
    # Each thread should have seen its own independent statistics
    if s.t1_stats is None or s.t2_stats is None:
        return True
    # Thread 1 did begin (sets to 1) then += 1 twice = 3
    # Thread 2 did begin (sets to 1) then += 1 once = 2
    return s.t1_stats["attempt_number"] == 3 and s.t2_stats["attempt_number"] == 2


def test_s1_thread_local_stats(max_attempts=20000, max_ops=400):
    """threading.local: statistics are isolated per thread."""
    with timeout_minutes(5):
        return print_result("thread_local stats", explore_interleavings(
            setup=lambda: ThreadLocalStatsState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_thread_local_stats_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- S2: Per-call RetryCallState isolation ---
# Each call to Retrying.__call__() creates its own RetryCallState,
# so concurrent calls should not interfere.

class PerCallStateState:
    def __init__(self):
        from tenacity import Retrying, stop_after_attempt, wait_none

        self.retrying = Retrying(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            sleep=lambda x: None,
        )
        self.results = [None, None]

    def thread1(self):
        # wraps creates a copy, so each call gets its own Retrying instance
        result = self.retrying.copy()(lambda: 42)
        self.results[0] = result

    def thread2(self):
        result = self.retrying.copy()(lambda: 99)
        self.results[1] = result


def _per_call_state_invariant(s):
    return s.results[0] == 42 and s.results[1] == 99


def test_s2_per_call_state(max_attempts=20000, max_ops=500):
    """Per-call RetryCallState: each call is isolated."""
    with timeout_minutes(5):
        return print_result("per-call state", explore_interleavings(
            setup=lambda: PerCallStateState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_per_call_state_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- S3: Immutable strategy objects are thread-safe ---
# stop_after_attempt, wait_fixed etc. are immutable objects that just
# compute based on RetryCallState. Sharing them is safe.

class ImmutableStrategyState:
    def __init__(self):
        from tenacity import stop_after_attempt, wait_none, retry_if_result, RetryCallState, Retrying

        self.stop = stop_after_attempt(3)
        self.wait = wait_none()
        self.retry = retry_if_result(lambda r: r is None)

        retrying = Retrying(
            stop=self.stop,
            wait=self.wait,
            sleep=lambda x: None,
        )
        self.rcs1 = RetryCallState(retry_object=retrying, fn=None, args=(), kwargs={})
        self.rcs2 = RetryCallState(retry_object=retrying, fn=None, args=(), kwargs={})
        self.rcs1.attempt_number = 2
        self.rcs2.attempt_number = 4
        self.t1_stop = None
        self.t2_stop = None

    def thread1(self):
        self.t1_stop = self.stop(self.rcs1)

    def thread2(self):
        self.t2_stop = self.stop(self.rcs2)


def _immutable_strategy_invariant(s):
    # stop_after_attempt(3): attempt 2 -> False, attempt 4 -> True
    return s.t1_stop is False and s.t2_stop is True


def test_s3_immutable_strategy(max_attempts=20000, max_ops=400):
    """Immutable strategies: concurrent evaluation is safe."""
    with timeout_minutes(5):
        return print_result("immutable strategy", explore_interleavings(
            setup=lambda: ImmutableStrategyState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_immutable_strategy_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- S4: copy() creates independent instances ---
# Each copy() should be fully independent, sharing no mutable state.

class CopyIsolationState:
    def __init__(self):
        from tenacity import Retrying, stop_after_attempt, wait_none

        self.original = Retrying(
            stop=stop_after_attempt(1),
            wait=wait_none(),
            sleep=lambda x: None,
        )
        self.results = [None, None]

    def thread1(self):
        copy1 = self.original.copy()
        self.results[0] = copy1(lambda: "hello")

    def thread2(self):
        copy2 = self.original.copy()
        self.results[1] = copy2(lambda: "world")


def _copy_isolation_invariant(s):
    return s.results[0] == "hello" and s.results[1] == "world"


def test_s4_copy_isolation(max_attempts=20000, max_ops=500):
    """copy(): independent instances don't interfere."""
    with timeout_minutes(5):
        return print_result("copy isolation", explore_interleavings(
            setup=lambda: CopyIsolationState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_copy_isolation_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# ===========================================================================
# Main runner
# ===========================================================================

BUG_TESTS = [
    ("B1: stats idle_for lost update", test_b1_stats_idle_for, 500, 200),
    ("B2: stats attempt_number lost update", test_b2_stats_attempt_number, 500, 200),
    ("B3: wrapped_f.statistics race", test_b3_wrapped_f_stats, 500, 300),
    ("B4: shared copy stats corruption", test_b4_shared_copy_stats, 500, 300),
    ("B5: idle_for lost update", test_b5_idle_for_lost_update, 500, 200),
    ("B6: attempt_number lost update", test_b6_attempt_number_lost_update, 500, 200),
]

SAFE_TESTS = [
    ("S1: thread_local stats isolation", test_s1_thread_local_stats, 20000, 400),
    ("S2: per-call state isolation", test_s2_per_call_state, 20000, 500),
    ("S3: immutable strategy safety", test_s3_immutable_strategy, 20000, 400),
    ("S4: copy isolation", test_s4_copy_isolation, 20000, 500),
]

ALL_TESTS = BUG_TESTS + SAFE_TESTS

if __name__ == "__main__":
    import time as _time

    print("=" * 72)
    print("tenacity Concurrency Tests via Frontrun")
    print("=" * 72)

    for name, fn, attempts, ops in ALL_TESTS:
        print(f"\n--- {name} ---")
        t0 = _time.monotonic()
        try:
            fn(max_attempts=attempts, max_ops=ops)
            print(f"  Time: {_time.monotonic() - t0:.1f}s")
        except Exception as e:
            print(f"  ERROR ({_time.monotonic() - t0:.1f}s): {e}")
