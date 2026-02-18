"""
Real-code exploration: TPool _should_keep_going() TOCTOU.

Runs frontrun's bytecode exploration directly against the real WildPool
class to find the TOCTOU where the worker exits while
tasks remain in the queue.

The bug: _should_keep_going() reads keep_going under worker_lock, then
checks _join_is_called and bench.empty() under join_lock.  Between the
two lock acquisitions, the main thread can enqueue work and set the join
flag.  The worker sees an empty queue and exits, leaving tasks unprocessed.
"""

import os
import sys
import threading

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "TPool", "src"))

from external_tests_helpers import print_exploration_result, print_seed_sweep_results
from TPool import WildPool

from frontrun.bytecode import explore_interleavings, run_with_schedule


class RealTPoolState:
    """Wraps a real WildPool instance for frontrun testing.

    The bug: _should_keep_going() reads keep_going under worker_lock,
    then checks _join_is_called and bench.empty() under join_lock.
    Between the two lock acquisitions, the main thread can enqueue
    a task and set _join_is_called=True.  The worker sees an empty
    queue and decides to stop, even though a task was just added.
    """

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        # Pre-set the worker as if it were running
        self.pool.keep_going = True
        self.worker_decided_to_stop = False
        self.items_in_queue_after = 0

    def worker_checks(self):
        """The worker thread calls _should_keep_going()."""
        result = self.pool._should_keep_going()
        if not result:
            self.worker_decided_to_stop = True
            self.items_in_queue_after = self.pool.bench.qsize()

    def main_thread_enqueues_and_joins(self):
        """Main thread: set join flag and enqueue a task."""
        # This mirrors what happens in join(): set flag, add sentinel
        with self.pool.join_lock:
            self.pool._join_is_called = True
        # Then add a real task (someone calls add_thread after join)
        dummy = threading.Thread(target=lambda: None)
        self.pool.add_thread(dummy)


def test_real_tpool_toctou_explore():
    """Find the TOCTOU in real WildPool._should_keep_going()."""

    result = explore_interleavings(
        setup=lambda: RealTPoolState(),
        threads=[
            lambda s: s.worker_checks(),
            lambda s: s.main_thread_enqueues_and_joins(),
        ],
        invariant=lambda s: (
            # If worker decided to stop, queue should be empty
            not s.worker_decided_to_stop or s.items_in_queue_after == 0
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


def test_real_tpool_toctou_sweep_seeds():
    """Sweep seeds to measure how easy the bug is to find."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: RealTPoolState(),
            threads=[
                lambda s: s.worker_checks(),
                lambda s: s.main_thread_enqueues_and_joins(),
            ],
            invariant=lambda s: not s.worker_decided_to_stop or s.items_in_queue_after == 0,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_tpool_reproduce():
    """Find a counterexample and reproduce it 10 times."""

    result = explore_interleavings(
        setup=lambda: RealTPoolState(),
        threads=[
            lambda s: s.worker_checks(),
            lambda s: s.main_thread_enqueues_and_joins(),
        ],
        invariant=lambda s: not s.worker_decided_to_stop or s.items_in_queue_after == 0,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    if not result.counterexample:
        print("No counterexample found — skipping reproduction")
        return

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: RealTPoolState(),
            threads=[
                lambda s: s.worker_checks(),
                lambda s: s.main_thread_enqueues_and_joins(),
            ],
        )
        is_bug = state.worker_decided_to_stop and state.items_in_queue_after > 0
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        print(
            f"  Run {i + 1}: worker_stopped={state.worker_decided_to_stop}, "
            f"queue_size={state.items_in_queue_after} [{status}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 200 attempts) ===")
    test_real_tpool_toctou_explore()

    print("\n=== Seed sweep (20 seeds × 100 attempts) ===")
    test_real_tpool_toctou_sweep_seeds()

    print("\n=== Deterministic reproduction ===")
    test_real_tpool_reproduce()
