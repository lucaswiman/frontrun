"""
Real-code exploration: threadpoolctl _get_libc() TOCTOU.

Runs frontrun's bytecode exploration directly against the real
ThreadpoolController._get_libc() classmethod to find
the TOCTOU where two threads racing to cache libc both create
separate ctypes.CDLL objects.

The bug: _get_libc() reads _system_libraries.get("libc"), checks if
None, then creates a CDLL and stores it.  Two threads can both see
None, both create expensive CDLL objects, and the second overwrites
the first.  The returned libc objects are different Python objects.

Note: This test requires a Unix system where find_library("c") works.
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "threadpoolctl"))

from external_tests_helpers import print_exploration_result, print_seed_sweep_results
from threadpoolctl import ThreadpoolController

from frontrun.bytecode import explore_interleavings, run_with_schedule


class RealGetLibcState:
    def __init__(self):
        # Clear the class-level cache so both threads must create
        ThreadpoolController._system_libraries.clear()
        self.libc_1 = None
        self.libc_2 = None

    def thread1(self):
        self.libc_1 = ThreadpoolController._get_libc()

    def thread2(self):
        self.libc_2 = ThreadpoolController._get_libc()


def test_real_threadpoolctl_get_libc_toctou():
    """Find the _get_libc() TOCTOU in real threadpoolctl."""

    result = explore_interleavings(
        setup=lambda: RealGetLibcState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # Both threads should get the same cached object.
            # If the TOCTOU fires, they get different CDLL instances.
            s.libc_1 is s.libc_2
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


def test_real_threadpoolctl_get_libc_sweep():
    """Sweep seeds to measure how easy the bug is to find."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: RealGetLibcState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: s.libc_1 is s.libc_2,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_threadpoolctl_reproduce():
    """Find a counterexample and reproduce it 10 times."""

    result = explore_interleavings(
        setup=lambda: RealGetLibcState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.libc_1 is s.libc_2,
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
            setup=lambda: RealGetLibcState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
        )
        is_bug = state.libc_1 is not state.libc_2
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        same = "same" if state.libc_1 is state.libc_2 else "DIFFERENT"
        print(f"  Run {i + 1}: libc objects {same} [{status}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 200 attempts) ===")
    test_real_threadpoolctl_get_libc_toctou()

    print("\n=== Seed sweep (20 seeds × 100 attempts) ===")
    test_real_threadpoolctl_get_libc_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_threadpoolctl_reproduce()
