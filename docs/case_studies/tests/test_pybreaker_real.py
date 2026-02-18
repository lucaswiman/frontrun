"""
Real-code exploration: pybreaker CircuitMemoryStorage.increment_counter() lost update.

Runs frontrun's bytecode exploration directly against the real
CircuitMemoryStorage class to find the fail_counter lost update race.

The bug: CircuitMemoryStorage.increment_counter() does
``self._fail_counter += 1``.  The += is not atomic at the bytecode level
(LOAD_ATTR, BINARY_OP, STORE_ATTR in Python 3.11+).  Two threads calling
increment_counter() concurrently can both load the same value, both add
one, and one write overwrites the other — losing one increment.

Expected result: after two concurrent increments the fail counter should
be 2, but the race leaves it at 1.

Repository: https://github.com/danielfm/pybreaker
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_test_dir, "..", "external_repos", "pybreaker", "src")
# Insert local repo path FIRST so frontrun can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_root))

from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from pybreaker import STATE_CLOSED, CircuitMemoryStorage  # noqa: E402

from frontrun.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


class PyBreakerState:
    """Shared state: two threads each call increment_counter() once."""

    def __init__(self):
        self.storage = CircuitMemoryStorage(STATE_CLOSED)

    def thread1(self):
        self.storage.increment_counter()

    def thread2(self):
        self.storage.increment_counter()


def _invariant(s: PyBreakerState) -> bool:
    return s.storage.counter == 2


def test_real_pybreaker_lost_update():
    """Find the increment_counter() lost update in real CircuitMemoryStorage."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PyBreakerState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_real_pybreaker_lost_update_sweep():
    """Sweep 20 seeds to measure detection reliability."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: PyBreakerState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_invariant,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_pybreaker_reproduce():
    """Find a counterexample then reproduce it deterministically 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PyBreakerState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )

    if not result.counterexample:
        print("No counterexample found — skipping reproduction")
        return 0

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: PyBreakerState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        is_bug = state.storage.counter != 2
        bugs_reproduced += is_bug
        print(f"  Run {i + 1}: counter={state.storage.counter} [{'BUG' if is_bug else 'ok'}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 500 attempts) ===")
    test_real_pybreaker_lost_update()

    print("\n=== Seed sweep (20 seeds x 200 attempts) ===")
    test_real_pybreaker_lost_update_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_pybreaker_reproduce()
