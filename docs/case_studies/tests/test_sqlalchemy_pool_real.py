"""
Real-code exploration: SQLAlchemy QueuePool._inc_overflow() lost update.

Runs frontrun's bytecode exploration against the real QueuePool._inc_overflow()
method to find the _overflow counter lost update when max_overflow is unlimited.

The bug: When max_overflow == -1 (unlimited overflow), _inc_overflow() does
``self._overflow += 1`` WITHOUT holding _overflow_lock:

    def _inc_overflow(self):
        if self._max_overflow == -1:
            self._overflow += 1   # <-- NO LOCK
            return True
        with self._overflow_lock:
            ...

The += is not atomic at the bytecode level.  Two threads calling _inc_overflow()
concurrently can both load the same value, both compute value+1, and one write
overwrites the other — losing one increment.

Expected result: after two concurrent _inc_overflow() calls
_overflow == initial_overflow + 2, but the race gives initial_overflow + 1.

No real database is needed — _inc_overflow() only manipulates an integer counter.

Repository: https://github.com/sqlalchemy/sqlalchemy
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_lib = os.path.join(_test_dir, "..", "external_repos", "sqlalchemy", "lib")
# Insert local repo path FIRST so frontrun can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_lib))

from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from sqlalchemy.pool import QueuePool  # noqa: E402

from frontrun.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


class SQLAlchemyPoolState:
    """Shared state: two threads each call _inc_overflow() once.

    pool_size=5 → _overflow starts at -5.
    max_overflow=-1 → unlimited overflow, so _inc_overflow() skips the lock.
    """

    def __init__(self):
        # creator is never invoked — we never check out a real connection.
        self.pool = QueuePool(lambda: None, pool_size=5, max_overflow=-1)
        self.initial_overflow = self.pool._overflow  # -5

    def thread1(self):
        self.pool._inc_overflow()

    def thread2(self):
        self.pool._inc_overflow()


def _invariant(s: SQLAlchemyPoolState) -> bool:
    return s.pool._overflow == s.initial_overflow + 2


def test_real_sqlalchemy_overflow_lost_update():
    """Find the _overflow lost update in real QueuePool._inc_overflow()."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SQLAlchemyPoolState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_real_sqlalchemy_overflow_sweep():
    """Sweep 20 seeds to measure detection reliability."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: SQLAlchemyPoolState(),
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


def test_real_sqlalchemy_reproduce():
    """Find a counterexample then reproduce it deterministically 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SQLAlchemyPoolState(),
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
            setup=lambda: SQLAlchemyPoolState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        expected = state.initial_overflow + 2
        actual = state.pool._overflow
        is_bug = actual != expected
        bugs_reproduced += is_bug
        print(f"  Run {i + 1}: _overflow={actual} (expected {expected}) [{'BUG' if is_bug else 'ok'}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 500 attempts) ===")
    test_real_sqlalchemy_overflow_lost_update()

    print("\n=== Seed sweep (20 seeds x 200 attempts) ===")
    test_real_sqlalchemy_overflow_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_sqlalchemy_reproduce()
