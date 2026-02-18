"""
Real-code exploration: urllib3 HTTPConnectionPool.num_connections lost update.

Runs frontrun's bytecode exploration directly against the real
HTTPConnectionPool._new_conn() method to find the num_connections counter
lost update race.

The bug: HTTPConnectionPool._new_conn() does ``self.num_connections += 1``.
The += is not atomic at the bytecode level.  Two threads creating connections
concurrently can both load the same counter value, both compute value+1, and
one write overwrites the other — losing one connection from the count.

Expected result: after two concurrent _new_conn() calls num_connections
should be 2, but the race leaves it at 1.

Note: _new_conn() constructs an HTTPConnection object without making any
real network connection, so no server is needed.

Repository: https://github.com/urllib3/urllib3
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_src = os.path.join(_test_dir, "..", "external_repos", "urllib3", "src")
# Insert local repo path FIRST so frontrun can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_src))

# Import directly from the submodule — the top-level __init__ tries to import
# a generated _version file that doesn't exist in a bare source checkout.
from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from urllib3.connectionpool import HTTPConnectionPool  # noqa: E402

from frontrun.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


class Urllib3State:
    """Shared state: two threads each call _new_conn() once.

    _new_conn() only instantiates an HTTPConnection object;
    it never dials the host, so no server is needed.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999)

    def thread1(self):
        self.pool._new_conn()

    def thread2(self):
        self.pool._new_conn()


def _invariant(s: Urllib3State) -> bool:
    return s.pool.num_connections == 2


def test_real_urllib3_num_connections_lost_update():
    """Find the num_connections lost update in real HTTPConnectionPool._new_conn()."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: Urllib3State(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_invariant,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_real_urllib3_num_connections_sweep():
    """Sweep 20 seeds to measure detection reliability."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: Urllib3State(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_invariant,
                max_attempts=200,
                max_ops=300,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_urllib3_reproduce():
    """Find a counterexample then reproduce it deterministically 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: Urllib3State(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_invariant,
            max_attempts=500,
            max_ops=300,
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
            setup=lambda: Urllib3State(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        is_bug = state.pool.num_connections != 2
        bugs_reproduced += is_bug
        print(f"  Run {i + 1}: num_connections={state.pool.num_connections} [{'BUG' if is_bug else 'ok'}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 500 attempts) ===")
    test_real_urllib3_num_connections_lost_update()

    print("\n=== Seed sweep (20 seeds x 200 attempts) ===")
    test_real_urllib3_num_connections_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_urllib3_reproduce()
