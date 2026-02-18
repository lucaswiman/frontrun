"""
Real-code exploration: cachetools Cache.__setitem__ lost update.

Runs frontrun's bytecode exploration directly against the real cachetools
Cache class to find the currsize lost update race.

The bug: Cache.__setitem__ reads self.__currsize, computes diffsize, then
does self.__currsize += diffsize.  The += is not atomic at the bytecode
level (LOAD_ATTR, LOAD_FAST, INPLACE_ADD, STORE_ATTR).  Two threads
inserting different keys can both load the same currsize, both add their
size, and one update is lost.

Expected result: currsize diverges from len(cache) (each item has size 1).
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "cachetools", "src"))

from cachetools import Cache
from external_tests_helpers import print_exploration_result, print_seed_sweep_results

from frontrun.bytecode import explore_interleavings, run_with_schedule


class RealCacheState:
    def __init__(self):
        self.cache = Cache(maxsize=100)

    def thread1(self):
        self.cache["a"] = "value_a"

    def thread2(self):
        self.cache["b"] = "value_b"


def test_real_cachetools_lost_update():
    """Find the currsize lost update in real Cache.__setitem__."""

    result = explore_interleavings(
        setup=lambda: RealCacheState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # Each item has default size 1, so currsize should equal len
            s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


def test_real_cachetools_lost_update_sweep():
    """Sweep seeds to measure how easy the bug is to find."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: RealCacheState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: s.cache.currsize == len(s.cache),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_cachetools_reproduce():
    """Find a counterexample and reproduce it 10 times."""

    result = explore_interleavings(
        setup=lambda: RealCacheState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.cache.currsize == len(s.cache),
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
            setup=lambda: RealCacheState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
        )
        is_bug = state.cache.currsize != len(state.cache)
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        print(f"  Run {i + 1}: currsize={state.cache.currsize}, len={len(state.cache)} [{status}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 200 attempts) ===")
    test_real_cachetools_lost_update()

    print("\n=== Seed sweep (20 seeds × 100 attempts) ===")
    test_real_cachetools_lost_update_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_cachetools_reproduce()
