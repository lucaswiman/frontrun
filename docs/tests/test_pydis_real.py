"""
Real-code exploration: pydis INCR lost update and SET NX race.

Runs frontrun's bytecode exploration directly against the real pydis
RedisProtocol methods to find concurrency bugs in the
shared global dictionary.

Bug 1 (INCR): com_incr reads the value, increments locally, writes back.
Two concurrent INCRs both read the same value, both write value+1,
and one increment is lost.

Bug 2 (SET NX): com_set with NX flag checks `if key in dictionary`,
then sets.  Two clients both pass the check and both write, violating
the set-if-not-exists semantics.

Note: pydis is an asyncio server, but the race conditions are in
synchronous read-modify-write patterns on module-level globals.  We
use threading-based exploration since the shared state bugs are
identical whether concurrency comes from threads or interleaved
async tasks.
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "pydis"))

from external_tests_helpers import print_exploration_result
from pydis.__main__ import RedisProtocol, dictionary, expiration

from frontrun.bytecode import explore_interleavings, run_with_schedule


class RealPydisIncrState:
    """Two 'clients' both INCR the same key."""

    def __init__(self):
        # Reset module-level globals
        dictionary.clear()
        expiration.clear()
        # Pre-set the counter
        dictionary[b"counter"] = b"0"
        # Create two protocol instances (like two client connections)
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_incr(b"counter")

    def thread2(self):
        self.client2.com_incr(b"counter")


class RealPydisSetNxState:
    """Two 'clients' both SET the same key with NX (set-if-not-exists)."""

    def __init__(self):
        # Reset module-level globals
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.result1 = None
        self.result2 = None

    def thread1(self):
        self.result1 = self.client1.com_set(b"key", b"value_A", b"NX")

    def thread2(self):
        self.result2 = self.client2.com_set(b"key", b"value_B", b"NX")


def test_real_pydis_incr_lost_update():
    """Find the INCR lost update in real pydis code."""

    result = explore_interleavings(
        setup=lambda: RealPydisIncrState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # After two INCRs starting from 0, counter should be 2
            int(dictionary.get(b"counter", b"0")) == 2
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


def test_real_pydis_set_nx_race():
    """Find the SET NX race in real pydis code."""

    result = explore_interleavings(
        setup=lambda: RealPydisSetNxState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # With NX, exactly one SET should succeed (+OK) and one should
            # fail ($-1).  If both succeed, the NX semantics are violated.
            (s.result1 == b"+OK\r\n") != (s.result2 == b"+OK\r\n")
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


def test_real_pydis_incr_sweep():
    """Sweep seeds for INCR lost update."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: RealPydisIncrState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: int(dictionary.get(b"counter", b"0")) == 2,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print("\nTotal interleavings explored:", total_explored)
    print(f"Seeds that found the INCR bug: {len(found_seeds)} / 20")
    for seed, n in found_seeds:
        print(f"  seed={seed}: found after {n} interleavings")

    return found_seeds


def test_real_pydis_incr_reproduce():
    """Find and reproduce the INCR lost update."""

    result = explore_interleavings(
        setup=lambda: RealPydisIncrState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: int(dictionary.get(b"counter", b"0")) == 2,
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
            setup=lambda: RealPydisIncrState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
        )
        counter_val = int(dictionary.get(b"counter", b"0"))
        is_bug = counter_val != 2
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        print(f"  Run {i + 1}: counter={counter_val} [{status}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== INCR Lost Update (seed=42, 200 attempts) ===")
    test_real_pydis_incr_lost_update()

    print("\n=== SET NX Race (seed=42, 200 attempts) ===")
    test_real_pydis_set_nx_race()

    print("\n=== INCR Seed sweep (20 seeds × 100 attempts) ===")
    test_real_pydis_incr_sweep()

    print("\n=== INCR Deterministic reproduction ===")
    test_real_pydis_incr_reproduce()
