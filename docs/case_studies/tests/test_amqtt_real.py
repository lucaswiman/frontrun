"""
Real-code exploration: amqtt Session.next_packet_id duplicate packet IDs.

Runs frontrun's bytecode exploration against the real amqtt Session's
next_packet_id property to find the duplicate packet ID race.

The bug: Session.next_packet_id is a synchronous property that modifies
shared instance state with no locking:

    @property
    def next_packet_id(self):
        self._packet_id = (self._packet_id % 65535) + 1
        limit = self._packet_id
        while self._packet_id in self.inflight_in or ...:
            self._packet_id = (self._packet_id % 65535) + 1
            ...
        return self._packet_id

Two threads calling next_packet_id concurrently on the same Session can
interleave their writes to _packet_id and both return the same value —
violating the MQTT protocol requirement that in-flight packet IDs be unique.

Race scenario (T1 = thread1, T2 = thread2):
  T1: _packet_id = (0 % 65535) + 1 → writes 1
  T2: _packet_id = (1 % 65535) + 1 → writes 2   (reads T1's 1)
  T1: limit = self._packet_id → reads 2!  (T2 already wrote 2)
  T1: returns 2
  T2: limit = self._packet_id → reads 2
  T2: returns 2
  ⇒ Both threads return 2: duplicate packet ID!

Repository: https://github.com/Yakifo/amqtt
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_test_dir, "..", "external_repos", "amqtt")
# Insert local repo path FIRST so frontrun can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_root))

from amqtt.session import Session  # noqa: E402
from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)

from frontrun.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


class AmqttSessionState:
    """Shared state: two threads each call next_packet_id once, recording the result."""

    def __init__(self):
        self.session = Session()
        self.packet_id_1 = None
        self.packet_id_2 = None

    def thread1(self):
        self.packet_id_1 = self.session.next_packet_id

    def thread2(self):
        self.packet_id_2 = self.session.next_packet_id


def _ids_unique(s: AmqttSessionState) -> bool:
    """Invariant: both threads must receive distinct packet IDs."""
    return s.packet_id_1 != s.packet_id_2


def test_real_amqtt_duplicate_packet_id():
    """Find the duplicate packet ID race in real Session.next_packet_id."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: AmqttSessionState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_ids_unique,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_real_amqtt_duplicate_packet_id_sweep():
    """Sweep 20 seeds to measure detection reliability."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: AmqttSessionState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_ids_unique,
                max_attempts=200,
                max_ops=300,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_amqtt_reproduce():
    """Find a counterexample then reproduce it deterministically 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: AmqttSessionState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_ids_unique,
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
            setup=lambda: AmqttSessionState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        is_bug = state.packet_id_1 == state.packet_id_2
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: thread1_id={state.packet_id_1},"
            f" thread2_id={state.packet_id_2} [{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 500 attempts) ===")
    test_real_amqtt_duplicate_packet_id()

    print("\n=== Seed sweep (20 seeds x 200 attempts) ===")
    test_real_amqtt_duplicate_packet_id_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_amqtt_reproduce()
