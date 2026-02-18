"""
Real-code exploration: PyDispatcher connect() TOCTOU.

Runs frontrun's bytecode exploration directly against the real
pydispatch.dispatcher.connect() function to find the
TOCTOU where two threads racing to connect receivers to the same
(sender, signal) lose one registration.

The bug: connect() checks `if senderkey in connections`, creates a new
signals dict if absent, then adds the receiver.  Two threads can both
see the key as absent, both create separate dicts, and the second
overwrites the first — losing the first receiver.
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "pydispatcher"))

from external_tests_helpers import print_exploration_result, print_seed_sweep_results
from pydispatch import dispatcher

from frontrun.bytecode import explore_interleavings, run_with_schedule


# Module-level functions with stable ids (avoid bound method id issues)
def recv1(**kw):
    pass


def recv2(**kw):
    pass


class RealDispatcherState:
    def __init__(self):
        # Reset PyDispatcher's global state for each run
        dispatcher.connections.clear()
        dispatcher.senders.clear()
        dispatcher.sendersBack.clear()
        self.sender = object()
        self.signal = "test_signal"

    def thread1(self):
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)

    def thread2(self):
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender, weak=False)


def test_real_pydispatcher_connect_race():
    """Find the connect() TOCTOU in real PyDispatcher."""

    result = explore_interleavings(
        setup=lambda: RealDispatcherState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # Both receivers should be registered for this sender+signal
            len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 2
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


def test_real_pydispatcher_connect_race_sweep():
    """Sweep seeds to measure how easy the bug is to find."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: RealDispatcherState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 2,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_pydispatcher_reproduce():
    """Find a counterexample and reproduce it 10 times."""

    result = explore_interleavings(
        setup=lambda: RealDispatcherState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 2,
        max_attempts=500,
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
            setup=lambda: RealDispatcherState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
        )
        registered = len(dispatcher.connections.get(id(state.sender), {}).get(state.signal, []))
        is_bug = registered != 2
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        print(f"  Run {i + 1}: registered={registered} [{status}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 500 attempts) ===")
    test_real_pydispatcher_connect_race()

    print("\n=== Seed sweep (20 seeds × 100 attempts) ===")
    test_real_pydispatcher_connect_race_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_pydispatcher_reproduce()
