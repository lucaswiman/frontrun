"""
Real-code exploration: pykka ActorRef.tell() TOCTOU — ghost messages.

Runs frontrun's bytecode exploration against the real pykka ActorRef.tell()
to find the TOCTOU race between the is_alive() check and actor_inbox.put().

The bug in ActorRef.tell() (src/pykka/_ref.py):

    def tell(self, message):
        if not self.is_alive():            # CHECK
            raise ActorDeadError(...)
        self.actor_inbox.put(Envelope(...)) # ACT  ← no lock between check and put

If a concurrent stop() call sets actor_stopped between the check and the put,
tell() succeeds without raising an error but the message lands in the inbox of
an actor whose loop has already exited — a "ghost message" that is never
processed.

We exercise this with a real daemon ThreadingActor:
  - Thread 1 calls actor_ref.tell("ping")   (may succeed or raise ActorDeadError)
  - Thread 2 calls actor_ref.stop(block=True, timeout=2.0)

The actor records every message it processes in a shared list.  After both
threads complete (and therefore the actor has fully stopped), we check:

    tell_successes == len(received)

If tell() succeeded but the actor never processed "ping"
(ghost message), this invariant is violated.

Repository: https://github.com/jodal/pykka
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_src = os.path.join(_test_dir, "..", "external_repos", "pykka", "src")
# Insert local repo path FIRST so frontrun can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_src))

import pykka  # noqa: E402
from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from pykka import ActorDeadError  # noqa: E402

from frontrun.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


class PykkaGhostMessageState:
    """
    Shared state for the tell() TOCTOU test.

    The actor runs as a daemon thread (use_daemon_thread = True) so it
    never blocks the process from exiting.  We use a shared list
    ``received`` to track every message the actor successfully processes.

    After both threads complete, thread2's stop(block=True) guarantees the
    actor loop has fully exited, so ``received`` is final.

    Invariant: tell_successes == len(received).
    A violation means tell() returned without an error but the actor never
    saw the message — a ghost message caused by the TOCTOU race.
    """

    def __init__(self):
        self.received: list = []
        received = self.received  # capture for the actor closure

        class DaemonActor(pykka.ThreadingActor):
            use_daemon_thread = True  # actor thread exits when main thread exits

            def on_receive(self, message):
                received.append(message)

        self.actor_ref = DaemonActor.start()
        self.tell_successes = 0
        self.tell_errors = 0

    def thread1(self):
        """Try to send a message to the actor."""
        try:
            self.actor_ref.tell("ping")
            self.tell_successes += 1
        except ActorDeadError:
            self.tell_errors += 1

    def thread2(self):
        """Stop the actor, blocking until the loop has fully exited."""
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except ActorDeadError:
            pass  # already stopped — that's fine


def _no_ghost_messages(s: PykkaGhostMessageState) -> bool:
    """Every successful tell() must result in the actor receiving the message."""
    return s.tell_successes == len(s.received)


def test_real_pykka_tell_toctou():
    """Find the tell() ghost-message TOCTOU in real pykka ActorRef."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PykkaGhostMessageState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_no_ghost_messages,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_real_pykka_tell_toctou_sweep():
    """Sweep 20 seeds to measure detection reliability."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: PykkaGhostMessageState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_no_ghost_messages,
                max_attempts=200,
                max_ops=500,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_real_pykka_reproduce():
    """Find a counterexample then reproduce it deterministically 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PykkaGhostMessageState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_no_ghost_messages,
            max_attempts=500,
            max_ops=500,
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
            setup=lambda: PykkaGhostMessageState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        is_bug = not _no_ghost_messages(state)
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: tell_successes={state.tell_successes},"
            f" received={len(state.received)} [{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


if __name__ == "__main__":
    print("=== Single run (seed=42, 500 attempts) ===")
    test_real_pykka_tell_toctou()

    print("\n=== Seed sweep (20 seeds x 200 attempts) ===")
    test_real_pykka_tell_toctou_sweep()

    print("\n=== Deterministic reproduction ===")
    test_real_pykka_reproduce()
