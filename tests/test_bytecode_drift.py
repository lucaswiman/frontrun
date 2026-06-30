"""Finding 5: random schedules must be able to produce relative drift > 1 opcode.

The old ``random_round_robin_schedule`` emitted only concatenations of full
permutations of the actors (one traced opcode per slot), so two runnable
threads never drifted more than +/-1 opcode apart.  A race that requires thread
B to execute several consecutive opcodes *inside* a narrow window of thread A
was therefore structurally unreachable, regardless of ``max_attempts``.

This test uses a clean drift detector: thread B increments a counter many
times; thread A samples the counter twice in quick succession.  Observing the
counter advance by >= 2 between A's two samples requires B to complete a whole
(multi-opcode) increment strictly inside A's two-opcode window — impossible
under +/-1 lockstep round-robin, reachable with variable-length bursts.
"""

from frontrun import explore_random


class _State:
    def __init__(self) -> None:
        self.counter = 0
        self.raced = False


def _thread_a(state: _State) -> None:
    s1 = state.counter
    s2 = state.counter
    if s2 - s1 >= 2:
        state.raced = True


def _thread_b(state: _State) -> None:
    for _ in range(20):
        state.counter = state.counter + 1


def test_random_exploration_finds_drift_requiring_race():
    """explore_random must reach a B-burst-inside-A-window interleaving."""
    result = explore_random(
        setup=_State,
        threads=[_thread_a, _thread_b],
        invariant=lambda s: not s.raced,
        max_attempts=400,
        max_ops=300,
        seed=12345,
    )

    assert not result.property_holds, (
        "Expected to find the drift-requiring race within the attempt budget, "
        f"but invariant held across {result.num_explored} interleavings. "
        "Random schedules cannot produce relative drift > 1 opcode."
    )
