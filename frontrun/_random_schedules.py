"""Shared random schedule generators for sync + async exploration drivers.

Both :mod:`frontrun.bytecode` (threaded opcode-level) and
:mod:`frontrun.async_shuffler` (await-point-level) drive their random search
loops with the same fair, round-robin schedule. This module owns the
schedule generation so both modes can stay in sync without code drift.
"""

from __future__ import annotations

import random
from typing import Any

# Default maximum per-actor burst length.  A *burst* is a run of consecutive
# slots given to one actor before yielding, which lets two runnable threads
# drift more than one opcode apart.  Without bursts (burst == 1 always) the
# schedule is pure lockstep round-robin and races that need actor B to execute
# k > 1 opcodes inside a narrow window of actor A are structurally unreachable.
_DEFAULT_MAX_BURST = 8

# Probability of an occasional larger "skew" burst, drawn from a wider range,
# to reach interleavings where one actor runs far ahead of the others.
_SKEW_PROBABILITY = 0.15
_SKEW_MAX_BURST = 32


def _draw_burst(rng: random.Random, max_burst: int, skew_max_burst: int) -> int:
    """Draw a burst length, usually in ``[1, max_burst]`` with rare larger skews."""
    if rng.random() < _SKEW_PROBABILITY:
        return rng.randint(1, max(1, skew_max_burst))
    return rng.randint(1, max(1, max_burst))


def random_round_robin_schedule(
    rng: random.Random,
    num_actors: int,
    max_ops: int,
    *,
    max_burst: int = _DEFAULT_MAX_BURST,
    skew_max_burst: int = _SKEW_MAX_BURST,
) -> list[int]:
    """Build a fair schedule of actor ids with variable-length bursts.

    Each round is a random permutation of ``range(num_actors)``; each actor in
    the permutation receives a *burst* of consecutive slots (length drawn from
    :func:`_draw_burst`).  This keeps every actor appearing in every round
    (fairness) while allowing relative opcode drift greater than one — so races
    requiring one thread to run several opcodes inside another thread's window
    become reachable.

    ``num_rounds`` is drawn uniformly from ``[1, max(1, max_ops // num_actors)]``.
    """
    num_rounds = rng.randint(1, max(1, max_ops // num_actors))
    schedule: list[int] = []
    for _ in range(num_rounds):
        round_perm = list(range(num_actors))
        rng.shuffle(round_perm)
        for actor in round_perm:
            burst = _draw_burst(rng, max_burst, skew_max_burst)
            schedule.extend([actor] * burst)
    return schedule


def fair_schedule_strategy(num_actors: int, max_ops: int) -> Any:
    """Hypothesis strategy producing fair, round-robin schedules.

    Mirrors :func:`random_round_robin_schedule` but draws permutations via
    ``hypothesis.strategies.permutations`` so generated schedules shrink
    sensibly. Returns the strategy lazily to avoid importing ``hypothesis``
    at module import time.
    """
    from hypothesis import strategies as st  # type: ignore[import-not-found]

    max_rounds = max(1, max_ops // num_actors)
    actors = list(range(num_actors))

    @st.composite  # type: ignore[attr-defined]
    def _fair_schedule(draw: st.DrawFn) -> list[int]:  # type: ignore[attr-defined,name-defined]
        num_rounds = draw(st.integers(min_value=1, max_value=max_rounds))  # type: ignore[attr-defined]
        schedule: list[int] = []
        for _ in range(num_rounds):
            for actor in draw(st.permutations(actors)):  # type: ignore[attr-defined]
                # Variable-length burst so the schedule can express relative
                # opcode drift > 1.  min_value=1 keeps lockstep round-robin in
                # the shrink target (Hypothesis shrinks toward 1).
                burst = draw(st.integers(min_value=1, max_value=_DEFAULT_MAX_BURST))  # type: ignore[attr-defined]
                schedule.extend([actor] * burst)
        return schedule

    return _fair_schedule()


__all__ = ["fair_schedule_strategy", "random_round_robin_schedule"]
