"""Serializability-baseline + race-failure formatting shared by sync/async DPOR."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from frontrun.common import (
    compute_serializable_states,
    compute_serializable_states_async,
    resolve_serializable_hash_fn,
)


def compute_serializable_baseline_sync(
    setup: Callable[[], Any],
    threads: list[Callable[[Any], None]],
    serializable_invariant: Callable[[Any], Any] | bool,
) -> tuple[set[Any] | None, Callable[[Any], Any]]:
    """Return ``(serial_valid_states, serial_hash_fn)`` for the sync driver.

    When *serializable_invariant* is ``False``, returns ``(None, repr)``.
    """
    if serializable_invariant is False:
        return None, repr
    hash_fn: Callable[[Any], Any] = resolve_serializable_hash_fn(serializable_invariant) or repr
    return compute_serializable_states(setup, threads, state_hash=hash_fn), hash_fn


async def compute_serializable_baseline_async(
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Any]],
    serializable_invariant: Callable[[Any], Any] | bool,
) -> tuple[set[Any] | None, Callable[[Any], Any]]:
    """Async counterpart to :func:`compute_serializable_baseline_sync`."""
    if serializable_invariant is False:
        return None, repr
    hash_fn: Callable[[Any], Any] = resolve_serializable_hash_fn(serializable_invariant) or repr
    return await compute_serializable_states_async(setup, tasks, state_hash=hash_fn), hash_fn


def format_race_failure_explanation(
    execution_num: int,
    num_races: int,
    *,
    actor_plural: str = "threads",
) -> str:
    """Build the ``error_on_any_race`` failure explanation."""
    return (
        f"Unsynchronized race detected in execution {execution_num}.\n"
        f"{num_races} race(s) found between {actor_plural} on shared objects."
    )
