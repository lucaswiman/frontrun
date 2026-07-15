"""Differential soundness oracle for the virtual/explored clock (asyncio).

Async mirror of ``tests/test_clock_differential.py`` — see that module's
docstring for the oracle's direction and the certification proxy.  Hypothesis
generates small asyncio programs (deterministic modulo scheduling) from a
fixed vocabulary — read-modify-write increments with an optional
``asyncio.sleep`` suspension between read and write (without one the RMW is
atomic in asyncio), plain sleeps, ``asyncio.Lock``-protected increments, and
``asyncio.Event`` set / ``asyncio.wait_for``-timed waits — and checks
virtual/explored-clock async DPOR against real-clock exploration.

Assertions:

1.  **No false certification**: an explored-clock certified pass
    (``property_holds=True``, tree exhausted under ``preemption_bound=None``)
    must not be contradicted by real-clock DPOR or bounded random real-clock
    exploration.  As in the sync module, ``clock="virtual"`` (autojump) is
    deliberately narrower than real-clock DPOR with ``patch_sleep=True``
    (sleep-gated writes are pinned after runnable work), so autojump soundness
    is checked via the subset relation against ``clock="explored"``.
2.  **Determinism**: the same virtual-clock exploration twice yields identical
    outcome, ``num_explored``, and counterexample.  (Async exploration is
    single-threaded at await granularity; unlike the sync module no timed-op
    tree-size wobble has been observed, so this is asserted for all specs.)
3.  **Failure evidence**: virtual/explored failures carry a counterexample and
    explanation, and replay reproduces them whenever attempted.

Known gap (async twin of the sync finding, 2026-07): a timed event wait whose
executed run is satisfied by another worker's ``set()`` loses its
pending-timeout branch under ``clock="explored"`` — see
``test_known_gap_async_explored_clock_drops_timeout_branch_of_satisfied_wait``.
In async the real clock cannot catch this (a real ``wait_for`` timeout cannot
be scheduled early), so the regression is asserted via monotonicity against
the pure-timeout variant, which explored-clock DPOR *does* find.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import frontrun
from frontrun.common import InterleavingResult

KEY = "counter"
MAX_EXECUTIONS = 200

Op = tuple[Any, ...]


@dataclass(frozen=True)
class AsyncProgramSpec:
    """A generated asyncio program: one op sequence per worker task."""

    workers: tuple[tuple[Op, ...], ...]

    def op_kinds(self) -> set[str]:
        return {op[0] for ops in self.workers for op in ops}

    def has_known_wait_set_gap(self) -> bool:
        """Family excluded from the no-false-certification assertion (see
        the known-gap xfail test at the bottom of this module)."""
        kinds = self.op_kinds()
        return "event_wait" in kinds and "event_set" in kinds


class _SharedState:
    def __init__(self, num_workers: int) -> None:
        self.counters: dict[str, int] = {KEY: 0}
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        # One private list per task; the invariant compares the shared counter
        # against the increments actually performed.
        self.performed: tuple[list[str], ...] = tuple([] for _ in range(num_workers))


async def _run_op(op: Op, state: _SharedState, performed: list[str]) -> None:
    kind = op[0]
    if kind == "incr":
        # RMW increment; a suspension between read and write (mid_sleep is not
        # None) makes it raceable — without one it is atomic in asyncio.
        _, mid_sleep = op
        tmp = state.counters[KEY]
        if mid_sleep is not None:
            await asyncio.sleep(mid_sleep)
        state.counters[KEY] = tmp + 1
        performed.append(KEY)
    elif kind == "lock_incr":
        async with state.lock:
            tmp = state.counters[KEY]
            await asyncio.sleep(0)  # suspension inside the critical section
            state.counters[KEY] = tmp + 1
        performed.append(KEY)
    elif kind == "sleep":
        await asyncio.sleep(op[1])
    elif kind == "event_set":
        state.event.set()
    elif kind == "event_wait":
        try:
            await asyncio.wait_for(state.event.wait(), timeout=op[1])
        except (TimeoutError, asyncio.TimeoutError):  # asyncio.TimeoutError is distinct on 3.10
            pass
    else:  # pragma: no cover - guards against vocabulary drift
        raise AssertionError(f"unknown op kind {kind!r}")


def _make_workers(spec: AsyncProgramSpec) -> list[Any]:
    def make(idx: int, ops: tuple[Op, ...]) -> Any:
        async def worker(state: _SharedState) -> None:
            performed = state.performed[idx]
            for op in ops:
                await _run_op(op, state, performed)

        return worker

    return [make(idx, ops) for idx, ops in enumerate(spec.workers)]


def _invariant(state: _SharedState) -> bool:
    expected = sum(len(performed) for performed in state.performed)
    got = state.counters[KEY]
    assert got == expected, f"lost update: counter is {got}, but {expected} increments were performed"
    return True


def _explore_dpor(spec: AsyncProgramSpec, clock: str, *, reproduce: int = 3) -> InterleavingResult:
    num_workers = len(spec.workers)
    return asyncio.run(
        frontrun.explore(
            setup=lambda: _SharedState(num_workers),
            workers=_make_workers(spec),
            invariant=_invariant,
            strategy="dpor",
            clock=clock,  # type: ignore[arg-type]
            preemption_bound=None,
            max_executions=MAX_EXECUTIONS,
            detect_io=False,
            reproduce_on_failure=reproduce,
            # Our programs finish in milliseconds; 2s of no progress is a real
            # stall, and a shorter timeout bounds the cost of the (tolerated)
            # inconclusive executions async DPOR occasionally produces.
            deadlock_timeout=2.0,
        )
    )


def _explore_random_real(spec: AsyncProgramSpec) -> InterleavingResult:
    # Note: no reproduce_on_failure here — the async random strategy rejects it.
    # timeout_per_run is tight because random schedules of lock-heavy programs
    # routinely park tasks until the timeout; those attempts produce an
    # *inconclusive* result (no counterexample), which the oracle tolerates.
    num_workers = len(spec.workers)
    return asyncio.run(
        frontrun.explore(
            setup=lambda: _SharedState(num_workers),
            workers=_make_workers(spec),
            invariant=_invariant,
            strategy="random",
            clock="real",
            max_attempts=15,
            seed=0,
            timeout_per_run=1.0,
            detect_io=False,
        )
    )


def _certified_pass(result: InterleavingResult, *, cap: int = MAX_EXECUTIONS) -> bool:
    """Full-coverage pass proxy — see the sync module docstring."""
    return result.property_holds and result.exhausted is not False and result.num_explored < cap


def _found_counterexample(result: InterleavingResult) -> bool:
    """True only for a *constructive* failure.

    ``property_holds=False`` with ``counterexample=None`` is an inconclusive
    search (timed-out or budget-exhausted attempts), not a counterexample; it
    cannot contradict a certification.
    """
    return not result.property_holds and result.counterexample is not None


def _assert_failure_evidence(result: InterleavingResult, label: str, spec: AsyncProgramSpec) -> None:
    assert result.counterexample is not None, f"[{label}] failure without a counterexample schedule; spec={spec}"
    assert result.explanation is not None, f"[{label}] failure without an explanation; spec={spec}"
    if result.reproduction_attempts:
        assert result.reproduction_successes == result.reproduction_attempts, (
            f"[{label}] counterexample replay reproduced only "
            f"{result.reproduction_successes}/{result.reproduction_attempts}; spec={spec}"
        )


def _run_oracle(spec: AsyncProgramSpec, *, include_known_gap_families: bool = False) -> None:
    result_virtual = _explore_dpor(spec, "virtual")
    result_virtual_again = _explore_dpor(spec, "virtual")

    # Assertion 2: determinism.
    assert result_virtual.property_holds == result_virtual_again.property_holds, (
        f"async virtual-clock outcome is nondeterministic: {result_virtual.property_holds} vs "
        f"{result_virtual_again.property_holds}; spec={spec}"
    )
    assert result_virtual.num_explored == result_virtual_again.num_explored, (
        f"async virtual-clock num_explored is nondeterministic: {result_virtual.num_explored} vs "
        f"{result_virtual_again.num_explored}; spec={spec}"
    )
    assert result_virtual.counterexample == result_virtual_again.counterexample, (
        f"async virtual-clock counterexample is nondeterministic; spec={spec}"
    )

    result_explored = _explore_dpor(spec, "explored")

    # Assertion 3: found failures are constructive proofs.  (A failure without
    # a counterexample is an *inconclusive* search — e.g. a timed-out
    # execution — which is neither a pass nor a proof; nothing to assert.)
    if _found_counterexample(result_virtual):
        _assert_failure_evidence(result_virtual, "clock=virtual", spec)
    if _found_counterexample(result_explored):
        _assert_failure_evidence(result_explored, "clock=explored", spec)

    # failures(virtual) ⊆ failures(explored).
    if _certified_pass(result_explored):
        assert not _found_counterexample(result_virtual), (
            f"clock='explored' certified a pass (exhausted after {result_explored.num_explored} executions) "
            f"but clock='virtual' found a counterexample:\n{result_virtual.explanation}\nspec={spec}"
        )

    # Assertion 1: no false certification against the real clock.
    if _certified_pass(result_explored) and (include_known_gap_families or not spec.has_known_wait_set_gap()):
        result_real = _explore_dpor(spec, "real", reproduce=0)
        assert not _found_counterexample(result_real), (
            f"SOUNDNESS: async clock='explored' certified a pass (exhausted after "
            f"{result_explored.num_explored} executions, preemption_bound=None) but real-clock DPOR found a "
            f"counterexample:\n{result_real.explanation}\nspec={spec}"
        )
        result_random = _explore_random_real(spec)
        assert not _found_counterexample(result_random), (
            f"SOUNDNESS: async clock='explored' certified a pass (exhausted after "
            f"{result_explored.num_explored} executions, preemption_bound=None) but random real-clock "
            f"exploration found a counterexample:\n{result_random.explanation}\nspec={spec}"
        )


# ---------------------------------------------------------------------------
# Program generator
# ---------------------------------------------------------------------------

_OPS: st.SearchStrategy[Op] = st.one_of(
    st.tuples(st.just("incr"), st.sampled_from((None, 0.0, 0.01))),
    st.tuples(st.just("lock_incr")),
    st.tuples(st.just("sleep"), st.sampled_from((0.0, 0.005, 0.02))),
    st.tuples(st.just("event_set")),
    st.tuples(st.just("event_wait"), st.just(0.01)),
)


def _suspends(op: Op) -> bool:
    kind = op[0]
    return kind in ("lock_incr", "sleep", "event_wait") or (kind == "incr" and op[1] is not None)


@st.composite
def program_specs(draw: st.DrawFn) -> AsyncProgramSpec:
    num_workers = draw(st.sampled_from((2, 2, 2, 3)))
    max_ops = 4 if num_workers == 2 else 3
    workers: list[tuple[Op, ...]] = []
    for _ in range(num_workers):
        ops = tuple(draw(st.lists(_OPS, min_size=2, max_size=max_ops)))
        if not any(_suspends(op) for op in ops):
            # Async DPOR is known to time out an execution (inconclusive, all
            # clocks equally) when several tasks contain no suspension point —
            # see test_known_quirk_async_dpor_suspension_free_tasks below.
            # A suspension-free task also offers no interleaving choice, so
            # normalizing it keeps the generated space interesting.
            ops = (*ops, ("sleep", 0.0))
        workers.append(ops)
    return AsyncProgramSpec(tuple(workers))


@settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(spec=program_specs())
def test_async_clock_differential_oracle(spec: AsyncProgramSpec) -> None:
    _run_oracle(spec)


# ---------------------------------------------------------------------------
# Harness calibration (red/green)
# ---------------------------------------------------------------------------

_SEEDED_RACE = AsyncProgramSpec(((("incr", 0.0), ("incr", 0.0)), (("incr", 0.0),)))
_SAFE_LOCKED = AsyncProgramSpec(((("lock_incr",), ("lock_incr",)), (("lock_incr",), ("lock_incr",))))


def test_canary_async_seeded_lost_update_is_found_with_evidence() -> None:
    result = _explore_dpor(_SEEDED_RACE, "virtual")
    assert not result.property_holds
    _assert_failure_evidence(result, "canary", _SEEDED_RACE)
    assert result.reproduction_attempts == 3
    assert result.reproduction_successes == 3


def test_canary_async_weakened_search_is_not_certified() -> None:
    result = asyncio.run(
        frontrun.explore(
            setup=lambda: _SharedState(len(_SEEDED_RACE.workers)),
            workers=_make_workers(_SEEDED_RACE),
            invariant=_invariant,
            strategy="dpor",
            clock="virtual",
            preemption_bound=None,
            max_executions=1,
            detect_io=False,
            reproduce_on_failure=0,
        )
    )
    assert result.property_holds  # the single explored interleaving happens to pass...
    assert result.num_explored == 1
    assert not _certified_pass(result, cap=1)  # ...but it is not a proof


def test_canary_async_safe_program_certifies_under_all_clocks() -> None:
    for clock in ("virtual", "explored", "real"):
        result = _explore_dpor(_SAFE_LOCKED, clock, reproduce=0)
        assert result.property_holds, f"clock={clock}: {result.explanation}"
        assert _certified_pass(result), f"clock={clock} did not exhaust: {result.num_explored}"
    _run_oracle(_SAFE_LOCKED)


def test_canary_async_autojump_narrowing_is_the_documented_semantics() -> None:
    """Same autojump-vs-explored narrowing as the sync module: a sleep-gated
    atomic increment is pinned after runnable work under clock='virtual'
    (certified pass), while explored and real-clock DPOR find the lost update
    against a raceable increment in the other worker."""
    sleep_gated = AsyncProgramSpec(((("sleep", 0.01), ("incr", None)), (("incr", 0.0),)))
    result_virtual = _explore_dpor(sleep_gated, "virtual", reproduce=0)
    assert result_virtual.property_holds and _certified_pass(result_virtual)
    result_explored = _explore_dpor(sleep_gated, "explored", reproduce=0)
    assert not result_explored.property_holds
    result_real = _explore_dpor(sleep_gated, "real", reproduce=0)
    assert not result_real.property_holds


# ---------------------------------------------------------------------------
# Discrepancy found by this oracle (minimized reproducer, async twin)
# ---------------------------------------------------------------------------

_WAIT_ONLY = AsyncProgramSpec(((("event_wait", 0.01), ("incr", 0.0)), (("incr", 0.0),)))
_WAIT_SET_GAP = AsyncProgramSpec(((("event_wait", 0.01), ("incr", 0.0)), (("incr", 0.0), ("event_set",))))


def test_async_explored_clock_finds_timeout_race_without_setter() -> None:
    """Baseline for the known-gap test: with no set() in sight the explored
    clock does fire the wait_for timeout early and finds the lost update."""
    result = _explore_dpor(_WAIT_ONLY, "explored")
    assert not result.property_holds
    _assert_failure_evidence(result, "clock=explored", _WAIT_ONLY)


# Three tasks, each two increments with no suspension point anywhere.
_ATOMIC_TRIO = AsyncProgramSpec(tuple((("incr", None), ("incr", None)) for _ in range(3)))


@pytest.mark.xfail(
    reason=(
        "Async DPOR robustness quirk found by the oracle (not clock-specific: virtual, explored, and real "
        "all behave identically): with three tasks containing no suspension points, one of the five explored "
        "executions times out ('4 interleaving(s), 1 additional execution(s) timed out') and the search is "
        "reported inconclusive instead of exhausted; with two such tasks exploration completes.  Each "
        "occurrence burns the full deadlock_timeout of wall time.  The generator above appends a "
        "('sleep', 0.0) to suspension-free workers to keep the property test clear of this quirk.  Found by "
        "test_async_clock_differential.py's differential oracle, 2026-07."
    ),
    strict=False,
)
def test_known_quirk_async_dpor_suspension_free_tasks_are_inconclusive() -> None:
    result = _explore_dpor(_ATOMIC_TRIO, "virtual", reproduce=0)
    assert result.property_holds, result.explanation


@pytest.mark.xfail(
    reason=(
        "Async explored-clock false certification (twin of the sync gap in test_clock_differential.py): "
        "worker0 awaits wait_for(event.wait(), 0.01) then does a raceable increment; worker1 increments then "
        "set()s.  Every failing interleaving of the setter-less variant above (clock actor fires the timeout "
        "before worker1 runs, the two increments race, counter == 1) is still reachable when worker1 merely "
        "appends a set() after its increment — yet with the set() present exploration certifies a pass after "
        "a single execution (num_explored == 1): the pending-timeout branch is dropped once the executed "
        "run's wait is satisfied by the wake edge.  The real clock cannot catch this one (a real wait_for "
        "timeout cannot be scheduled early), so the assertion is monotonicity against the setter-less "
        "variant.  Found by test_async_clock_differential.py's differential oracle, 2026-07."
    ),
    strict=False,
)
def test_known_gap_async_explored_clock_drops_timeout_branch_of_satisfied_wait() -> None:
    result = _explore_dpor(_WAIT_SET_GAP, "explored")
    assert not result.property_holds, (
        f"clock='explored' certified a pass after {result.num_explored} execution(s) despite the reachable "
        "wait-timeout lost-update race (found by the same engine on the setter-less variant)"
    )
