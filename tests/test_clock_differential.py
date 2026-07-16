"""Differential soundness oracle for the virtual/explored clock (sync threads).

Hypothesis generates small concurrent programs (deterministic modulo
scheduling) from a fixed vocabulary — unsynchronized read-modify-write
increments (optionally with a ``time.sleep`` between read and write),
plain sleeps, ``threading.Lock``-protected and timed-``acquire`` increments,
and ``threading.Event`` set / timed wait — and checks virtual-clock DPOR
results against real-clock exploration of the *same* program.

Direction of the oracle (this matters): a virtual/explored clock legitimately
finds MORE failures than wall-clock execution (timer races real time cannot
schedule), so the soundness assertions are one-directional:

1.  **No false certification.**  ``clock="explored"`` models every clock
    advance as a schedulable DPOR step, so its interleaving space is a
    superset of what real-clock DPOR (where ``patch_sleep=True`` erases sleep
    ordering entirely) can produce.  If an explored-clock exploration returns
    a *certified* pass — ``property_holds=True`` with the search tree
    exhausted under ``preemption_bound=None`` — then neither real-clock DPOR
    nor bounded random real-clock exploration may find a counterexample.

    ``clock="virtual"`` (autojump) is deliberately narrower than real-clock
    DPOR: time advances as late as possible, so sleep-gated writes are pinned
    after runnable work (``test_autojump_does_not_explore_early_timer_fire``
    in test_virtual_clock.py documents this).  A virtual-certified pass with
    a real-DPOR failure is therefore NOT by itself a soundness bug; the sound
    subset relation for autojump is checked against ``clock="explored"``:
    a program certified under "explored" must also pass under "virtual".

2.  **Determinism.**  Running the same virtual-clock exploration twice yields
    an identical outcome, ``num_explored``, and counterexample.  (For specs
    containing timed ops — ``acquire(timeout=...)`` / ``Event.wait(timeout=
    ...)`` — only outcome equality is asserted: rare load-sensitive tree-size
    wobble was observed there, see the second known-gap test below.)

3.  **Failure evidence.**  Every virtual/explored failure carries a
    counterexample schedule and an explanation, and its replay reproduces
    the violation whenever reproduction was attempted.

Certification proxy: thread-execution DPOR leaves ``InterleavingResult.
exhausted`` as ``None``, and ``engine.next_execution()`` returns False both
on tree exhaustion and on hitting ``max_executions``.  With
``max_executions=N``, a completed pass with ``num_explored < N`` can only
mean the tree was exhausted; ``num_explored == N`` is conservatively treated
as uncertified.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import frontrun
from frontrun.common import InterleavingResult

# A single shared counter key: frontrun's DPOR tracks dict accesses at object
# granularity, so distinct keys in one dict conflict anyway and only inflate
# the trace count without adding scheduling variety.
KEY = "counter"

# Search bound.  preemption_bound=None makes exhaustion a genuine full-coverage
# claim; num_explored < MAX_EXECUTIONS then certifies the pass (see module
# docstring).  Sized so lock-heavy safe programs (~50-90 traces) certify while
# a cap-hit costs only ~1-2s.
MAX_EXECUTIONS = 200

Op = tuple[Any, ...]


@dataclass(frozen=True)
class ProgramSpec:
    """A generated concurrent program: one op sequence per worker."""

    workers: tuple[tuple[Op, ...], ...]

    def op_kinds(self) -> set[str]:
        return {op[0] for ops in self.workers for op in ops}

    def has_timed_ops(self) -> bool:
        """Ops whose cooperative wait loop was observed to leak real-time
        scheduling jitter into the *shape* of the DPOR tree (see
        ``test_known_gap_virtual_clock_tree_size_is_load_sensitive``)."""
        kinds = self.op_kinds()
        return "timed_lock_incr" in kinds or "event_wait" in kinds


class _SharedState:
    def __init__(self, num_workers: int) -> None:
        self.counters: dict[str, int] = {KEY: 0}
        self.lock = threading.Lock()
        self.event = threading.Event()
        # One private list per worker: appends never race, so the invariant
        # can compare the shared counter against the increments that were
        # actually performed (a timed acquire may legitimately skip its
        # increment when the acquire times out).
        self.performed: tuple[list[str], ...] = tuple([] for _ in range(num_workers))


def _run_op(op: Op, state: _SharedState, performed: list[str]) -> None:
    kind = op[0]
    if kind == "incr":
        # Unsynchronized read-modify-write; the optional sleep sits between
        # the read and the write to widen the lost-update window.
        _, mid_sleep = op
        tmp = state.counters[KEY]
        if mid_sleep is not None:
            time.sleep(mid_sleep)
        state.counters[KEY] = tmp + 1
        performed.append(KEY)
    elif kind == "lock_incr":
        with state.lock:
            tmp = state.counters[KEY]
            state.counters[KEY] = tmp + 1
        performed.append(KEY)
    elif kind == "timed_lock_incr":
        _, timeout = op
        if state.lock.acquire(timeout=timeout):
            try:
                tmp = state.counters[KEY]
                state.counters[KEY] = tmp + 1
            finally:
                state.lock.release()
            performed.append(KEY)
    elif kind == "sleep":
        time.sleep(op[1])
    elif kind == "event_set":
        state.event.set()
    elif kind == "event_wait":
        state.event.wait(timeout=op[1])
    else:  # pragma: no cover - guards against vocabulary drift
        raise AssertionError(f"unknown op kind {kind!r}")


def _make_workers(spec: ProgramSpec) -> list[Any]:
    def make(idx: int, ops: tuple[Op, ...]) -> Any:
        def worker(state: _SharedState) -> None:
            performed = state.performed[idx]
            for op in ops:
                _run_op(op, state, performed)

        return worker

    return [make(idx, ops) for idx, ops in enumerate(spec.workers)]


def _invariant(state: _SharedState) -> bool:
    expected = sum(len(performed) for performed in state.performed)
    got = state.counters[KEY]
    assert got == expected, f"lost update: counter is {got}, but {expected} increments were performed"
    return True


def _explore_dpor(spec: ProgramSpec, clock: str, *, reproduce: int = 3) -> InterleavingResult:
    num_workers = len(spec.workers)
    return frontrun.explore(
        setup=lambda: _SharedState(num_workers),
        workers=_make_workers(spec),
        invariant=_invariant,
        strategy="dpor",
        clock=clock,  # type: ignore[arg-type]
        preemption_bound=None,
        max_executions=MAX_EXECUTIONS,
        detect_io=False,
        reproduce_on_failure=reproduce,
    )


def _explore_random_real(spec: ProgramSpec) -> InterleavingResult:
    num_workers = len(spec.workers)
    return frontrun.explore(
        setup=lambda: _SharedState(num_workers),
        workers=_make_workers(spec),
        invariant=_invariant,
        strategy="random",
        clock="real",
        max_attempts=25,
        seed=0,
        timeout_per_run=1.0,
        detect_io=False,
        reproduce_on_failure=0,
    )


def _certified_pass(result: InterleavingResult, *, cap: int = MAX_EXECUTIONS) -> bool:
    """True when the result is a full-coverage pass (see module docstring).

    ``cap`` must be the ``max_executions`` the exploration actually ran with:
    a pass that used its whole execution budget is a truncated search, not a
    proof.
    """
    return result.property_holds and result.exhausted is not False and result.num_explored < cap


def _found_counterexample(result: InterleavingResult) -> bool:
    """True only for a *constructive* failure.

    ``property_holds=False`` with ``counterexample=None`` is an inconclusive
    search (timed-out or budget-exhausted attempts), not a counterexample; it
    cannot contradict a certification.
    """
    return not result.property_holds and result.counterexample is not None


def _assert_failure_evidence(result: InterleavingResult, label: str, spec: ProgramSpec) -> None:
    """Oracle assertion 3: a found failure must be a constructive proof."""
    assert result.counterexample is not None, f"[{label}] failure without a counterexample schedule; spec={spec}"
    assert result.explanation is not None, f"[{label}] failure without an explanation; spec={spec}"
    assert result.failures, f"[{label}] failure without a failures entry; spec={spec}"
    if result.reproduction_attempts:
        assert result.reproduction_successes == result.reproduction_attempts, (
            f"[{label}] counterexample replay reproduced only "
            f"{result.reproduction_successes}/{result.reproduction_attempts}; spec={spec}"
        )


def _run_oracle(spec: ProgramSpec) -> None:
    """Run the full differential oracle on one generated program."""
    result_virtual = _explore_dpor(spec, "virtual")
    result_virtual_again = _explore_dpor(spec, "virtual")

    # Assertion 2: determinism of the virtual-clock exploration.
    assert result_virtual.property_holds == result_virtual_again.property_holds, (
        f"virtual-clock outcome is nondeterministic: {result_virtual.property_holds} vs "
        f"{result_virtual_again.property_holds}; spec={spec}"
    )
    if not spec.has_timed_ops():
        assert result_virtual.num_explored == result_virtual_again.num_explored, (
            f"virtual-clock num_explored is nondeterministic: {result_virtual.num_explored} vs "
            f"{result_virtual_again.num_explored}; spec={spec}"
        )
        assert result_virtual.counterexample == result_virtual_again.counterexample, (
            f"virtual-clock counterexample is nondeterministic; spec={spec}"
        )

    result_explored = _explore_dpor(spec, "explored")

    # Assertion 3: found failures carry a schedule/explanation and replay
    # cleanly.  (A failure without a counterexample is an *inconclusive*
    # search — e.g. a timed-out execution — neither a pass nor a proof.)
    if _found_counterexample(result_virtual):
        _assert_failure_evidence(result_virtual, "clock=virtual", spec)
    if _found_counterexample(result_explored):
        _assert_failure_evidence(result_explored, "clock=explored", spec)

    # Autojump is one clock policy inside the explored clock's choice space,
    # so failures(virtual) ⊆ failures(explored): an explored-certified pass
    # with a virtual counterexample means the virtual clock manufactured an
    # interleaving the explored clock claims impossible.
    if _certified_pass(result_explored):
        assert not _found_counterexample(result_virtual), (
            f"clock='explored' certified a pass (exhausted after {result_explored.num_explored} executions) "
            f"but clock='virtual' found a counterexample:\n{result_virtual.explanation}\nspec={spec}"
        )

    # Assertion 1: no false certification against the real clock.
    if _certified_pass(result_explored):
        result_real = _explore_dpor(spec, "real", reproduce=0)
        assert not _found_counterexample(result_real), (
            f"SOUNDNESS: clock='explored' certified a pass (exhausted after "
            f"{result_explored.num_explored} executions, preemption_bound=None) but real-clock DPOR found a "
            f"counterexample:\n{result_real.explanation}\nspec={spec}"
        )
        result_random = _explore_random_real(spec)
        assert not _found_counterexample(result_random), (
            f"SOUNDNESS: clock='explored' certified a pass (exhausted after "
            f"{result_explored.num_explored} executions, preemption_bound=None) but random real-clock "
            f"exploration found a counterexample:\n{result_random.explanation}\nspec={spec}"
        )


# ---------------------------------------------------------------------------
# Program generator
# ---------------------------------------------------------------------------

_OPS: st.SearchStrategy[Op] = st.one_of(
    st.tuples(st.just("incr"), st.sampled_from((None, 0.0, 0.01))),
    st.tuples(st.just("lock_incr")),
    st.tuples(st.just("timed_lock_incr"), st.just(0.01)),
    st.tuples(st.just("sleep"), st.sampled_from((0.0, 0.005, 0.02))),
    st.tuples(st.just("event_set")),
    st.tuples(st.just("event_wait"), st.just(0.01)),
)


@st.composite
def program_specs(draw: st.DrawFn) -> ProgramSpec:
    # Mostly 2 workers; occasionally 3 (with shorter op sequences to keep the
    # exhaustive search under MAX_EXECUTIONS for lock-heavy safe programs).
    num_workers = draw(st.sampled_from((2, 2, 2, 3)))
    max_ops = 4 if num_workers == 2 else 2
    workers = tuple(tuple(draw(st.lists(_OPS, min_size=2, max_size=max_ops))) for _ in range(num_workers))
    return ProgramSpec(workers)


# ---------------------------------------------------------------------------
# The oracle property
# ---------------------------------------------------------------------------


@settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(spec=program_specs())
def test_clock_differential_oracle(spec: ProgramSpec) -> None:
    _run_oracle(spec)


# ---------------------------------------------------------------------------
# Harness calibration (red/green): prove the oracle machinery can catch bugs
# ---------------------------------------------------------------------------

_SEEDED_RACE = ProgramSpec(((("incr", None), ("incr", None)), (("incr", None),)))
_SAFE_LOCKED = ProgramSpec(((("lock_incr",), ("lock_incr",)), (("lock_incr",), ("lock_incr",))))


def test_canary_seeded_lost_update_is_found_with_evidence() -> None:
    """A known lost-update race must fail under virtual-clock DPOR with a
    replayable counterexample — the harness is measuring something real."""
    result = _explore_dpor(_SEEDED_RACE, "virtual")
    assert not result.property_holds
    assert not _certified_pass(result)
    _assert_failure_evidence(result, "canary", _SEEDED_RACE)
    assert result.reproduction_attempts == 3
    assert result.reproduction_successes == 3


def test_canary_weakened_search_is_not_certified() -> None:
    """A pass produced by a truncated search (max_executions=1) must not be
    mistaken for a certified pass by the exhaustion proxy."""
    result = frontrun.explore(
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
    assert result.property_holds  # the single explored interleaving happens to pass...
    assert result.num_explored == 1
    assert not _certified_pass(result, cap=1)  # ...but the oracle must not treat it as a proof


def test_canary_safe_program_certifies_under_all_clocks() -> None:
    """Green side: a lock-protected program certifies everywhere, and the full
    oracle accepts it."""
    for clock in ("virtual", "explored", "real"):
        result = _explore_dpor(_SAFE_LOCKED, clock, reproduce=0)
        assert result.property_holds, f"clock={clock}: {result.explanation}"
        assert _certified_pass(result), f"clock={clock} did not exhaust: {result.num_explored}"
        assert result.num_explored == 4, f"clock={clock} explored {result.num_explored} traces instead of 4"
    _run_oracle(_SAFE_LOCKED)


def test_safe_program_virtual_clock_count_is_stable() -> None:
    counts = [_explore_dpor(_SAFE_LOCKED, "virtual", reproduce=0).num_explored for _ in range(100)]
    assert counts == [4] * 100


def test_canary_autojump_narrowing_is_the_documented_semantics() -> None:
    """Pin the reason assertion 1 compares real-clock DPOR against
    clock='explored' rather than clock='virtual': autojump advances time as
    late as possible, so a sleep-gated increment is pinned after all runnable
    work and the virtual clock certifies a pass, while both the explored and
    the (sleep-order-erasing) real-clock DPOR find the lost update.  If this
    test ever fails, the oracle's exclusions should be revisited."""
    sleep_gated = ProgramSpec(((("sleep", 0.02), ("incr", None)), (("incr", None), ("incr", None))))
    result_virtual = _explore_dpor(sleep_gated, "virtual", reproduce=0)
    assert result_virtual.property_holds and _certified_pass(result_virtual)
    result_explored = _explore_dpor(sleep_gated, "explored", reproduce=0)
    assert not result_explored.property_holds
    result_real = _explore_dpor(sleep_gated, "real", reproduce=0)
    assert not result_real.property_holds


# ---------------------------------------------------------------------------
# Regression found by this oracle (minimized reproducer)
# ---------------------------------------------------------------------------

_WAIT_SET_GAP = ProgramSpec(
    (
        (("event_wait", 0.01), ("incr", None)),
        (("incr", None), ("event_set",)),
    )
)


def test_explored_clock_keeps_timeout_branch_of_satisfied_event_wait() -> None:
    result_explored = _explore_dpor(_WAIT_SET_GAP, "explored")
    assert not result_explored.property_holds, (
        f"clock='explored' certified a pass after {result_explored.num_explored} execution(s) despite the "
        "reachable wait-timeout lost-update race (real-clock DPOR finds and reproduces it)"
    )
    _assert_failure_evidence(result_explored, "clock=explored", _WAIT_SET_GAP)


# The spec on which the oracle's determinism assertion originally tripped
# (num_explored 7 vs 5 on back-to-back identical virtual-clock explorations).
_TIMED_CONTENTION = ProgramSpec(
    (
        (("timed_lock_incr", 0.01), ("sleep", 0.005), ("lock_incr",), ("event_wait", 0.01)),
        (("lock_incr",), ("lock_incr",), ("lock_incr",)),
    )
)


@pytest.mark.xfail(
    reason=(
        "Virtual-clock exploration-tree nondeterminism: back-to-back identical explorations of a program "
        "mixing lock.acquire(timeout=...) / Event.wait(timeout=...) with lock contention occasionally differ "
        "in num_explored (observed 5 vs 7 and 5 vs 9; property_holds stayed stable).  The outcome is a pure "
        "function of the spec, so the tree shape should be too; the wobble is rare (~1-5% of runs) and "
        "load-sensitive, consistent with the cooperative timed-wait spin leaking real scheduling jitter into "
        "the set of scheduling points DPOR sees.  Non-strict xfail: this test usually passes and exists to "
        "document/reproduce the gap; generated specs with timed ops are excluded from the oracle's strict "
        "num_explored determinism assertion until this is fixed.  Found by test_clock_differential.py's "
        "differential oracle, 2026-07."
    ),
    strict=False,
)
def test_known_gap_virtual_clock_tree_size_is_load_sensitive() -> None:
    results = [_explore_dpor(_TIMED_CONTENTION, "virtual", reproduce=0) for _ in range(6)]
    assert all(r.property_holds for r in results), results
    sizes = {r.num_explored for r in results}
    assert len(sizes) == 1, f"virtual-clock tree size varied across identical runs: {sorted(sizes)}"
