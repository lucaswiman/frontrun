"""Mutation / fault-injection tests attacking the pass-certification gate.

docs/design-principles.rst ("Fail closed: a pass is a certificate, not a
default"): ``property_holds=True`` is a positive claim producible only by
``frontrun._certificate.certify_pass`` with real evidence — at least one
completed interleaving, every worker body entered, no coverage-degrading
events.  Results are tri-state: pass / fail (implies a counterexample) /
inconclusive (``property_holds=None`` with a machine-readable reason).
Internally contradictory evidence is a frontrun internal error, never a
verdict.  Each test here is a mutation the gate must kill.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

import frontrun
from frontrun.common import InterleavingResult

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Counter:
    def __init__(self) -> None:
        self.value = 0


def _unsafe_increment(c: _Counter) -> None:
    v = c.value
    c.value = v + 1


def _safe_noop(_state: object) -> None:
    return None


# ---------------------------------------------------------------------------
# certify_pass: the chokepoint constructor
# ---------------------------------------------------------------------------


def test_certify_pass_with_positive_evidence_produces_pass() -> None:
    from frontrun._certificate import PassEvidence, certify_pass

    result = certify_pass(evidence=PassEvidence(executions=3, workers_executed=[True, True]))
    assert result.property_holds is True
    assert result.inconclusive_reason is None
    assert result.assert_holds() is None


def test_zero_executions_is_inconclusive_with_reason() -> None:
    """Honest vacuity (user budget expired before any execution) is inconclusive, never a pass."""
    from frontrun._certificate import PassEvidence, certify_pass

    result = certify_pass(
        evidence=PassEvidence(
            executions=0,
            workers_executed=[],
            vacuous_reason="total_timeout=0.001s elapsed before any interleaving completed; increase total_timeout",
        )
    )
    assert result.property_holds is None
    assert result.inconclusive_reason is not None
    assert "total_timeout" in result.inconclusive_reason


def test_degradation_events_block_pass() -> None:
    """Completed executions plus a coverage-degrading event cannot certify a pass."""
    from frontrun._certificate import PassEvidence, certify_pass

    result = certify_pass(
        evidence=PassEvidence(
            executions=2,
            workers_executed=[True],
            degradation_events=["1 execution timed out before completion; increase timeout_per_run"],
        ),
        result=InterleavingResult(property_holds=None, num_explored=3),
    )
    assert result.property_holds is None
    assert result.inconclusive_reason is not None
    assert "timed out" in result.inconclusive_reason


def test_contradictory_evidence_worker_never_ran_is_internal_error() -> None:
    """Claimed completed interleavings while a worker body never ran: a frontrun bug, not a verdict."""
    from frontrun._certificate import FrontrunInternalError, PassEvidence, certify_pass

    with pytest.raises(FrontrunInternalError, match="never"):
        certify_pass(evidence=PassEvidence(executions=2, workers_executed=[True, False]))


def test_contradictory_evidence_no_worker_evidence_is_internal_error() -> None:
    from frontrun._certificate import FrontrunInternalError, PassEvidence, certify_pass

    with pytest.raises(FrontrunInternalError):
        certify_pass(evidence=PassEvidence(executions=1, workers_executed=[]))


def test_contradictory_evidence_more_executions_than_explored_is_internal_error() -> None:
    from frontrun._certificate import FrontrunInternalError, PassEvidence, certify_pass

    with pytest.raises(FrontrunInternalError):
        certify_pass(
            evidence=PassEvidence(executions=5, workers_executed=[True]),
            result=InterleavingResult(property_holds=None, num_explored=1),
        )


def test_contradictory_evidence_explored_but_unexplained_zero_executions() -> None:
    """num_explored > 0 with zero completed executions and no recorded cause is contradictory."""
    from frontrun._certificate import FrontrunInternalError, PassEvidence, certify_pass

    with pytest.raises(FrontrunInternalError):
        certify_pass(
            evidence=PassEvidence(executions=0, workers_executed=[]),
            result=InterleavingResult(property_holds=None, num_explored=2),
        )


def test_certify_pass_refuses_result_that_already_carries_a_failure() -> None:
    from frontrun._certificate import FrontrunInternalError, PassEvidence, certify_pass

    failing = InterleavingResult(property_holds=False, counterexample=[0, 1, 0], num_explored=1)
    with pytest.raises(FrontrunInternalError):
        certify_pass(evidence=PassEvidence(executions=1, workers_executed=[True]), result=failing)


def test_exhausted_claim_is_never_upgraded_by_certification() -> None:
    """Historical bug resurrected: exhausted=True after a truncated search is a false proof."""
    from frontrun._certificate import PassEvidence, certify_pass

    truncated = InterleavingResult(property_holds=None, num_explored=4, exhausted=False)
    result = certify_pass(
        evidence=PassEvidence(executions=4, workers_executed=[True, True], exhausted_claim=True),
        result=truncated,
    )
    assert result.property_holds is True
    assert result.exhausted is False, "a demoted coverage claim must survive certification"


def test_exhausted_and_property_holds_are_independent_axes() -> None:
    """A row-lock-redirect style demotion (exhausted=False) does not make a pass inconclusive."""
    from frontrun._certificate import PassEvidence, certify_pass

    demoted = InterleavingResult(property_holds=None, num_explored=2, exhausted=False)
    result = certify_pass(
        evidence=PassEvidence(executions=2, workers_executed=[True, True]),
        result=demoted,
    )
    assert result.property_holds is True
    assert result.exhausted is False


# ---------------------------------------------------------------------------
# tri-state assert_holds()
# ---------------------------------------------------------------------------


def test_assert_holds_raises_inconclusive_exploration_on_none() -> None:
    from frontrun._certificate import InconclusiveExploration

    result = InterleavingResult(
        property_holds=None,
        inconclusive_reason="total_timeout=0.01s elapsed before any interleaving completed; increase total_timeout",
    )
    with pytest.raises(InconclusiveExploration, match="total_timeout"):
        result.assert_holds()
    # Weaker claim requires opting in by name at the call site.
    assert result.assert_holds(allow_inconclusive=True) is None


def test_assert_holds_failure_and_inconclusive_are_distinct_exception_types() -> None:
    from frontrun._certificate import InconclusiveExploration

    failing = InterleavingResult(property_holds=False, counterexample=[0, 1], explanation="race found")
    with pytest.raises(AssertionError, match="race found"):
        failing.assert_holds()
    assert not issubclass(InconclusiveExploration, AssertionError)


def test_allow_inconclusive_does_not_swallow_genuine_failures() -> None:
    failing = InterleavingResult(property_holds=False, counterexample=[1, 0], explanation="boom")
    with pytest.raises(AssertionError, match="boom"):
        failing.assert_holds(allow_inconclusive=True)


def test_inconclusive_exploration_exported_from_frontrun() -> None:
    from frontrun._certificate import InconclusiveExploration

    assert frontrun.InconclusiveExploration is InconclusiveExploration


# ---------------------------------------------------------------------------
# zero-execution runs through the public API (the historical vacuous pass)
# ---------------------------------------------------------------------------


def test_explore_random_zero_budget_is_inconclusive() -> None:
    """A pre-expired total_timeout explores nothing: inconclusive, not a green pass."""
    from frontrun._certificate import InconclusiveExploration

    result = frontrun.explore_random(
        setup=_Counter,
        threads=[_unsafe_increment, _unsafe_increment],
        invariant=lambda c: True,
        max_attempts=5,
        total_timeout=1e-12,
        detect_io=False,
        reproduce_on_failure=0,
        seed=0,
    )
    assert result.property_holds is None
    assert result.num_explored == 0
    assert result.inconclusive_reason is not None
    assert "total_timeout" in result.inconclusive_reason
    with pytest.raises(InconclusiveExploration, match="total_timeout"):
        result.assert_holds()
    result.assert_holds(allow_inconclusive=True)


def test_explore_async_random_zero_budget_is_inconclusive() -> None:
    from frontrun._certificate import InconclusiveExploration

    async def task(_state: object) -> None:
        await asyncio.sleep(0)

    result = asyncio.run(
        frontrun.explore_async_random(
            setup=object,
            tasks=[task, task],
            invariant=lambda _state: True,
            max_attempts=5,
            total_timeout=1e-12,
            seed=0,
        )
    )
    assert result.property_holds is None
    assert result.num_explored == 0
    with pytest.raises(InconclusiveExploration, match="total_timeout"):
        result.assert_holds()


# ---------------------------------------------------------------------------
# fault injection: worker bodies silently skipped must not certify
# ---------------------------------------------------------------------------


def test_dpor_worker_body_skipped_cannot_certify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the runner so worker bodies never execute while exploration 'completes'."""
    import frontrun._dpor_runtime.explore as dpor_explore
    from frontrun._certificate import FrontrunInternalError

    class _SkippingRunner(dpor_explore.DporBytecodeRunner):
        def run(self, funcs, timeout=None):  # type: ignore[override]  # noqa: ARG002
            return None

    monkeypatch.setattr(dpor_explore, "DporBytecodeRunner", _SkippingRunner)
    with pytest.raises(FrontrunInternalError, match="never"):
        frontrun.explore(
            setup=_Counter,
            workers=[_safe_noop, _safe_noop],
            invariant=lambda _c: True,
            detect_io=False,
            reproduce_on_failure=0,
            max_executions=3,
        )


# ---------------------------------------------------------------------------
# False still implies a counterexample
# ---------------------------------------------------------------------------


def test_genuine_failure_keeps_false_with_counterexample() -> None:
    result = frontrun.explore(
        setup=_Counter,
        workers=[_unsafe_increment, _unsafe_increment],
        invariant=lambda c: c.value == 2,
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert result.property_holds is False
    assert result.counterexample is not None
    assert result.explanation is not None
    with pytest.raises(AssertionError):
        result.assert_holds()


# ---------------------------------------------------------------------------
# cross-process result conversion
# ---------------------------------------------------------------------------


def test_xproc_ok_zero_iterations_is_inconclusive() -> None:
    from frontrun._dpor_runtime.xproc.coordinator import CrossProcessResult
    from frontrun.cross_process import _to_interleaving_result

    cp = CrossProcessResult(
        ok=True,
        iterations=0,
        exhausted=False,
        truncation="total_timeout=0.01s expired during worker startup",
    )
    ir = _to_interleaving_result(cp)
    assert ir.property_holds is None
    assert ir.inconclusive_reason is not None
    assert "total_timeout" in ir.inconclusive_reason


def test_xproc_ok_with_worker_evidence_certifies() -> None:
    from frontrun._dpor_runtime.xproc.coordinator import CrossProcessResult
    from frontrun.cross_process import _to_interleaving_result

    cp = CrossProcessResult(ok=True, iterations=3, exhausted=True, workers_executed=[True, True])
    ir = _to_interleaving_result(cp)
    assert ir.property_holds is True
    assert ir.exhausted is True


def test_xproc_ok_without_worker_evidence_is_internal_error() -> None:
    from frontrun._certificate import FrontrunInternalError
    from frontrun._dpor_runtime.xproc.coordinator import CrossProcessResult
    from frontrun.cross_process import _to_interleaving_result

    with pytest.raises(FrontrunInternalError):
        _to_interleaving_result(CrossProcessResult(ok=True, iterations=3, exhausted=False))


# ---------------------------------------------------------------------------
# lint tripwire: property_holds=True literals live only in the chokepoint
# ---------------------------------------------------------------------------


def test_property_holds_true_literal_only_in_certificate_module() -> None:
    package_root = Path(frontrun.__file__).resolve().parent
    pattern = re.compile(r"property_holds\s*=\s*True")
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root).as_posix()
        if rel == "_certificate.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"frontrun/{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "property_holds=True may only be produced by frontrun._certificate.certify_pass; "
        "route these through the chokepoint:\n" + "\n".join(offenders)
    )
