"""Pass-certificate chokepoint: the only sanctioned producer of ``property_holds=True``.

A pass is a certificate, not a default (see :doc:`docs/design-principles`,
"Fail closed").  Every exploration strategy routes its "no failure found"
endpoint through :func:`certify_pass` with the evidence it actually gathered;
the chokepoint decides between the three verdicts:

* **pass** — at least one interleaving completed, every worker body entered,
  no coverage-degrading events;
* **inconclusive** — honest vacuity (a user budget expired before any
  interleaving completed) or a degraded search: ``property_holds=None`` with a
  machine-readable ``inconclusive_reason`` naming cause and remedy;
* **internal error** — internally contradictory evidence raises
  :class:`FrontrunInternalError` immediately and never becomes a verdict.

A lint tripwire (tests/test_pass_certificate.py) enforces that no other module
in the package contains a ``property_holds=True`` literal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontrun.common import InterleavingResult


class FrontrunInternalError(Exception):
    """Exploration evidence is internally contradictory — a frontrun bug, never a verdict."""


class InconclusiveExploration(Exception):  # noqa: N818  # public name mandated by the tri-state verdict design
    """``assert_holds()`` on an inconclusive result (``property_holds=None``).

    Exploration produced no evidence either way — e.g. a user-set budget or
    timeout expired before any interleaving completed.  Distinct from the
    ``AssertionError`` raised for a genuine counterexample; accept the weaker
    claim explicitly with ``assert_holds(allow_inconclusive=True)``.
    """


@dataclass(frozen=True)
class PassEvidence:
    """Positive evidence backing a pass certificate.

    Attributes:
        executions: Interleavings that ran to completion and had their
            invariant (and race/serializability checks) evaluated.  A pass
            requires at least one.
        workers_executed: Per-worker flags: ``workers_executed[i]`` is True
            iff worker *i*'s body was entered at least once across the
            exploration.  A pass requires all True (and at least one entry).
        degradation_events: Coverage-degrading events observed during
            exploration (timed-out executions, truncated schedules, tracer
            failures).  Any entry demotes a would-be pass to inconclusive;
            each string should name cause and remedy.
        exhausted_claim: The strategy's coverage claim, threaded through to
            ``result.exhausted``.  Certification never *upgrades* an existing
            claim — a demotion recorded on the result (e.g. a row-lock
            redirect) survives; exhaustiveness and the verdict are
            independent axes.
        vacuous_reason: Cause-and-remedy message used when ``executions == 0``
            (honest, user-induced vacuity such as a pre-expired
            ``total_timeout``).
    """

    executions: int
    workers_executed: Sequence[bool]
    degradation_events: Sequence[str] = ()
    exhausted_claim: bool | None = None
    vacuous_reason: str | None = None


def _make_inconclusive(result: InterleavingResult, reason: str) -> InterleavingResult:
    result.property_holds = None
    result.inconclusive_reason = reason
    if result.explanation is None:
        result.explanation = reason
    return result


def certify_pass(*, evidence: PassEvidence, result: InterleavingResult | None = None) -> InterleavingResult:
    """Stamp a verdict onto *result* from *evidence* — the only producer of a pass.

    *result* is the strategy's accumulated statistics (``num_explored``,
    ``unique_interleavings``, ``exhausted``, ...) with ``property_holds=None``
    and no failure recorded; omitted, a fresh result is built from the
    evidence.  Returns *result* with ``property_holds=True`` when the evidence
    certifies, or ``property_holds=None`` plus ``inconclusive_reason`` when
    the exploration was vacuous or degraded.  Contradictory evidence raises
    :class:`FrontrunInternalError`.
    """
    from frontrun.common import InterleavingResult

    if result is None:
        result = InterleavingResult(property_holds=None, num_explored=evidence.executions)
    if result.property_holds is not None or result.counterexample is not None or result.failures:
        raise FrontrunInternalError(
            "certify_pass() called on a result that already carries a verdict or failure: "
            f"property_holds={result.property_holds!r}, counterexample={result.counterexample!r}, "
            f"failures={len(result.failures)}"
        )
    if evidence.executions < 0:
        raise FrontrunInternalError(f"negative completed-execution count in pass evidence: {evidence.executions}")
    if evidence.executions > result.num_explored:
        raise FrontrunInternalError(
            f"pass evidence claims {evidence.executions} completed interleaving(s) but only "
            f"{result.num_explored} were explored"
        )

    # Exhaustiveness is an independent axis: fill in the strategy's claim, or
    # demote an over-claim, but never upgrade a demotion already on the result.
    if result.exhausted is None:
        result.exhausted = evidence.exhausted_claim
    elif evidence.exhausted_claim is False:
        result.exhausted = False

    if evidence.executions == 0:
        if result.num_explored > 0 and not evidence.degradation_events and evidence.vacuous_reason is None:
            raise FrontrunInternalError(
                f"{result.num_explored} interleaving(s) explored but zero completed executions were recorded "
                "and no cause (degradation event / vacuous reason) explains the gap"
            )
        reason = (
            evidence.vacuous_reason
            or "; ".join(evidence.degradation_events)
            or "exploration completed no interleavings; increase the time/iteration budget or reduce the workload"
        )
        return _make_inconclusive(result, reason)

    workers = list(evidence.workers_executed)
    if not workers:
        raise FrontrunInternalError(
            f"{evidence.executions} completed interleaving(s) but no worker-execution evidence was recorded"
        )
    if not all(workers):
        never_ran = [i for i, ran in enumerate(workers) if not ran]
        raise FrontrunInternalError(
            f"{evidence.executions} interleaving(s) reported complete but worker(s) {never_ran} never entered "
            "their bodies; exploration evidence is internally contradictory"
        )
    if evidence.degradation_events:
        return _make_inconclusive(result, "; ".join(evidence.degradation_events))

    result.property_holds = True
    result.inconclusive_reason = None
    return result
