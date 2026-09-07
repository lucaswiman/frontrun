"""Shared assertions for the sync and async clock differential oracles."""

from collections.abc import Callable
from typing import Any

from frontrun.common import InterleavingResult


def certified_pass(result: InterleavingResult, cap: int) -> bool:
    """Recognize a full-coverage pass without treating a cap hit as proof."""
    return bool(result.property_holds and result.exhausted is not False and result.num_explored < cap)


def found_counterexample(result: InterleavingResult) -> bool:
    """Recognize a constructive failure, rather than merely a false flag."""
    return not result.property_holds and result.counterexample is not None


def assert_failure_evidence(result: InterleavingResult, label: str, spec: Any, *, require_failures: bool) -> None:
    assert result.counterexample is not None, f"[{label}] failure without a counterexample schedule; spec={spec}"
    assert result.explanation is not None, f"[{label}] failure without an explanation; spec={spec}"
    if require_failures:
        assert result.failures, f"[{label}] failure without a failures entry; spec={spec}"
    if result.reproduction_attempts:
        assert result.reproduction_successes == result.reproduction_attempts, (
            f"[{label}] counterexample replay reproduced only "
            f"{result.reproduction_successes}/{result.reproduction_attempts}; spec={spec}"
        )


def run_clock_oracle(
    spec: Any,
    explore_dpor: Callable[..., InterleavingResult],
    explore_random_real: Callable[[Any], InterleavingResult],
    *,
    cap: int,
    timed_ops: bool = False,
    require_failures: bool = False,
) -> None:
    """Run the common virtual/explored/real clock oracle for either driver."""
    virtual = explore_dpor(spec, "virtual")
    virtual_again = explore_dpor(spec, "virtual")
    assert virtual.property_holds == virtual_again.property_holds, f"virtual-clock outcome is nondeterministic: {spec}"
    if not timed_ops:
        assert virtual.num_explored == virtual_again.num_explored, (
            f"virtual-clock tree size is nondeterministic: {spec}"
        )
        assert virtual.counterexample == virtual_again.counterexample, (
            f"virtual-clock counterexample is nondeterministic: {spec}"
        )

    explored = explore_dpor(spec, "explored")
    for result, label in ((virtual, "clock=virtual"), (explored, "clock=explored")):
        if found_counterexample(result):
            assert_failure_evidence(result, label, spec, require_failures=require_failures)

    if certified_pass(explored, cap):
        assert not found_counterexample(virtual), (
            f"explored certified a pass but virtual found a counterexample: {virtual.explanation}; spec={spec}"
        )
        real = explore_dpor(spec, "real", reproduce=0)
        assert not found_counterexample(real), (
            f"explored certified a pass but real DPOR found a counterexample: {real.explanation}; spec={spec}"
        )
        random = explore_random_real(spec)
        assert not found_counterexample(random), (
            f"explored certified a pass but random real exploration found a counterexample: {random.explanation}; spec={spec}"
        )
