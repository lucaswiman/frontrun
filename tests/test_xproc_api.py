"""API-surface tests for cross-process exploration entry points.

Covers (no real worker processes are spawned here — coordinator classes are
stubbed where a call would otherwise launch subprocesses):

* ``explore_processes`` rejects ``max_iterations`` under ``strategy="dpor"``
  (it only bounds the exhaustive coordinator — a silent no-op otherwise).
* ``explore_processes`` wires the DPOR knobs (``stop_on_first`` /
  ``total_timeout`` / ``search`` / ``max_branches``) through to the DPOR
  coordinator, and rejects them when explicitly set with
  ``strategy="exhaustive"``.
* ``_to_interleaving_result`` preserves the structured fields
  (``exhausted`` / ``failure_kind`` / ``failures``) instead of flattening
  everything into the explanation string.
* ``CrossProcessResult`` is exported at the top level (PEP 562 lazy import).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

import frontrun
import frontrun.cross_process
from frontrun._dpor_runtime.xproc.coordinator import CrossProcessResult

_TARGET = "tests.xproc_demo_counter:increment"


def test_process_extra_installs_sql_parser_required_for_xproc_sql() -> None:
    """The documented process extra must be sufficient for SQL exploration.

    Cross-process workers deliberately drop the LD_PRELOAD fallback, so without
    sqlglot ordinary SELECT/UPDATE statements produce no scheduling points and
    a racy execution can be falsely reported as safe and exhausted.
    """
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    process_extra = re.search(r"(?ms)^process\s*=\s*\[(.*?)^\]", pyproject)

    assert process_extra is not None
    assert re.search(r"['\"]sqlglot(?:[^'\"]*)['\"]", process_extra.group(1))


class _RecordingCoordinator:
    """Stands in for DporCrossProcessCoordinator: records kwargs, spawns nothing."""

    captured: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).captured = dict(kwargs)

    def explore(self, **_kwargs: Any) -> CrossProcessResult:
        return CrossProcessResult(ok=True, iterations=0, exhausted=True)


class _UnexpectedCoordinator:
    def __init__(self, **_kwargs: Any) -> None:
        pytest.fail("invalid public bounds must be rejected before coordinator construction")


# ---------------------------------------------------------------------------
# max_iterations must not silently no-op under strategy="dpor"
# ---------------------------------------------------------------------------


def test_explore_processes_rejects_max_iterations_with_dpor(monkeypatch: pytest.MonkeyPatch) -> None:
    # The DPOR branch never reads max_iterations; an explicitly-passed value
    # must raise (before any process is spawned) rather than silently no-op.
    monkeypatch.setattr(frontrun.cross_process, "DporCrossProcessCoordinator", _RecordingCoordinator)
    with pytest.raises(ValueError, match="exhaustive"):
        frontrun.explore_processes(
            frontrun.Subprocess(_TARGET, ("unused.db",)),
            count=2,
            setup=lambda: None,
            invariant=lambda _state: True,
            max_iterations=50,
        )


def test_explore_processes_max_iterations_error_names_the_dpor_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # The message must be actionable: point at max_executions for DPOR runs.
    monkeypatch.setattr(frontrun.cross_process, "DporCrossProcessCoordinator", _RecordingCoordinator)
    with pytest.raises(ValueError, match="max_executions"):
        frontrun.explore_processes(
            frontrun.Subprocess(_TARGET, ("unused.db",)),
            count=2,
            setup=lambda: None,
            invariant=lambda _state: True,
            strategy="dpor",
            max_iterations=50,
        )


def test_explore_processes_wires_exhaustive_step_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontrun.cross_process, "CrossProcessCoordinator", _RecordingCoordinator)
    frontrun.explore_processes(
        frontrun.Subprocess(_TARGET, ("unused.db",)),
        count=2,
        setup=lambda: None,
        invariant=lambda _state: True,
        strategy="exhaustive",
        max_steps_per_run=123,
    )
    assert _RecordingCoordinator.captured["max_steps_per_run"] == 123


def test_explore_processes_rejects_exhaustive_step_limit_with_dpor() -> None:
    with pytest.raises(ValueError, match="exhaustive"):
        frontrun.explore_processes(
            frontrun.Subprocess(_TARGET, ("unused.db",)),
            count=2,
            setup=lambda: None,
            invariant=lambda _state: True,
            max_steps_per_run=123,
        )


@pytest.mark.parametrize(
    ("strategy", "kwargs", "option"),
    [
        ("dpor", {"deadlock_timeout": 0.0}, "deadlock_timeout"),
        ("dpor", {"max_executions": 0}, "max_executions"),
        ("dpor", {"max_branches": 0}, "max_branches"),
        ("dpor", {"total_timeout": 0.0}, "total_timeout"),
        ("dpor", {"preemption_bound": -1}, "preemption_bound"),
        ("exhaustive", {"max_iterations": 0}, "max_iterations"),
        ("exhaustive", {"max_steps_per_run": 0}, "max_steps_per_run"),
    ],
)
def test_explore_processes_rejects_nonpositive_bounds_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    kwargs: dict[str, Any],
    option: str,
) -> None:
    monkeypatch.setattr(frontrun.cross_process, "DporCrossProcessCoordinator", _UnexpectedCoordinator)
    monkeypatch.setattr(frontrun.cross_process, "CrossProcessCoordinator", _UnexpectedCoordinator)

    with pytest.raises(ValueError, match=option):
        frontrun.explore_processes(
            frontrun.Subprocess(_TARGET, ("unused.db",)),
            setup=lambda: None,
            invariant=lambda _state: True,
            strategy=strategy,  # type: ignore[arg-type]
            **kwargs,
        )


# ---------------------------------------------------------------------------
# DPOR knobs: wired through under strategy="dpor", rejected under exhaustive
# ---------------------------------------------------------------------------


def test_explore_processes_wires_dpor_knobs_to_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontrun.cross_process, "DporCrossProcessCoordinator", _RecordingCoordinator)
    frontrun.explore_processes(
        frontrun.Subprocess(_TARGET, ("unused.db",)),
        count=2,
        setup=lambda: None,
        invariant=lambda _state: True,
        stop_on_first=False,
        total_timeout=2.5,
        search="round-robin",
        max_branches=123,
    )
    captured = _RecordingCoordinator.captured
    assert captured["stop_on_first"] is False
    assert captured["total_timeout"] == 2.5
    assert captured["search"] == "round-robin"
    assert captured["max_branches"] == 123


def test_explore_processes_preserves_mapping_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontrun.cross_process, "DporCrossProcessCoordinator", _RecordingCoordinator)
    result = frontrun.explore_processes(
        {
            "checkout": frontrun.Subprocess(_TARGET, ("unused.db",)),
            "inventory": frontrun.Subprocess(_TARGET, ("unused.db",)),
        },
        setup=lambda: None,
        invariant=lambda _state: True,
    )
    assert result.worker_labels == {0: "checkout", 1: "inventory"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stop_on_first": False},
        {"total_timeout": 1.0},
        {"search": "dfs"},
        {"max_branches": 7},
        # Pre-existing rejections, kept as regressions alongside the new knobs.
        {"max_executions": 3},
        {"preemption_bound": 3},
    ],
    ids=lambda kw: next(iter(kw)),
)
def test_explore_processes_exhaustive_rejects_dpor_only_knobs(kwargs: dict[str, Any]) -> None:
    # The exhaustive coordinator has no engine; explicitly-passed DPOR knobs
    # must raise instead of silently doing nothing.
    with pytest.raises(ValueError, match="dpor"):
        frontrun.explore_processes(
            frontrun.Subprocess(_TARGET, ("unused.db",)),
            count=2,
            setup=lambda: None,
            invariant=lambda _state: True,
            strategy="exhaustive",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# _to_interleaving_result must carry the structured fields
# ---------------------------------------------------------------------------


def test_to_interleaving_result_carries_structured_fields() -> None:
    from frontrun.cross_process import _to_interleaving_result

    cp = CrossProcessResult(
        ok=False,
        iterations=5,
        exhausted=False,
        failing_schedule=[0, 1, 0],
        failure="invariant violated",
        failure_kind="invariant",
        accesses=[(0, "sql:counter:1", "read"), (1, "sql:counter:1", "write")],
        failures=[(2, [0, 1, 0]), (5, [1, 0, 0])],
    )
    ir = _to_interleaving_result(cp)
    assert ir.property_holds is False
    assert ir.exhausted is False
    assert ir.failure_kind == "invariant"
    assert ir.failures == [(2, [0, 1, 0]), (5, [1, 0, 0])]
    assert ir.counterexample == [0, 1, 0]
    # The human-readable explanation is kept alongside the structured fields.
    assert "invariant violated" in (ir.explanation or "")


def test_to_interleaving_result_ok_maps_exhausted() -> None:
    from frontrun.cross_process import _to_interleaving_result

    ir = _to_interleaving_result(CrossProcessResult(ok=True, iterations=3, exhausted=True))
    assert ir.property_holds is True
    assert ir.exhausted is True
    assert ir.failure_kind is None
    assert ir.failures == []
    assert ir.explanation is None


def test_interleaving_result_defaults_leave_new_fields_unset() -> None:
    # Thread-mode constructors don't pass the new fields; defaults must apply.
    from frontrun.common import InterleavingResult

    ir = InterleavingResult(property_holds=True)
    assert ir.exhausted is None
    assert ir.failure_kind is None


# ---------------------------------------------------------------------------
# CrossProcessResult top-level export
# ---------------------------------------------------------------------------


def test_cross_process_result_exported_at_top_level() -> None:
    assert frontrun.CrossProcessResult is CrossProcessResult
    assert "CrossProcessResult" in frontrun.__all__
    assert "CrossProcessResult" in dir(frontrun)
