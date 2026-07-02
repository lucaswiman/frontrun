"""Fast dispatch-level tests for the process execution path (no spawning).

These pin argument wiring that would otherwise only be observable through slow
e2e spawns: the execution-dependent ``deadlock_timeout`` default and the
``count=`` shorthand that replicates a single :class:`Subprocess`.
"""

from __future__ import annotations

import pytest

import frontrun
from frontrun import cross_process
from frontrun._dpor_runtime.xproc.launch import Subprocess


def test_process_deadlock_timeout_defaults_to_15(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake(setup, workers, invariant, **kwargs):  # noqa: ARG001
        seen.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(cross_process, "_explore_process", fake)
    out = frontrun.explore(
        setup=lambda: None, workers=lambda s: None, count=2, invariant=lambda s: True, execution="process"
    )
    assert out == "sentinel"
    assert seen["deadlock_timeout"] == 15.0


def test_process_deadlock_timeout_explicit_is_respected(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake(setup, workers, invariant, **kwargs):  # noqa: ARG001
        seen.update(kwargs)
        return None

    monkeypatch.setattr(cross_process, "_explore_process", fake)
    frontrun.explore(
        setup=lambda: None,
        workers=lambda s: None,
        count=2,
        invariant=lambda s: True,
        execution="process",
        deadlock_timeout=3.0,
    )
    assert seen["deadlock_timeout"] == 3.0


def test_explore_processes_count_replicates_single_spec() -> None:
    spec = Subprocess("pkg.mod:go", ("a",))
    specs = cross_process._resolve_specs(spec, count=3)
    assert specs == [spec, spec, spec]


def test_explore_processes_count_rejects_mapping() -> None:
    with pytest.raises(ValueError, match="count"):
        cross_process._resolve_specs({"w0": Subprocess("pkg.mod:go")}, count=2)


def test_process_result_explanation_includes_kind_and_accesses() -> None:
    from frontrun._dpor_runtime.xproc.coordinator import CrossProcessResult

    cpr = CrossProcessResult(
        ok=False,
        iterations=3,
        exhausted=False,
        failing_schedule=[0, 1],
        failure="cross-worker deadlock",
        failure_kind="deadlock",
        accesses=[(0, "sql:accounts:id=1", "write"), (1, "sql:accounts:id=2", "write")],
    )
    ir = cross_process._to_interleaving_result(cpr)
    assert not ir.property_holds
    assert "[deadlock]" in (ir.explanation or "")
    assert "sql:accounts:id=1" in (ir.explanation or "")


def test_process_rejects_silent_noop_kwargs() -> None:
    with pytest.raises(ValueError, match="serializable_invariant"):
        frontrun.explore(
            setup=lambda: None,
            workers=lambda s: None,
            count=2,
            invariant=lambda s: True,
            execution="process",
            serializable_invariant=True,
        )


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("detect_io", False),
        ("patch_sleep", False),
        ("timeout_per_run", 30.0),
        ("reproduce_on_failure", 3),
        ("warn_nondeterministic_sql", False),
    ],
)
def test_process_rejects_more_silent_noop_kwargs(kwarg: str, value: object) -> None:
    # These thread-mode knobs are not plumbed into process workers (state is
    # external; there is no in-process trace/replay), so a non-default value must
    # raise rather than silently no-op — the same porting footgun the existing
    # rejection list closes for serializable_invariant et al.
    with pytest.raises(ValueError, match=kwarg):
        frontrun.explore(
            setup=lambda: None,
            workers=lambda s: None,
            count=2,
            invariant=lambda s: True,
            execution="process",
            **{kwarg: value},
        )


def test_explore_processes_reuse_rejects_exhaustive() -> None:
    with pytest.raises(ValueError, match="reuse_workers"):
        frontrun.explore_processes(
            Subprocess("pkg.mod:go"),
            count=2,
            setup=lambda: None,
            invariant=lambda _state: True,
            strategy="exhaustive",
            reuse_workers=True,
        )
