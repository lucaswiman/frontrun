"""Tests that frontrun-internal errors are not swallowed as user crashes / dropped accesses.

Each test injects a *frontrun-internal* failure (as opposed to a legitimate user
worker crash) and asserts it surfaces instead of being silently absorbed.  These
are the red/green regressions for the exception-narrowing soundness fixes:

- HIGH-1: the LOAD_ATTR handler guarded the frontrun-internal weak-read
  ``_report_access`` together with the user-facing attribute resolution, so an
  internal failure silently dropped the access (under-merging -> false pass).
- HIGH-2: the sync DPOR reproduction loop caught ``Exception`` around its own
  replay engine, so an internal bug counted as "not a reproduction".
- HIGH-3: the random bytecode reproduction loop had the same blanket catch.
"""

from __future__ import annotations

import pytest

import frontrun

# ---------------------------------------------------------------------------
# HIGH-1: LOAD_ATTR handler must not swallow _report_access failures
# ---------------------------------------------------------------------------


def test_load_attr_weak_read_report_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bug in the frontrun-internal weak-read ``_report_access`` fired by
    LOAD_ATTR must surface rather than being swallowed by the guard that only
    exists to absorb user descriptor side effects during ``_safe_getattr``.

    Old behavior: the ``try/except Exception`` around ``_safe_getattr`` also
    wrapped the weak-read ``_report_access`` call, so an internal failure was
    caught, ``None`` was pushed, the access was dropped, and the worker
    continued -- a classic under-merge that can produce a false pass.
    """
    import frontrun._opcode_observer as obs
    from frontrun._dpor_runtime.replay import _run_dpor_schedule

    real_report = obs._report_access

    def failing_report(engine, execution, thread_id, obj, name, lock, sids, kind):  # type: ignore[no-untyped-def]
        if kind == "weak_read":
            raise RuntimeError("injected weak-read _report_access failure")
        return real_report(engine, execution, thread_id, obj, name, lock, sids, kind)

    monkeypatch.setattr(obs, "_report_access", failing_report)

    class State:
        def __init__(self) -> None:
            self.data = [1, 2, 3]
            self.flag = 0

    def worker(s: State) -> None:
        d = s.data  # LOAD_ATTR loads a list -> triggers the weak-read report
        s.flag = len(d)

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - message asserted below
        _run_dpor_schedule([0], State, [worker], detect_io=False)

    # The injected internal failure must have propagated (possibly wrapped in a
    # worker-execution wrapper), not been swallowed inside the LOAD_ATTR handler.
    surfaced = str(excinfo.value) + str(getattr(excinfo.value, "cause", ""))
    assert "injected weak-read _report_access failure" in surfaced


# ---------------------------------------------------------------------------
# HIGH-2: sync DPOR reproduction must let internal engine errors propagate
# ---------------------------------------------------------------------------


def test_dpor_reproduction_internal_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frontrun-internal error inside the replay engine must propagate out of
    the reproduction loop, not be scored as "not a reproduction".

    Old behavior: ``except Exception: continue`` around ``_run_dpor_schedule``
    swallowed the internal error and quietly lowered ``reproduction_successes``.
    """
    from frontrun._dpor_runtime import replay as replay_mod

    class State:
        def __init__(self) -> None:
            self.counter = 0

    def worker(s) -> None:  # type: ignore[no-untyped-def]
        x = s.counter
        s.counter = x

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("frontrun-internal bug in _run_dpor_schedule")

    monkeypatch.setattr(replay_mod, "_run_dpor_schedule", boom)

    with pytest.raises(RuntimeError, match="frontrun-internal bug in _run_dpor_schedule"):
        replay_mod._reproduce_dpor_counterexample(
            schedule_list=[0, 1, 0, 1, 0, 1, 0, 1],
            setup=State,
            threads=[worker, worker],
            timeout_per_run=5.0,
            deadlock_timeout=2.0,
            reproduce_on_failure=3,
            lock_timeout=None,
            invariant=lambda s: s.counter == 2,
            detect_io=False,
        )


def test_dpor_reproduction_user_worker_crash_is_absorbed() -> None:
    """Guard: a genuine user worker crash during replay is still absorbed as a
    non-reproduction (attempts counted, zero successes), never propagated."""
    from frontrun._dpor_runtime.replay import _reproduce_dpor_counterexample

    class State:
        def __init__(self) -> None:
            self.counter = 0

    def crashing_worker(s) -> None:  # type: ignore[no-untyped-def]
        s.counter = s.counter + 1
        raise ValueError("user worker crash during replay")

    attempts, successes = _reproduce_dpor_counterexample(
        schedule_list=[0, 1, 0, 1],
        setup=State,
        threads=[crashing_worker, crashing_worker],
        timeout_per_run=5.0,
        deadlock_timeout=2.0,
        reproduce_on_failure=3,
        lock_timeout=None,
        invariant=lambda s: s.counter == 99,  # never satisfied
        detect_io=False,
    )
    assert attempts == 3
    assert successes == 0


# ---------------------------------------------------------------------------
# HIGH-3: random bytecode reproduction must let internal errors propagate
# ---------------------------------------------------------------------------


def test_bytecode_reproduction_internal_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frontrun-internal error inside the random-bytecode replay must propagate
    out of the reproduction loop, not be scored as "not a reproduction".

    Old behavior: ``except Exception: pass`` around ``run_with_schedule`` +
    ``check_invariant`` swallowed the internal error.
    """
    import frontrun.bytecode as bc

    real_rws = bc.run_with_schedule

    def fake_rws(*args, **kwargs):  # type: ignore[no-untyped-def]
        # The exploration call passes ``_recorded_schedule``; the reproduction
        # call does not.  Only fail the reproduction path so exploration still
        # finds the real counterexample normally.
        if "_recorded_schedule" in kwargs:
            return real_rws(*args, **kwargs)
        raise RuntimeError("frontrun-internal bug during bytecode replay")

    monkeypatch.setattr(bc, "run_with_schedule", fake_rws)

    class State:
        def __init__(self) -> None:
            self.value = 0

    def worker(s: State) -> None:
        tmp = s.value
        s.value = tmp + 1

    with pytest.raises(RuntimeError, match="frontrun-internal bug during bytecode replay"):
        frontrun.explore_random(
            State,
            [worker, worker],
            lambda s: s.value == 2,
            max_attempts=200,
            reproduce_on_failure=3,
            detect_io=False,
            seed=1234,
        )
