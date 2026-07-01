"""Launch-time failure UX for cross-process workers.

The first mistakes a new user makes are (1) passing a non-picklable worker to
``execution="process"`` and (2) naming a target module that does not import.
Both used to surface as an inscrutable pickling traceback or a misleading
``TimeoutError`` with the real cause discarded. These tests pin the friendlier
behaviour: a clear frontrun-level message and, for subprocesses, the child's
actual error folded into the diagnostic.
"""

from __future__ import annotations

import pytest

from frontrun._dpor_runtime.xproc.dpor_coordinator import _connection_failure, _launch_error
from frontrun._dpor_runtime.xproc.launch import MpLauncher, Subprocess, SubprocessLauncher


class _FakeLauncher:
    def __init__(self, detail: str | None) -> None:
        self._detail = detail

    def diagnose(self, handles) -> str | None:  # noqa: ARG002 - handles unused in fake
        return self._detail


def test_mplauncher_rejects_unpicklable_worker(tmp_path) -> None:
    # A closure defined here is not picklable; execution="process" requires
    # module-level workers. The launcher must say so, not dump a raw
    # PicklingError from deep in multiprocessing.
    def local_worker(state) -> None:  # noqa: ARG001 - unpicklable by construction
        return None

    launcher = MpLauncher([local_worker], state_fn=lambda: None)
    with pytest.raises(TypeError, match="picklable"):
        launcher.launch(str(tmp_path / "s.sock"), [0])


def test_subprocess_launcher_diagnoses_bad_target(tmp_path) -> None:
    # A target in a module that does not exist makes the child exit immediately
    # with ModuleNotFoundError. diagnose() must recover that real cause from the
    # child's stderr rather than leaving the coordinator to guess "timeout".
    launcher = SubprocessLauncher([Subprocess("frontrun_no_such_module:go")])
    handles = launcher.launch(str(tmp_path / "s.sock"), [0])
    for proc in handles:
        proc.wait(timeout=10)
    detail = launcher.diagnose(handles)
    assert detail is not None
    assert "No module named" in detail or "ModuleNotFoundError" in detail


def test_connection_failure_folds_in_launcher_diagnosis() -> None:
    # A connect timeout enriched with the launcher's diagnosis must report the
    # real cause (no leaked internal exception-class name).
    enriched = _launch_error(_FakeLauncher("worker 0: ModuleNotFoundError: no mod"), None, TimeoutError("timed out"))
    result = _connection_failure(enriched, iterations=1)
    assert result.failure_kind == "worker_error"
    assert "ModuleNotFoundError" in (result.failure or "")
    assert "_WorkerLaunchError" not in (result.failure or "")


def test_connection_failure_without_diagnosis_is_plain() -> None:
    # No diagnosis available -> fall back to the bare exception description.
    same = _launch_error(_FakeLauncher(None), None, TimeoutError("timed out"))
    result = _connection_failure(same, iterations=1)
    assert "TimeoutError" in (result.failure or "")
