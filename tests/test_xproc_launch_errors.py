"""Launch-time failure UX for cross-process workers.

``execution="process"`` serialises workers with dill, so closures and lambdas
work (not just module-level functions); only genuinely unserialisable captures
(open sockets, ...) fail, and then with a clear frontrun-level message. A bad
``module:callable`` subprocess target surfaces the child's real error via
``diagnose()`` instead of a misleading ``TimeoutError``.
"""

from __future__ import annotations

import socket

import pytest

from frontrun._dpor_runtime.xproc.dpor_coordinator import _connection_failure, _launch_error
from frontrun._dpor_runtime.xproc.launch import Subprocess, SubprocessLauncher, _dumps_worker


class _FakeLauncher:
    def __init__(self, detail: str | None) -> None:
        self._detail = detail

    def diagnose(self, handles) -> str | None:  # noqa: ARG002 - handles unused in fake
        return self._detail


def test_dumps_worker_handles_closures() -> None:
    # dill serialises closures (stdlib pickle would not), so a locally-defined
    # worker capturing state round-trips — the parity win over module-level-only.
    import dill

    def make(bump: int):
        def worker(state) -> int:
            return state + bump

        return worker

    fn, state = dill.loads(_dumps_worker(make(100), 1))
    assert fn(state) == 101


def test_dumps_worker_rejects_truly_unserialisable() -> None:
    # An open socket in the setup() state can't be serialised even by dill; the
    # error must be a clear frontrun message, not a raw pickling traceback.
    with pytest.raises(TypeError, match="could not serialise"):
        _dumps_worker(lambda state: state, socket.socket())


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
