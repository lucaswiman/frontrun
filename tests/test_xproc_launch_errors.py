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

from frontrun._dpor_core.worker import WorkerTarget
from frontrun._dpor_runtime.xproc.dpor_coordinator import _connection_failure, _launch_error
from frontrun._dpor_runtime.xproc.launch import MpLauncher, Subprocess, SubprocessLauncher, _dumps_worker


class _FakeWorkerSet:
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
    worker_set = SubprocessLauncher([Subprocess("frontrun_no_such_module:go")])
    handles = worker_set.launch([WorkerTarget(worker_id=0, args=(str(tmp_path / "s.sock"),))])
    for proc in handles:
        proc.wait(timeout=10)
    detail = worker_set.diagnose(handles)
    assert detail is not None
    assert "No module named" in detail or "ModuleNotFoundError" in detail


class _FakeProc:
    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode


def test_mp_any_exited_ignores_clean_exit() -> None:
    # A worker that connected, ran no scheduled access, and exited cleanly (0)
    # must NOT trip the fast-fail "worker exited before connecting" path while a
    # co-worker's HELLO is still being accepted — only a genuine crash (nonzero)
    # should. This matches diagnose()'s nonzero filter.
    ls = MpLauncher([lambda state: None], state_fn=lambda: None)
    assert ls.any_exited([_FakeProc(0)]) is False
    assert ls.any_exited([_FakeProc(None), _FakeProc(0)]) is False
    assert ls.any_exited([_FakeProc(1)]) is True
    assert ls.any_exited([_FakeProc(None), _FakeProc(-9)]) is True


class _FakePopen:
    def __init__(self, rc: int | None) -> None:
        self._rc = rc

    def poll(self) -> int | None:
        return self._rc


def test_subprocess_any_exited_ignores_clean_exit() -> None:
    # Same clean-exit rule for the subprocess backend: a returncode of 0 is not a
    # crash and must not fast-fail the accept loop.
    ls = SubprocessLauncher([Subprocess("pkg.mod:go")])
    assert ls.any_exited([_FakePopen(0)]) is False
    assert ls.any_exited([_FakePopen(None)]) is False
    assert ls.any_exited([_FakePopen(1)]) is True
    assert ls.any_exited([_FakePopen(-9)]) is True


def test_mp_launcher_rejects_stdin_main(monkeypatch, tmp_path) -> None:
    # multiprocessing's spawn start method cannot re-import a stdin/-c __main__;
    # fail before starting children so users get a frontrun-level explanation.
    import __main__

    monkeypatch.setattr(__main__, "__file__", "<stdin>", raising=False)
    worker_set = MpLauncher([lambda state: None], state_fn=lambda: None)
    with pytest.raises(RuntimeError, match="file-backed Python module"):
        worker_set.launch([WorkerTarget(worker_id=0, args=(str(tmp_path / "s.sock"),))])


def test_connection_failure_folds_in_launcher_diagnosis() -> None:
    # A connect timeout enriched with the WorkerSet's diagnosis must report the
    # real cause (no leaked internal exception-class name).
    enriched = _launch_error(_FakeWorkerSet("worker 0: ModuleNotFoundError: no mod"), None, TimeoutError("timed out"))
    result = _connection_failure(enriched, iterations=1)
    assert result.failure_kind == "worker_error"
    assert "ModuleNotFoundError" in (result.failure or "")
    assert "_WorkerLaunchError" not in (result.failure or "")


def test_connection_failure_without_diagnosis_is_plain() -> None:
    # No diagnosis available -> fall back to the bare exception description.
    same = _launch_error(_FakeWorkerSet(None), None, TimeoutError("timed out"))
    result = _connection_failure(same, iterations=1)
    assert "TimeoutError" in (result.failure or "")
