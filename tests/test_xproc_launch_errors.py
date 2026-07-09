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

from frontrun._dpor_core.worker import IterationCustomizer, LivenessProbe, WorkerTarget
from frontrun._dpor_runtime.xproc.dpor_coordinator import _connection_failure, _launch_error
from frontrun._dpor_runtime.xproc.launch import (
    MpLauncher,
    Subprocess,
    SubprocessLauncher,
    _dumps_worker,
    _stderr_last_line,
)


class _FakeWorkerSet:
    def __init__(self, detail: str | None) -> None:
        self._detail = detail

    def any_exited(self, handles) -> bool:  # noqa: ARG002 - handles unused in fake
        return self._detail is not None

    def all_exited(self, handles) -> bool:  # noqa: ARG002 - handles unused in fake
        return self._detail is not None

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


def test_stderr_last_line_prefers_traceback_error_over_trailing_noise(tmp_path) -> None:
    # A crashing child can print output after its traceback (atexit handlers,
    # multiprocessing teardown chatter). diagnose() must surface the actual
    # error line, not whatever happened to be written last.
    path = tmp_path / "stderr.w0"
    path.write_text(
        "Traceback (most recent call last):\n"
        '  File "worker.py", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'nope'\n"
        "worker teardown: closing connections\n"
    )
    assert _stderr_last_line(str(path)) == "ModuleNotFoundError: No module named 'nope'"


def test_stderr_last_line_falls_back_to_last_nonblank_line(tmp_path) -> None:
    path = tmp_path / "stderr.w1"
    path.write_text("some diagnostic output\nfinal words before dying\n\n")
    assert _stderr_last_line(str(path)) == "final words before dying"


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


# --- Fix 4: all_exited (clean exit before HELLO must fail fast) -----------


def test_mp_all_exited_counts_clean_exits() -> None:
    # Unlike any_exited, all_exited counts clean (0) exits: a target that
    # sys.exit(0)s at import before sending HELLO leaves every child exited, so
    # the accept loop can fail fast instead of blocking the whole connect budget.
    ls = MpLauncher([lambda state: None], state_fn=lambda: None)
    assert ls.all_exited([_FakeProc(0)]) is True
    assert ls.all_exited([_FakeProc(1)]) is True
    assert ls.all_exited([_FakeProc(None)]) is False
    assert ls.all_exited([_FakeProc(0), _FakeProc(None)]) is False


def test_subprocess_all_exited_counts_clean_exits() -> None:
    # Same semantics for the subprocess backend.
    ls = SubprocessLauncher([Subprocess("pkg.mod:go")])
    assert ls.all_exited([_FakePopen(0)]) is True
    assert ls.all_exited([_FakePopen(-9)]) is True
    assert ls.all_exited([_FakePopen(None)]) is False
    assert ls.all_exited([_FakePopen(0), _FakePopen(None)]) is False


# --- Fix 3: MpLauncher.diagnose surfaces a child's captured stderr ---------


def test_mp_diagnose_surfaces_child_stderr(tmp_path) -> None:
    # A pre-HELLO crash (e.g. dill.loads / import failure) writes a traceback to
    # the child's redirected stderr file; diagnose() must surface its last line
    # rather than a bare exit code, matching SubprocessLauncher.
    ls = MpLauncher([lambda state: None], state_fn=lambda: None)
    err = tmp_path / "w0.err"
    err.write_text("Traceback (most recent call last):\n  ...\nModuleNotFoundError: No module named 'nope'\n")
    ls._stderr_files = [str(err)]
    detail = ls.diagnose([_FakeProc(1)])
    assert detail is not None
    assert "ModuleNotFoundError: No module named 'nope'" in detail
    # A clean exit is not diagnosed.
    assert ls.diagnose([_FakeProc(0)]) is None


def test_mp_diagnose_falls_back_to_exit_code_without_stderr(tmp_path) -> None:
    # Empty (or missing) capture file -> fall back to the bare exit-code message.
    ls = MpLauncher([lambda state: None], state_fn=lambda: None)
    empty = tmp_path / "w0.err"
    empty.write_text("")
    ls._stderr_files = [str(empty)]
    assert "process exited with code 2" in (ls.diagnose([_FakeProc(2)]) or "")


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


# --- Change 1: partial-launch cleanup (no orphaned children) --------------


class _FakeMpProc:
    """Fake multiprocessing handle recording lifecycle calls."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self._fail_start = fail_start
        self.started = False
        self.terminated = False
        self.killed = False
        self._alive = False

    def start(self) -> None:
        if self._fail_start:
            raise RuntimeError("boom: cannot spawn worker")
        self.started = True
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002 - fake
        pass


class _FakeCtx:
    def __init__(self, procs: list[_FakeMpProc]) -> None:
        self._it = iter(procs)

    def Process(self, *args, **kwargs) -> _FakeMpProc:  # noqa: N802, ARG002 - mimics mp API
        return next(self._it)


def test_mp_launcher_cleans_up_on_partial_launch(monkeypatch, tmp_path) -> None:
    # If spawning the 2nd worker raises, the 1st (already started) child must be
    # terminated before the exception propagates — otherwise it leaks and hangs
    # forever waiting for a GRANT that never comes.
    procs = [_FakeMpProc(), _FakeMpProc(fail_start=True)]
    ws = MpLauncher([lambda state: None, lambda state: None], state_fn=lambda: None)
    monkeypatch.setattr(ws, "_ctx", _FakeCtx(procs))
    sock = str(tmp_path / "s.sock")
    with pytest.raises(RuntimeError, match="boom"):
        ws.launch([WorkerTarget(worker_id=0, args=(sock,)), WorkerTarget(worker_id=1, args=(sock,))])
    assert procs[0].started
    assert procs[0].terminated
    assert not procs[0].is_alive()


class _FakePopenProc:
    def __init__(self) -> None:
        self.killed = False
        self._alive = True

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002 - fake
        return 0


def test_subprocess_launcher_cleans_up_on_partial_launch(monkeypatch, tmp_path) -> None:
    # Same leak for the subprocess backend: if the 2nd Popen construction raises,
    # the 1st spawned child must be killed rather than orphaned.
    from frontrun._dpor_runtime.xproc import launch as launch_mod

    made: list[_FakePopenProc] = []

    def fake_popen(*args, **kwargs):  # noqa: ARG001 - mimics Popen signature
        if made:
            raise RuntimeError("boom: cannot spawn worker")
        proc = _FakePopenProc()
        made.append(proc)
        return proc

    monkeypatch.setattr(launch_mod.subprocess, "Popen", fake_popen)
    ws = SubprocessLauncher([Subprocess("pkg.mod:go"), Subprocess("pkg.mod:go")])
    sock = str(tmp_path / "s.sock")
    with pytest.raises(RuntimeError, match="boom"):
        ws.launch([WorkerTarget(worker_id=0, args=(sock,)), WorkerTarget(worker_id=1, args=(sock,))])
    assert len(made) == 1
    assert made[0].killed


# --- Change 2: MpLauncher.join escalates to SIGKILL -----------------------


class _StubbornMpProc:
    """A handle that ignores terminate() (SIGTERM) and only dies on kill()."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002 - fake
        pass

    def is_alive(self) -> bool:
        return not self.killed

    def terminate(self) -> None:
        self.terminated = True  # ignored: worker keeps running

    def kill(self) -> None:
        self.killed = True


def test_mp_launcher_join_escalates_to_kill() -> None:
    # A worker that ignores SIGTERM must be SIGKILLed, matching
    # SubprocessLauncher; otherwise execution="process" leaves it running.
    ws = MpLauncher([lambda state: None], state_fn=lambda: None)
    proc = _StubbornMpProc()
    alive = ws.join([proc], timeout=0.01)
    assert proc in alive
    assert proc.terminated
    assert proc.killed


# --- stderr capture must not deadlock a chatty worker ----------------------


def test_subprocess_worker_flooding_stderr_still_exits(tmp_path) -> None:
    # A worker that writes more than a pipe buffer (~64 KiB) to stderr must
    # still be able to finish. Capturing stderr with an undrained PIPE blocks
    # the child's write forever: it never exits, join() times out and kills it,
    # and the run misreports a hang. Capture must therefore go to a file. The
    # flood target writes 256 KiB then exits 3, so diagnose() must also recover
    # its final line from the capture.
    worker_set = SubprocessLauncher([Subprocess("tests.xproc_stderr_flood:main")])
    handles = worker_set.launch([WorkerTarget(worker_id=0, args=(str(tmp_path / "s.sock"),))])
    alive = worker_set.join(handles, timeout=8.0)
    assert alive == [], "child blocked writing stderr: capture is deadlock-prone"
    detail = worker_set.diagnose(handles)
    assert detail is not None
    assert "flooded stderr with 256 KiB" in detail


# --- final launch's stderr capture files must not leak ----------------------


def test_mp_launcher_removes_stderr_files_when_collected(tmp_path) -> None:
    # _cleanup_stderr_files only runs on the *next* launch, so without a
    # finalizer the final iteration's capture files leak (one per worker per
    # explore() call). Dropping the launcher must remove them.
    ls = MpLauncher([lambda state: None], state_fn=lambda: None)
    err = tmp_path / "w0.err"
    err.write_text("leftover")
    ls._stderr_files = [str(err)]
    del ls
    assert not err.exists()


def test_subprocess_launcher_removes_stderr_files_when_collected(tmp_path) -> None:
    # Same finalizer contract for the subprocess backend.
    ls = SubprocessLauncher([Subprocess("pkg.mod:go")])
    err = tmp_path / "w0.err"
    err.write_text("leftover")
    ls._stderr_files = [str(err)]
    del ls
    assert not err.exists()


# --- Change 3: typed capability Protocols (rename safety net) --------------


def test_process_launchers_expose_capability_protocols() -> None:
    # These isinstance checks are the safety net: a future rename of any of the
    # capability methods breaks these loudly instead of silently degrading.
    mp = MpLauncher([lambda state: None], state_fn=lambda: None)
    sub = SubprocessLauncher([Subprocess("pkg.mod:go")])
    assert isinstance(mp, LivenessProbe)
    assert isinstance(mp, IterationCustomizer)
    assert isinstance(sub, LivenessProbe)
    assert not isinstance(sub, IterationCustomizer)
