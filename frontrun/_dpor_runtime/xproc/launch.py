"""Real-subprocess launcher for cross-process exploration.

``SubprocessLauncher`` spawns one OS process per worker, each running
``python -m frontrun._dpor_runtime.xproc.worker_main`` with the coordinator
socket path, worker id, and ``module:callable`` target passed through the
environment. It implements the shared ``WorkerSet`` launch/join port, so the
coordinator's scheduling logic is identical across thread-backed functional
tests and real subprocess runs.
"""

from __future__ import annotations

import base64
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from frontrun._dpor_core.worker import WorkerTarget

from . import protocol as proto

_WORKER_MODULE = "frontrun._dpor_runtime.xproc.worker_main"


class WorkerSerializationError(RuntimeError):
    """``(worker_fn, state)`` could not be serialised for a subprocess worker.

    Raised only by :func:`_dumps_worker` (dill missing, or a payload even dill
    cannot serialise), so coordinators can convert exactly this failure into a
    structured ``worker_error`` result without also swallowing unrelated
    ``TypeError`` / ``ImportError`` bugs from the launch machinery.
    """


class WorkerTerminationError(RuntimeError):
    """Poisoned worker processes survived forced termination."""


def _terminate_procs(procs: Sequence[Any], timeout: float = 1.0) -> None:
    """Terminate/kill and reap already-started multiprocessing children.

    Used to clean up after a partial launch so a spawn failure mid-loop never
    orphans the children that did start. Every child is checked after SIGTERM
    and SIGKILL: replacement workers must never overlap a surviving old child.
    """
    for proc in procs:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    deadline = time.monotonic() + max(0.0, timeout)
    for proc in procs:
        try:
            proc.join(max(0.0, deadline - time.monotonic()))
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    survivors = [proc for proc in procs if proc.is_alive()]
    for proc in survivors:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - final liveness check is authoritative
            pass
    kill_deadline = time.monotonic() + max(0.0, timeout)
    for proc in survivors:
        try:
            proc.join(max(0.0, kill_deadline - time.monotonic()))
        except Exception:  # noqa: BLE001 - final liveness check is authoritative
            pass
    survivors = [proc for proc in survivors if proc.is_alive()]
    if survivors:
        raise WorkerTerminationError(
            f"{len(survivors)} worker process(es) still alive after terminate/kill; refusing to relaunch"
        )


def _make_stderr_file(worker_id: int) -> str:
    """Create an empty temp file to capture one child's stderr; return its path.

    A file, not a pipe: nobody drains the capture while the child runs, so a
    PIPE would block a worker that writes more than the pipe buffer (~64 KiB on
    Linux) — it would never finish, and join() would kill it and misreport a
    hang. Files absorb unbounded stderr without back-pressure.
    """
    fd, path = tempfile.mkstemp(prefix="frontrun-xproc-stderr-", suffix=f".w{worker_id}")
    os.close(fd)  # children reopen by path (os.dup2) or inherit a fresh handle
    return path


# Serialises MpLauncher.launch's temporary os.environ scrub (see launch()).
_env_scrub_lock = threading.Lock()

# Matches a traceback's final "SomeError: message" line (possibly dotted, e.g.
# "pkg.mod.SomeError: ..."), so trailing post-crash output does not hide it.
_TRACEBACK_ERROR_RE = re.compile(r"^[\w.]*\w(Error|Exception)\b")


def _stderr_last_line(path: str | None) -> str | None:
    """Return the most diagnostic captured stderr line, or ``None`` if unreadable/empty.

    Prefers the last line that looks like a traceback's error line — a crashing
    child can print more after its traceback (atexit handlers, multiprocessing
    teardown chatter) — falling back to the last non-blank line.
    """
    if not path:
        return None
    try:
        with open(path, errors="replace") as fh:
            text = fh.read().strip()
    except OSError:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        if _TRACEBACK_ERROR_RE.match(line):
            return line
    return lines[-1]


def _unlink_all(paths: Sequence[str]) -> None:
    """Best-effort removal of stderr capture files."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _dumps_worker(worker_fn: Callable[[Any], Any], state: Any) -> bytes:
    """Serialise ``(worker_fn, state)`` with dill, raising a clear error on failure.

    dill (unlike stdlib pickle, which ``multiprocessing`` uses) can serialise
    closures and lambdas, so ``execution="process"`` workers need not be
    module-level — matching what thread execution already accepts.
    """
    try:
        import dill
    except ImportError as exc:
        raise WorkerSerializationError(
            "explore(execution='process') needs the 'dill' package to serialise worker "
            "callables; install it with `pip install frontrun[process]`."
        ) from exc
    try:
        return dill.dumps((worker_fn, state))
    except Exception as exc:  # noqa: BLE001 - surface any serialisation failure clearly
        raise WorkerSerializationError(
            "explore(execution='process') could not serialise a worker or the setup() state "
            f"(even with dill): {exc}. Avoid capturing unpicklable objects such as open "
            "connections, sockets, or locks in the worker or the setup() return value."
        ) from exc


def _require_file_backed_main() -> None:
    """Fail clearly when multiprocessing spawn cannot re-import ``__main__``."""
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if isinstance(main_file, str) and main_file and not main_file.startswith("<") and os.path.exists(main_file):
        return
    raise RuntimeError(
        "explore(execution='process') uses multiprocessing 'spawn', which requires the parent "
        "program to run from a file-backed Python module. It cannot be launched from stdin, "
        "`python -c`, or a REPL/notebook cell; put the test in a .py file or use "
        "frontrun.explore_processes() with importable module:callable targets."
    )


def _mp_worker_entry(
    socket_path: str,
    worker_id: int,
    payload: bytes,
    reuse: bool = False,
    stderr_path: str | None = None,
) -> None:
    """multiprocessing entry: connect, install interception, run ``worker_fn(state)``.

    Runs in the spawned child. ``payload`` is the dill-serialised ``(worker_fn,
    state)`` pair (dill, not stdlib pickle, so closures/lambdas survive). With
    ``reuse`` the child stays alive and re-runs the target per ITER_START.

    When ``stderr_path`` is given, fd 2 is redirected to that file *before* any
    work (including ``dill.loads``) so that a pre-HELLO crash — which
    multiprocessing children would otherwise print to the parent's inherited
    stderr — is captured where ``MpLauncher.diagnose`` can recover its real
    cause. The redirect is at the fd level (``os.dup2``) so the interpreter's
    own fatal-traceback writer is captured, not just Python-level ``sys.stderr``.
    """
    if stderr_path:
        fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fd, 2)
        os.close(fd)

    import dill

    from .worker import _connect_and_serve, _serve_persistent
    from .worker_main import _install_interception, _reset_iteration_state

    worker_fn, state = dill.loads(payload)

    def run_target(proxy: Any) -> None:
        worker_fn(state)

    if reuse:

        def refresh_iteration(iter_msg: dict[str, Any]) -> None:
            nonlocal worker_fn, state
            encoded = iter_msg.get("payload")
            if isinstance(encoded, str):
                worker_fn, state = dill.loads(base64.b64decode(encoded.encode("ascii")))
            _reset_iteration_state()

        _serve_persistent(
            socket_path,
            worker_id,
            run_target,
            on_connect=lambda proxy: _install_interception(proxy, worker_id),
            before_iteration=refresh_iteration,
        )
    else:

        def body(proxy: Any) -> None:
            _install_interception(proxy, worker_id)
            worker_fn(state)

        _connect_and_serve(socket_path, worker_id, body)


class MpLauncher:
    """Spawn Python worker callables via ``multiprocessing`` (the primary backend).

    ``worker_fns`` are callables serialised with dill, so closures and lambdas
    work as well as module-level functions (same as thread execution);
    ``state_fn`` yields the current state produced by ``setup()``, serialised to
    each worker. Uses the ``spawn`` start method and scrubs ``LD_PRELOAD`` /
    ``FRONTRUN_IO_FD`` so children do not inherit the C-level I/O preload (whose
    event pipe has no reader here).
    """

    def __init__(
        self,
        worker_fns: Sequence[Callable[[Any], Any]],
        state_fn: Callable[[], Any],
        *,
        reuse: bool = False,
    ) -> None:
        self._worker_fns = list(worker_fns)
        self._state_fn = state_fn
        self._reuse = reuse
        self._ctx = multiprocessing.get_context("spawn")
        self._procs: list[Any] | None = None
        # Per-worker temp files capturing each child's stderr (parallel to the
        # launched procs), so diagnose() can surface a pre-HELLO crash the way
        # SubprocessLauncher does from its stderr pipe. multiprocessing children
        # otherwise inherit the parent's stderr, hiding the real cause.
        self._stderr_files: list[str] = []

    def launch(self, targets: Sequence[WorkerTarget]) -> list[Any]:
        _require_file_backed_main()
        # Reuse mode spawns each worker once; later iterations reconnect the same
        # long-lived processes via ITER_START rather than respawning.
        if self._reuse and self._procs is not None:
            return self._procs
        targets = list(targets)
        socket_path = str(targets[0].args[0]) if targets else ""
        state = None if self._reuse else self._state_fn()
        # Fresh per-worker stderr capture files for this launch (drop the
        # previous iteration's so they do not accumulate).
        self._cleanup_stderr_files()
        self._stderr_files = [_make_stderr_file(target.worker_id) for target in targets]
        # Serialise with dill up front (raises a clear error for genuinely
        # unserialisable workers/state) so children receive plain bytes that
        # multiprocessing's own stdlib pickling handles trivially.
        procs = [
            self._ctx.Process(
                target=_mp_worker_entry,
                args=(
                    socket_path,
                    target.worker_id,
                    _dumps_worker(self._worker_fns[target.worker_id], state),
                    self._reuse,
                    self._stderr_files[idx],
                ),
                daemon=True,
            )
            for idx, target in enumerate(targets)
        ]
        # Scrub the C-level preload from the children's environment. This must
        # happen parent-side: multiprocessing has no per-child env parameter,
        # and a child-side scrub would come too late — LD_PRELOAD is consumed
        # by the dynamic loader at exec, and the preload library reads
        # FRONTRUN_IO_FD lazily at the first intercepted call (crates/io
        # get_pipe_fd), which fires during the child interpreter's own startup
        # file reads, long before any Python-level code runs. The module lock
        # serialises concurrent launches so one launch's restore cannot clobber
        # another's scrub (os.environ is process-global).
        with _env_scrub_lock:
            scrubbed = {k: os.environ.pop(k) for k in ("LD_PRELOAD", "FRONTRUN_IO_FD") if k in os.environ}
            try:
                started: list[Any] = []
                try:
                    for p in procs:
                        p.start()
                        started.append(p)
                except BaseException:
                    # A spawn failed mid-loop (e.g. resource exhaustion): the
                    # already-started children would otherwise leak — their worker
                    # socket blocks forever awaiting a GRANT — because the exception
                    # escapes launch() before returning handles for the caller's
                    # try/finally to join. Reap them here before re-raising.
                    _terminate_procs(started)
                    raise
            finally:
                os.environ.update(scrubbed)
        if self._reuse:
            self._procs = procs
        return procs

    def iter_start_message(self, worker_id: int) -> dict[str, Any]:
        """Build the ITER_START frame for one reused multiprocessing worker."""
        state = self._state_fn()
        payload = _dumps_worker(self._worker_fns[worker_id], state)
        return {"t": proto.ITER_START, "payload": base64.b64encode(payload).decode("ascii")}

    def join(self, handles: Any, timeout: float) -> list[Any]:
        alive: list[Any] = []
        for proc in handles:
            proc.join(timeout)
            if proc.is_alive():
                alive.append(proc)
                proc.terminate()
                proc.join(1.0)
                if proc.is_alive():
                    # Worker ignored SIGTERM; escalate to SIGKILL so
                    # execution="process" cannot leave a runaway child, matching
                    # SubprocessLauncher.join()'s .kill() escalation.
                    proc.kill()
                    proc.join(1.0)
        return alive

    def terminate(self, handles: Any, timeout: float) -> None:
        """Forcibly retire poisoned persistent children so they can be replaced."""
        _terminate_procs(handles, timeout)
        if handles is self._procs:
            self._procs = None

    def any_exited(self, handles: Any) -> bool:
        """Non-destructive: has any worker process crashed (nonzero exit)?

        Only an abnormal exit counts, mirroring ``diagnose``'s nonzero filter: a
        worker that connected, ran no scheduled access, and exited cleanly (0)
        must not fast-fail the accept loop of a co-worker still sending HELLO.
        """
        return any(proc.exitcode not in (None, 0) for proc in handles)

    def all_exited(self, handles: Any) -> bool:
        """Non-destructive: has *every* worker process exited (any code)?

        Unlike ``any_exited`` this counts clean (0) exits too, so the accept loop
        can fail fast when a target exits before HELLO (e.g. ``sys.exit(0)`` at
        import) instead of waiting the whole connect budget.
        """
        return all(proc.exitcode is not None for proc in handles)

    def diagnose(self, handles: Any) -> str | None:
        """Describe any worker that exited before connecting (nonzero exit code).

        When the child's captured stderr holds a traceback, surface its last
        line (e.g. a ``dill.loads``/``ModuleNotFoundError`` failure) instead of a
        bare exit code, matching ``SubprocessLauncher.diagnose``.
        """
        parts: list[str] = []
        for i, proc in enumerate(handles):
            if proc.exitcode in (None, 0):
                continue
            path = self._stderr_files[i] if i < len(self._stderr_files) else None
            detail = _stderr_last_line(path) or f"process exited with code {proc.exitcode}"
            parts.append(f"worker {i}: {detail}")
        return "; ".join(parts) or None

    def _cleanup_stderr_files(self) -> None:
        """Best-effort removal of the previous launch's stderr capture files."""
        _unlink_all(self._stderr_files)
        self._stderr_files = []

    def __del__(self) -> None:
        # The per-launch cleanup only runs on the *next* launch, so the final
        # iteration's capture files would leak once the exploration finishes.
        # diagnose() may be read after join(), so this finalizer is the earliest
        # safe point to drop them.
        try:
            self._cleanup_stderr_files()
        except Exception:  # noqa: BLE001 - interpreter-shutdown best effort
            pass


@dataclass(frozen=True)
class Subprocess:
    """A worker to spawn: a ``"module:callable"`` target and its positional args.

    ``args`` are passed to the spawned process as JSON through the environment,
    so they must be JSON-serialisable **and survive a JSON round-trip**: a tuple
    arrives as a list and a dict with non-string keys comes back string-keyed.
    Pass plain scalars / lists / string-keyed dicts, or use
    ``frontrun.explore(execution="process")`` (which pickles) for richer args.
    The callable must be synchronous; async/awaitable targets are rejected.
    It runs in the child with frontrun's SQL interception routed to the
    coordinator.
    """

    target: str
    args: tuple[Any, ...] = field(default_factory=tuple)


class SubprocessLauncher:
    def __init__(self, specs: list[Subprocess], *, reuse: bool = False) -> None:
        self._specs = specs
        self._reuse = reuse
        # Per-worker temp files capturing each child's stderr (parallel to the
        # launched procs). Files, not PIPEs: an undrained PIPE blocks a child
        # that writes more than the pipe buffer, deadlocking the exploration.
        self._stderr_files: list[str] = []

    def launch(self, targets: Sequence[WorkerTarget]) -> list[subprocess.Popen[bytes]]:
        base_env = self._child_env_base()
        self._cleanup_stderr_files()
        procs: list[subprocess.Popen[bytes]] = []
        try:
            for target in targets:
                wid = target.worker_id
                socket_path = str(target.args[0])
                spec = self._specs[wid]
                env = dict(base_env)
                env["FRONTRUN_XPROC_SOCKET"] = socket_path
                env["FRONTRUN_XPROC_WORKER_ID"] = str(wid)
                env["FRONTRUN_XPROC_TARGET"] = spec.target
                env["FRONTRUN_XPROC_ARGS"] = json.dumps(list(spec.args))
                if self._reuse:
                    env["FRONTRUN_XPROC_REUSE"] = "1"
                self._stderr_files.append(_make_stderr_file(wid))
                with open(self._stderr_files[-1], "wb") as stderr_fh:
                    procs.append(subprocess.Popen([sys.executable, "-m", _WORKER_MODULE], env=env, stderr=stderr_fh))
        except BaseException:
            # A Popen failed mid-loop: kill and reap the children that did start
            # so they cannot leak and hang on a GRANT that never arrives (the
            # exception escapes launch() before the caller can join them).
            self._reap_partial(procs)
            raise
        return procs

    @staticmethod
    def _reap_partial(procs: Sequence[subprocess.Popen[bytes]]) -> None:
        for proc in procs:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        for proc in procs:
            try:
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def join(self, handles: Any, timeout: float) -> list[subprocess.Popen[bytes]]:
        alive: list[subprocess.Popen[bytes]] = []
        for proc in handles:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                alive.append(proc)
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        return alive

    def terminate(self, handles: Any, timeout: float) -> None:
        """Forcibly retire poisoned persistent children so they can be replaced."""
        for proc in handles:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 - final poll verifies death
                pass
        deadline = time.monotonic() + max(0.0, timeout)
        for proc in handles:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:  # noqa: BLE001 - final poll verifies death
                pass
        survivors = [proc for proc in handles if proc.poll() is None]
        if survivors:
            raise WorkerTerminationError(
                f"{len(survivors)} worker process(es) still alive after kill; refusing to relaunch"
            )

    def any_exited(self, handles: Any) -> bool:
        """Non-destructive: has any worker process crashed (nonzero exit)?

        A clean exit (returncode 0) is not a crash and must not fast-fail the
        accept loop; only nonzero or signal deaths do, matching ``diagnose``.
        """
        return any(proc.poll() not in (None, 0) for proc in handles)

    def all_exited(self, handles: Any) -> bool:
        """Non-destructive: has *every* worker process exited (any code)?

        Counts clean (0) exits too, so the accept loop can fail fast when a
        target exits before HELLO (e.g. ``sys.exit(0)`` at import) rather than
        blocking for the full connect budget.
        """
        return all(proc.poll() is not None for proc in handles)

    def diagnose(self, handles: Any) -> str | None:
        """Recover the real cause when a worker died before connecting.

        A bad ``module:callable`` target makes the child exit with a traceback on
        stderr; surface its last line so the coordinator reports e.g.
        ``ModuleNotFoundError`` instead of a bare connection timeout.
        """
        parts: list[str] = []
        for i, proc in enumerate(handles):
            rc = proc.poll()
            if rc is None or rc == 0:
                continue
            path = self._stderr_files[i] if i < len(self._stderr_files) else None
            last = _stderr_last_line(path) or f"exit code {rc}"
            parts.append(f"worker {i}: {last}")
        return "; ".join(parts) or None

    def _cleanup_stderr_files(self) -> None:
        """Best-effort removal of the previous launch's stderr capture files."""
        _unlink_all(self._stderr_files)
        self._stderr_files = []

    def __del__(self) -> None:
        # Same finalizer contract as MpLauncher: the per-launch cleanup only
        # runs on the next launch, so drop the final launch's files here.
        try:
            self._cleanup_stderr_files()
        except Exception:  # noqa: BLE001 - interpreter-shutdown best effort
            pass

    @staticmethod
    def _child_env_base() -> dict[str, str]:
        env = dict(os.environ)
        # Let children import whatever the parent can (target modules on sys.path).
        parent_paths = os.pathsep.join(p for p in sys.path if p)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(x for x in (parent_paths, existing) if x)
        # Phase 1 workers do their interception in Python and talk to the
        # coordinator over the socket; the LD_PRELOAD C-level interception is
        # neither needed nor wanted (its event pipe belongs to the parent CLI
        # wrapper). Scrub it so a worker spawned under `frontrun pytest` cannot
        # block writing preload events nobody is draining.
        env.pop("LD_PRELOAD", None)
        env.pop("FRONTRUN_IO_FD", None)
        return env
