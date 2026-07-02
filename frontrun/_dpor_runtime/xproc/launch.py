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
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from frontrun._dpor_core.worker import WorkerTarget

from . import protocol as proto

_WORKER_MODULE = "frontrun._dpor_runtime.xproc.worker_main"


def _dumps_worker(worker_fn: Callable[[Any], Any], state: Any) -> bytes:
    """Serialise ``(worker_fn, state)`` with dill, raising a clear error on failure.

    dill (unlike stdlib pickle, which ``multiprocessing`` uses) can serialise
    closures and lambdas, so ``execution="process"`` workers need not be
    module-level — matching what thread execution already accepts.
    """
    try:
        import dill
    except ImportError as exc:
        raise ImportError(
            "explore(execution='process') needs the 'dill' package to serialise worker "
            "callables; install it with `pip install frontrun[process]`."
        ) from exc
    try:
        return dill.dumps((worker_fn, state))
    except Exception as exc:  # noqa: BLE001 - surface any serialisation failure clearly
        raise TypeError(
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
) -> None:
    """multiprocessing entry: connect, install interception, run ``worker_fn(state)``.

    Runs in the spawned child. ``payload`` is the dill-serialised ``(worker_fn,
    state)`` pair (dill, not stdlib pickle, so closures/lambdas survive). With
    ``reuse`` the child stays alive and re-runs the target per ITER_START.
    """
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

    def launch(self, targets: Sequence[WorkerTarget]) -> list[Any]:
        _require_file_backed_main()
        # Reuse mode spawns each worker once; later iterations reconnect the same
        # long-lived processes via ITER_START rather than respawning.
        if self._reuse and self._procs is not None:
            return self._procs
        targets = list(targets)
        socket_path = str(targets[0].args[0]) if targets else ""
        state = None if self._reuse else self._state_fn()
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
                ),
                daemon=True,
            )
            for target in targets
        ]
        scrubbed = {k: os.environ.pop(k) for k in ("LD_PRELOAD", "FRONTRUN_IO_FD") if k in os.environ}
        try:
            for p in procs:
                p.start()
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
        return alive

    def any_exited(self, handles: Any) -> bool:
        """Non-destructive: has any worker process crashed (nonzero exit)?

        Only an abnormal exit counts, mirroring ``diagnose``'s nonzero filter: a
        worker that connected, ran no scheduled access, and exited cleanly (0)
        must not fast-fail the accept loop of a co-worker still sending HELLO.
        """
        return any(proc.exitcode not in (None, 0) for proc in handles)

    def diagnose(self, handles: Any) -> str | None:
        """Describe any worker that exited before connecting (nonzero exit code)."""
        parts = [
            f"worker {i}: process exited with code {proc.exitcode}"
            for i, proc in enumerate(handles)
            if proc.exitcode not in (None, 0)
        ]
        return "; ".join(parts) or None


@dataclass(frozen=True)
class Subprocess:
    """A worker to spawn: a ``"module:callable"`` target and its positional args.

    ``args`` are passed to the spawned process as JSON through the environment,
    so they must be JSON-serialisable **and survive a JSON round-trip**: a tuple
    arrives as a list and a dict with non-string keys comes back string-keyed.
    Pass plain scalars / lists / string-keyed dicts, or use
    ``frontrun.explore(execution="process")`` (which pickles) for richer args.
    The callable runs in the child with frontrun's SQL interception routed to
    the coordinator.
    """

    target: str
    args: tuple[Any, ...] = field(default_factory=tuple)


class SubprocessLauncher:
    def __init__(self, specs: list[Subprocess], *, reuse: bool = False) -> None:
        self._specs = specs
        self._reuse = reuse

    def launch(self, targets: Sequence[WorkerTarget]) -> list[subprocess.Popen[bytes]]:
        base_env = self._child_env_base()
        procs: list[subprocess.Popen[bytes]] = []
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
            procs.append(subprocess.Popen([sys.executable, "-m", _WORKER_MODULE], env=env, stderr=subprocess.PIPE))
        return procs

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

    def any_exited(self, handles: Any) -> bool:
        """Non-destructive: has any worker process crashed (nonzero exit)?

        A clean exit (returncode 0) is not a crash and must not fast-fail the
        accept loop; only nonzero or signal deaths do, matching ``diagnose``.
        """
        return any(proc.poll() not in (None, 0) for proc in handles)

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
            err = ""
            if proc.stderr is not None:
                try:
                    err = proc.stderr.read().decode(errors="replace").strip()
                except (OSError, ValueError):
                    err = ""
            last = err.splitlines()[-1] if err else f"exit code {rc}"
            parts.append(f"worker {i}: {last}")
        return "; ".join(parts) or None

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
