"""Real-subprocess launcher for cross-process exploration.

``SubprocessLauncher`` spawns one OS process per worker, each running
``python -m frontrun._dpor_runtime.xproc.worker_main`` with the coordinator
socket path, worker id, and ``module:callable`` target passed through the
environment. It implements the same ``Launcher`` interface the coordinator uses
for the in-process ``ThreadLauncher``, so the coordinator's scheduling logic is
identical across both.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

_WORKER_MODULE = "frontrun._dpor_runtime.xproc.worker_main"


def _mp_worker_entry(socket_path: str, worker_id: int, worker_fn: Callable[[Any], Any], state: Any) -> None:
    """multiprocessing entry: connect, install interception, run ``worker_fn(state)``.

    Runs in the spawned child. Mirrors ``worker_main`` but receives the worker
    callable and the (picklable) setup state directly via multiprocessing pickling
    instead of a ``module:callable`` string + env-encoded args.
    """
    from .worker import _connect_and_serve
    from .worker_main import _install_interception

    def body(proxy: Any) -> None:
        _install_interception(proxy, worker_id)
        worker_fn(state)

    _connect_and_serve(socket_path, worker_id, body)


class MpLauncher:
    """Spawn Python worker callables via ``multiprocessing`` (the primary backend).

    ``worker_fns`` are plain picklable callables (module-level functions, same as
    threads); ``state_fn`` yields the current picklable state produced by
    ``setup()``, pickled to each worker. Uses the ``spawn`` start method and
    scrubs ``LD_PRELOAD`` / ``FRONTRUN_IO_FD`` so children do not inherit the
    C-level I/O preload (whose event pipe has no reader here).
    """

    def __init__(self, worker_fns: Sequence[Callable[[Any], Any]], state_fn: Callable[[], Any]) -> None:
        self._worker_fns = list(worker_fns)
        self._state_fn = state_fn
        self._ctx = multiprocessing.get_context("spawn")

    def launch(self, socket_path: str, worker_ids: list[int]) -> list[Any]:
        state = self._state_fn()
        scrubbed = {k: os.environ.pop(k) for k in ("LD_PRELOAD", "FRONTRUN_IO_FD") if k in os.environ}
        try:
            procs = [
                self._ctx.Process(
                    target=_mp_worker_entry,
                    args=(socket_path, wid, self._worker_fns[wid], state),
                    daemon=True,
                )
                for wid in worker_ids
            ]
            for p in procs:
                p.start()
        finally:
            os.environ.update(scrubbed)
        return procs

    def join(self, handles: Any, timeout: float) -> None:
        for proc in handles:
            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)


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

    def launch(self, socket_path: str, worker_ids: list[int]) -> list[subprocess.Popen[bytes]]:
        base_env = self._child_env_base()
        procs: list[subprocess.Popen[bytes]] = []
        for wid in worker_ids:
            spec = self._specs[wid]
            env = dict(base_env)
            env["FRONTRUN_XPROC_SOCKET"] = socket_path
            env["FRONTRUN_XPROC_WORKER_ID"] = str(wid)
            env["FRONTRUN_XPROC_TARGET"] = spec.target
            env["FRONTRUN_XPROC_ARGS"] = json.dumps(list(spec.args))
            if self._reuse:
                env["FRONTRUN_XPROC_REUSE"] = "1"
            procs.append(subprocess.Popen([sys.executable, "-m", _WORKER_MODULE], env=env))
        return procs

    def join(self, handles: Any, timeout: float) -> None:
        for proc in handles:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
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
