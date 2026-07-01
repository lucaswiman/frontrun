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
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

_WORKER_MODULE = "frontrun._dpor_runtime.xproc.worker_main"


@dataclass(frozen=True)
class Subprocess:
    """A worker to spawn: a ``"module:callable"`` target and its positional args.

    ``args`` must be JSON-serialisable — they are passed to the spawned process
    through the environment. The callable runs in the child with frontrun's SQL
    interception routed to the coordinator.
    """

    target: str
    args: tuple[Any, ...] = field(default_factory=tuple)


class SubprocessLauncher:
    def __init__(self, specs: list[Subprocess]) -> None:
        self._specs = specs

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
