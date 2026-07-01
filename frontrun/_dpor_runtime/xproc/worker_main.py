"""Entry point for a spawned cross-process worker.

Run as ``python -m frontrun._dpor_runtime.xproc.worker_main`` by
:class:`~frontrun._dpor_runtime.xproc.launch.SubprocessLauncher`. Reads the
coordinator socket path, worker id, and ``module:callable`` target from the
environment, installs frontrun's SQL interception routed to a
:class:`SchedulerProxy`, and runs the target. Between external accesses the
target runs uncontrolled; at each SQL statement the interception layer forwards
a scheduling request to the coordinator over the socket.
"""

from __future__ import annotations

import importlib
import json
import os

from .proxy import SchedulerProxy
from .worker import _connect_and_serve


def _install_interception(proxy: SchedulerProxy, worker_id: int) -> None:
    """Route this process's SQL and Redis access through *proxy*.

    Installs the same context the in-process runner installs (minus opcode
    tracing): the scheduler proxy, the worker's logical id, and the io-reporter.
    SQL cursor patching is global; Redis patching is installed only when the
    ``redis`` package is importable so SQL-only workers need no Redis dependency.
    """
    from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id, set_io_reporter
    from frontrun._sql_cursor import patch_sql

    patch_sql()  # global: every subsequent sqlite3/psycopg connection is traced
    try:
        from frontrun._redis_client import patch_redis

        patch_redis()
    except ImportError:
        pass
    set_dpor_scheduler(proxy)
    set_dpor_thread_id(worker_id)
    set_io_reporter(proxy.io_report)


def _resolve_target(target: str) -> object:
    module_name, sep, attr = target.partition(":")
    if not sep:
        raise ValueError(f"target must be 'module:callable', got {target!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def main() -> None:
    socket_path = os.environ["FRONTRUN_XPROC_SOCKET"]
    worker_id = int(os.environ["FRONTRUN_XPROC_WORKER_ID"])
    target = os.environ["FRONTRUN_XPROC_TARGET"]
    args = tuple(json.loads(os.environ.get("FRONTRUN_XPROC_ARGS", "[]")))

    fn = _resolve_target(target)

    def body(proxy: SchedulerProxy) -> None:
        _install_interception(proxy, worker_id)
        fn(*args)  # type: ignore[operator]

    _connect_and_serve(socket_path, worker_id, body)


if __name__ == "__main__":
    main()
