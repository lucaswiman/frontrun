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
import inspect
import json
import os

from .proxy import SchedulerProxy
from .worker import _connect_and_serve, _serve_persistent


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


def _reset_iteration_state() -> None:
    """Clear per-connection SQL state that would otherwise leak across reused runs."""
    try:
        from frontrun._sql_cursor import clear_sql_metadata
        from frontrun._sql_endpoint_suppression import clear_permanent_suppressions
        from frontrun._sql_insert_tracker import clear_insert_tracker
        from frontrun._sql_transactions import reset_connection_state

        clear_permanent_suppressions()
        clear_insert_tracker()
        clear_sql_metadata()
        reset_connection_state()
    except ImportError:
        pass


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
    reuse = os.environ.get("FRONTRUN_XPROC_REUSE") == "1"

    fn = _resolve_target(target)

    def run_target(proxy: SchedulerProxy) -> None:
        result = fn(*args)  # type: ignore[operator]
        if inspect.isawaitable(result):
            # Avoid both the false-success verdict and an unawaited-coroutine
            # warning. Cross-process scheduling is sync-only: there is no async
            # scheduler in the child that could make this execution meaningful.
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise TypeError(
                f"explore_processes() target {target!r} returned an awaitable; "
                "async workers are not supported with execution='process'"
            )

    if reuse:
        # Install interception once (it is global and shares the persistent
        # proxy); reset per-connection SQL state before each run.
        _serve_persistent(
            socket_path,
            worker_id,
            run_target,
            on_connect=lambda proxy: _install_interception(proxy, worker_id),
            before_iteration=lambda _msg: _reset_iteration_state(),
        )
    else:

        def body(proxy: SchedulerProxy) -> None:
            _install_interception(proxy, worker_id)
            run_target(proxy)

        _connect_and_serve(socket_path, worker_id, body)


if __name__ == "__main__":
    main()
