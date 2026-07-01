"""Worker bootstrap + in-process launcher for cross-process exploration.

``_connect_and_serve`` is the worker entry shared by every backend: connect to
the coordinator, announce the worker id, install a :class:`SchedulerProxy`, run
the target, and report completion/errors. ``ThreadLauncher`` runs workers as
in-process threads (used by functional tests); the subprocess launcher for real
cross-process runs lives in ``launch.py``.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .proxy import SchedulerProxy

# A worker body receives its SchedulerProxy and drives its own external accesses
# through it (directly in tests; via the SQL interception layer in real runs).
WorkerBody = Callable[[SchedulerProxy], None]


def _connect_and_serve(socket_path: str, worker_id: int, body: WorkerBody) -> None:
    """Connect to the coordinator, run *body(proxy)*, and report done/error."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    proxy = SchedulerProxy(sock, worker_id)
    try:
        proxy.hello()
        try:
            body(proxy)
        except Exception as exc:  # noqa: BLE001 - report any worker failure upstream
            message = f"{type(exc).__name__}: {exc}"
            _safe(lambda: proxy.report_error(message))
        else:
            _safe(proxy.mark_done)
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _safe(thunk: Callable[[], None]) -> None:
    """Run *thunk*, swallowing socket errors (the coordinator may have hung up)."""
    try:
        thunk()
    except OSError:
        pass


class ThreadLauncher:
    """Launch workers as in-process daemon threads (functional-test backend)."""

    def __init__(self, bodies: Sequence[WorkerBody]) -> None:
        self._bodies = list(bodies)

    def launch(self, socket_path: str, worker_ids: list[int]) -> list[threading.Thread]:
        threads: list[threading.Thread] = []
        for wid in worker_ids:
            body = self._bodies[wid]
            t = threading.Thread(
                target=_connect_and_serve,
                args=(socket_path, wid, body),
                name=f"xproc-worker-{wid}",
                daemon=True,
            )
            t.start()
            threads.append(t)
        return threads

    def join(self, handles: Any, timeout: float) -> None:
        for t in handles:
            t.join(timeout)
