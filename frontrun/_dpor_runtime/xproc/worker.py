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
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any

from frontrun._dpor_core.worker import WorkerTarget

from . import protocol as proto
from .proxy import SchedulerProxy

# A worker body receives its SchedulerProxy and drives its own external accesses
# through it (directly in tests; via the SQL interception layer in real runs).
WorkerBody = Callable[[SchedulerProxy], None]


def _safe(thunk: Callable[[], None]) -> None:
    """Run *thunk*, swallowing socket errors (the coordinator may have hung up)."""
    try:
        thunk()
    except OSError:
        pass


@contextmanager
def _connected_proxy(socket_path: str, worker_id: int) -> Generator[tuple[SchedulerProxy, socket.socket]]:
    """Connect to the coordinator, announce the worker id, and clean up on exit."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    proxy = SchedulerProxy(sock, worker_id)
    try:
        proxy.hello()
        yield proxy, sock
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _run_iteration(proxy: SchedulerProxy, body: WorkerBody) -> None:
    """Run one iteration of *body(proxy)* and report done/error to the coordinator."""
    proxy.reset()
    try:
        body(proxy)
    except Exception as exc:  # noqa: BLE001 - report any worker failure upstream
        message = f"{type(exc).__name__}: {exc}"
        _safe(lambda: proxy.report_error(message))
    else:
        if proxy.fatal_error is not None:
            _safe(lambda: proxy.report_error(f"RuntimeError: {proxy.fatal_error}"))
        else:
            _safe(proxy.mark_done)


def _connect_and_serve(socket_path: str, worker_id: int, body: WorkerBody) -> None:
    """Connect to the coordinator, run *body(proxy)* once, and report done/error."""
    with _connected_proxy(socket_path, worker_id) as (proxy, _sock):
        _run_iteration(proxy, body)


def _serve_persistent(
    socket_path: str,
    worker_id: int,
    body: WorkerBody,
    *,
    on_connect: Callable[[SchedulerProxy], None] | None = None,
    before_iteration: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Connect once and run *body(proxy)* once per ITER_START until SHUTDOWN.

    ``on_connect`` runs once after HELLO (e.g. install SQL/Redis interception,
    which is global and shares the persistent proxy). ``before_iteration`` sees
    the ITER_START frame and runs before each target call (e.g. refresh
    per-iteration state or reset per-connection SQL metadata). The socket has no
    read timeout: the worker may
    idle arbitrarily long between iterations while the coordinator runs setup
    and checks invariants.
    """
    with _connected_proxy(socket_path, worker_id) as (proxy, sock):
        if on_connect is not None:
            on_connect(proxy)
        while True:
            try:
                msg = proto.recv_msg(sock)
            except OSError:
                break
            if msg is None or msg.get("t") == proto.SHUTDOWN:
                break
            if msg.get("t") != proto.ITER_START:
                continue

            def iteration(iter_proxy: SchedulerProxy) -> None:
                # Target refresh/deserialization is user-controlled work: keep
                # it inside the iteration error boundary so reducer failures
                # become ERROR frames rather than pre-HELLO/disconnect noise.
                if before_iteration is not None:
                    before_iteration(msg)
                body(iter_proxy)

            _run_iteration(proxy, iteration)


class ThreadLauncher:
    """Launch workers as in-process daemon threads (functional-test backend)."""

    def __init__(self, bodies: Sequence[WorkerBody]) -> None:
        self._bodies = list(bodies)

    def launch(self, targets: Sequence[WorkerTarget]) -> list[threading.Thread]:
        threads: list[threading.Thread] = []
        for target in targets:
            wid = target.worker_id
            socket_path = str(target.args[0])
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

    def join(self, handles: Any, timeout: float) -> list[threading.Thread]:
        for t in handles:
            t.join(timeout)
        return [t for t in handles if t.is_alive()]


class PersistentThreadLauncher:
    """In-process persistent-worker backend (functional tests for reuse mode).

    Spawns one long-lived thread per worker on the first ``launch`` call; later
    calls return the same threads. The coordinator drives iterations by sending
    ITER_START / SHUTDOWN frames over the sockets, exactly as for subprocesses.
    """

    def __init__(self, bodies: Sequence[WorkerBody]) -> None:
        self._bodies = list(bodies)
        self._threads: list[threading.Thread] = []

    def launch(self, targets: Sequence[WorkerTarget]) -> list[threading.Thread]:
        if not self._threads:
            for target in targets:
                wid = target.worker_id
                socket_path = str(target.args[0])
                body = self._bodies[wid]
                t = threading.Thread(
                    target=_serve_persistent,
                    args=(socket_path, wid, body),
                    name=f"xproc-persistent-{wid}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
        return self._threads

    def join(self, handles: Any, timeout: float) -> list[threading.Thread]:
        for t in self._threads:
            t.join(timeout)
        return [t for t in self._threads if t.is_alive()]
