"""Coordinator for the Phase 4 proof-of-concept: scheduling unmodified workers.

Pairs with ``crates/xproc_sched/frontrun_xproc_sched.c``. Worker processes need
no frontrun code at all — the LD_PRELOAD shim blocks each socket ``send()`` until
this coordinator grants it, so we control the interleaving of *unmodified*
(potentially non-Python) processes at C-level socket granularity.

Wire protocol (one byte each unless noted):

* worker -> coordinator on connect: ``[HELLO][worker_id]`` (2 bytes)
* worker -> coordinator before each send: ``[REQ_SEND][worker_id]`` (2 bytes)
* coordinator -> worker: ``GRANT`` or ``ABORT`` (1 byte)

This PoC schedules at raw-send granularity and does not parse the SQL wire
protocol to classify statements — that remains the separately-deferred
wire-parsing roadmap item.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
from collections.abc import Callable
from typing import Any

HELLO = 0x48  # 'H'
REQ_SEND = 0x53  # 'S'
GRANT = 0x01
ABORT = 0x00

Launch = Callable[[str], list[Any]]
"""Given the coordinator socket path, spawn workers and return join handles."""


class RawSocketScheduler:
    """Drive an explicit send-ordering schedule over unmodified worker processes."""

    def __init__(self, *, num_workers: int, socket_path: str | None = None, timeout: float = 10.0) -> None:
        self.num_workers = num_workers
        self.timeout = timeout
        self._own_dir: str | None = None
        if socket_path is None:
            self._own_dir = tempfile.mkdtemp(prefix="frontrun-xsched-")
            socket_path = os.path.join(self._own_dir, "s")
        self.socket_path = socket_path

    def run(self, *, launch: Launch, schedule: list[int]) -> list[int]:
        """Grant sends in the order given by *schedule* and return the order granted.

        Each entry is the worker id whose next ``send()`` should proceed. The
        coordinator blocks until that worker has reached its send (so other
        workers stay parked inside their send hooks), enforcing the global order.
        """
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        listener.listen(self.num_workers)
        listener.settimeout(self.timeout)
        handles: list[Any] = []
        conns: dict[int, socket.socket] = {}
        try:
            handles = launch(self.socket_path)
            for _ in range(self.num_workers):
                sock, _addr = listener.accept()
                sock.settimeout(self.timeout)
                hello = _recv_exact(sock, 2)
                if hello is None or hello[0] != HELLO:
                    raise RuntimeError(f"expected HELLO, got {hello!r}")
                conns[hello[1]] = sock

            granted: list[int] = []
            for wid in schedule:
                sock = conns[wid]
                req = _recv_exact(sock, 2)
                if req is None or req[0] != REQ_SEND or req[1] != wid:
                    raise RuntimeError(f"worker {wid}: expected REQ_SEND, got {req!r}")
                sock.sendall(bytes([GRANT]))
                granted.append(wid)
            return granted
        finally:
            for sock in conns.values():
                try:
                    sock.close()
                except OSError:
                    pass
            for handle in handles:
                _join_handle(handle, self.timeout)
            listener.close()
            self._cleanup()

    def _cleanup(self) -> None:
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        if self._own_dir is not None:
            shutil.rmtree(self._own_dir, ignore_errors=True)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except OSError:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _join_handle(handle: Any, timeout: float) -> None:
    wait = getattr(handle, "wait", None)
    if wait is not None:  # subprocess.Popen
        try:
            wait(timeout=timeout)
        except Exception:  # noqa: BLE001 - best-effort teardown
            handle.kill()
    else:  # threading.Thread
        handle.join(timeout)
