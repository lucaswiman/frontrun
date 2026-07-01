"""Phase 4 (proof-of-concept): scheduling unmodified/non-Python workers.

Spawns worker processes that import only the standard library — no frontrun —
under the isolated ``frontrun_xproc_sched`` LD_PRELOAD shim. The shim blocks each
socket ``send()`` until the coordinator grants it, so the coordinator dictates
the exact order in which the unmodified workers' sends fire, at C level, with
zero cooperation from the workers.

The coordinator's returned grant order is the ground-truth global send order
(each grant precedes exactly one real send); that the handshake completes at all
proves the workers were genuinely blocked in their send hooks. Per-connection
byte order is also checked. Cross-connection network *arrival* order is not
asserted (a real caveat of scheduling below the application without acks).

Marked e2e; skipped unless the shim is built (``make build-xproc-sched``).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading

import pytest

import frontrun
from frontrun._dpor_runtime.xproc.raw_sched import RawSocketScheduler

pytestmark = pytest.mark.e2e

_SHIM = os.path.join(os.path.dirname(frontrun.__file__), "libfrontrun_xproc_sched.so")
_WORKER = os.path.join(
    os.path.dirname(os.path.dirname(frontrun.__file__)),
    "crates",
    "xproc_sched",
    "raw_socket_worker.py",
)

requires_shim = pytest.mark.skipif(
    not (os.path.exists(_SHIM) and os.path.exists(_WORKER)),
    reason="Phase 4 shim not built (run: make build-xproc-sched)",
)


class _CollectorServer:
    """Accept N connections and record the bytes received on each."""

    def __init__(self, num_conns: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(num_conns)
        self.port = self._sock.getsockname()[1]
        self._num = num_conns
        self.streams: list[bytes] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _serve(self) -> None:
        conns = [self._sock.accept()[0] for _ in range(self._num)]
        buffers = [b""] * self._num
        open_conns = list(range(self._num))
        import selectors

        sel = selectors.DefaultSelector()
        for i, c in enumerate(conns):
            sel.register(c, selectors.EVENT_READ, i)
        while open_conns:
            for key, _ in sel.select(timeout=10.0):
                i = key.data
                chunk = conns[i].recv(4096)
                if not chunk:
                    sel.unregister(conns[i])
                    open_conns.remove(i)
                else:
                    buffers[i] += chunk
        self.streams = buffers

    def join(self, timeout: float = 10.0) -> None:
        self._thread.join(timeout)

    def close(self) -> None:
        self._sock.close()


def _make_launch(port: int, num_workers: int, num_msgs: int):
    def launch(socket_path: str) -> list[subprocess.Popen]:
        procs: list[subprocess.Popen] = []
        for wid in range(num_workers):
            env = dict(os.environ)
            env["LD_PRELOAD"] = _SHIM
            env["FRONTRUN_XPROC_SCHED_PATH"] = socket_path
            env["FRONTRUN_XPROC_WORKER_ID"] = str(wid)
            procs.append(
                subprocess.Popen(
                    [sys.executable, _WORKER, "127.0.0.1", str(port), str(wid), str(num_msgs)],
                    env=env,
                )
            )
        return procs

    return launch


@requires_shim
@pytest.mark.parametrize("schedule", [[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
def test_coordinator_controls_unmodified_worker_send_order(schedule) -> None:
    server = _CollectorServer(num_conns=2)
    server.start()
    try:
        sched = RawSocketScheduler(num_workers=2, timeout=15.0)
        granted = sched.run(launch=_make_launch(server.port, num_workers=2, num_msgs=2), schedule=schedule)
    finally:
        server.join()
        server.close()

    # The coordinator dictated the exact global send order of unmodified workers.
    assert granted == schedule
    # Every message arrived, and each worker's two messages kept per-connection order.
    assert sorted(server.streams) == sorted([b"w0-0w0-1", b"w1-0w1-1"])
