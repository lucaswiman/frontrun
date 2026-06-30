"""Worker-side cross-process exploration: wire protocol + SchedulerProxy.

Phase 1 of ideas/cross_process_exploration.md. These tests exercise the
worker half over an in-process ``socket.socketpair()`` loopback — a fake
coordinator on one end, the real ``SchedulerProxy`` on the other — so the
message exchange and blocking semantics are pinned without spawning
subprocesses or touching a database.
"""

from __future__ import annotations

import socket
import threading

import pytest

from frontrun._deadlock import SchedulerAbort
from frontrun._dpor_runtime.xproc import protocol as proto
from frontrun._dpor_runtime.xproc.proxy import SchedulerProxy


def _pair() -> tuple[socket.socket, socket.socket]:
    a, b = socket.socketpair()
    return a, b


def test_protocol_roundtrip() -> None:
    a, b = _pair()
    try:
        proto.send_msg(a, {"t": proto.ACCESS, "w": 3, "rid": "sql:users", "kind": "write"})
        assert proto.recv_msg(b) == {"t": proto.ACCESS, "w": 3, "rid": "sql:users", "kind": "write"}
    finally:
        a.close()
        b.close()


def test_recv_returns_none_on_closed_peer() -> None:
    a, b = _pair()
    a.close()
    try:
        assert proto.recv_msg(b) is None
    finally:
        b.close()


def test_multiple_frames_are_not_coalesced() -> None:
    a, b = _pair()
    try:
        proto.send_msg(a, {"t": proto.DONE, "w": 0})
        proto.send_msg(a, {"t": proto.DONE, "w": 1})
        assert proto.recv_msg(b) == {"t": proto.DONE, "w": 0}
        assert proto.recv_msg(b) == {"t": proto.DONE, "w": 1}
    finally:
        a.close()
        b.close()


def test_io_report_sends_access_frame() -> None:
    worker, coord = _pair()
    try:
        proxy = SchedulerProxy(worker, worker_id=2)
        proxy.io_report("sql:orders:id=7", "write")
        msg = proto.recv_msg(coord)
        assert msg == {"t": proto.ACCESS, "w": 2, "rid": "sql:orders:id=7", "kind": "write"}
    finally:
        worker.close()
        coord.close()


def test_report_and_wait_blocks_until_grant() -> None:
    worker, coord = _pair()
    result: list[bool] = []
    proceed = threading.Event()

    def coordinator() -> None:
        msg = proto.recv_msg(coord)
        assert msg == {"t": proto.REPORT_AND_WAIT, "w": 1}
        proceed.wait(timeout=2.0)
        proto.send_msg(coord, {"t": proto.GRANT})

    t = threading.Thread(target=coordinator)
    t.start()
    try:
        proxy = SchedulerProxy(worker, worker_id=1)

        def call() -> None:
            result.append(proxy.report_and_wait(None, 1))

        caller = threading.Thread(target=call)
        caller.start()
        # The caller must still be blocked while the coordinator withholds the grant.
        caller.join(timeout=0.3)
        assert caller.is_alive()
        proceed.set()
        caller.join(timeout=2.0)
        assert result == [True]
    finally:
        t.join(timeout=2.0)
        worker.close()
        coord.close()


def test_report_and_wait_returns_false_on_abort() -> None:
    worker, coord = _pair()

    def coordinator() -> None:
        assert proto.recv_msg(coord)["t"] == proto.REPORT_AND_WAIT
        proto.send_msg(coord, {"t": proto.ABORT})

    t = threading.Thread(target=coordinator)
    t.start()
    try:
        proxy = SchedulerProxy(worker, worker_id=1)
        assert proxy.report_and_wait(None, 1) is False
        # Once aborted, further scheduling points short-circuit without I/O.
        assert proxy.report_and_wait(None, 1) is False
    finally:
        t.join(timeout=2.0)
        worker.close()
        coord.close()


def test_acquire_row_locks_sends_resources_and_waits() -> None:
    worker, coord = _pair()

    def coordinator() -> None:
        msg = proto.recv_msg(coord)
        assert msg == {"t": proto.ACQUIRE_LOCKS, "w": 4, "res": ["sql:users:id=1", "sql:posts:id=2"]}
        proto.send_msg(coord, {"t": proto.GRANT})

    t = threading.Thread(target=coordinator)
    t.start()
    try:
        proxy = SchedulerProxy(worker, worker_id=4)
        proxy.acquire_row_locks(4, ["sql:users:id=1", "sql:posts:id=2"])  # returns without raising
    finally:
        t.join(timeout=2.0)
        worker.close()
        coord.close()


def test_acquire_row_locks_raises_on_abort() -> None:
    worker, coord = _pair()

    def coordinator() -> None:
        assert proto.recv_msg(coord)["t"] == proto.ACQUIRE_LOCKS
        proto.send_msg(coord, {"t": proto.ABORT})

    t = threading.Thread(target=coordinator)
    t.start()
    try:
        proxy = SchedulerProxy(worker, worker_id=4)
        with pytest.raises(SchedulerAbort):
            proxy.acquire_row_locks(4, ["sql:users:id=1"])
    finally:
        t.join(timeout=2.0)
        worker.close()
        coord.close()


def test_release_row_locks_is_fire_and_forget() -> None:
    worker, coord = _pair()
    try:
        proxy = SchedulerProxy(worker, worker_id=5)
        proxy.release_row_locks(5)  # must not block waiting for a reply
        assert proto.recv_msg(coord) == {"t": proto.RELEASE_LOCKS, "w": 5}
    finally:
        worker.close()
        coord.close()
