"""Robustness tests for cross-process exploration failure modes.

Each test pins one way a worker/coordinator interaction can go wrong outside
the happy path: malformed frames, corrupt length prefixes, invalid or
duplicate worker ids, concurrent proxy use from a threaded worker target, and
resource cleanup when exploration fails before any worker is launched.  The
common bar: every failure must surface as a *structured* result (or a clear,
typed exception) — never an uncaught ``KeyError``/``ValueError`` escaping
``explore()``, a silent false ``ok=True``, or a leaked temp directory.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time

import pytest

import frontrun
from frontrun._dpor_runtime.xproc import protocol as proto
from frontrun._dpor_runtime.xproc.coordinator import CrossProcessCoordinator, accept_hello
from frontrun._dpor_runtime.xproc.proxy import SchedulerProxy
from frontrun._dpor_runtime.xproc.worker import ThreadLauncher, _connect_and_serve


def test_total_timeout_bounds_spawned_worker_cleanup() -> None:
    """A total search deadline must not turn into a per-process join delay."""
    started = time.monotonic()
    result = frontrun.explore_processes(
        frontrun.Subprocess("time:sleep", (60,)),
        setup=lambda: None,
        invariant=lambda _state: True,
        total_timeout=0.2,
        deadlock_timeout=1.0,
    )
    elapsed = time.monotonic() - started

    assert result.ok
    assert not result.exhausted
    assert elapsed < 1.0, f"total_timeout=0.2 took {elapsed:.3f}s while cleaning up the worker"

# ---------------------------------------------------------------------------
# Malformed frames must become structured worker errors, not uncaught KeyErrors
# ---------------------------------------------------------------------------


def test_malformed_frame_missing_type_is_worker_error_not_keyerror() -> None:
    # A post-HELLO frame with no "t" key previously raised a raw KeyError from
    # _advance that escaped explore() (the DPOR relay already structures the
    # same failure). It must be a worker_error result instead.
    def bad(proxy) -> None:
        proto.send_msg(proxy._sock, {"w": 0})

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=3.0)
    result = coord.explore(
        worker_set=ThreadLauncher([bad]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=10,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "malformed" in (result.failure or "")


def test_acquire_locks_frame_without_resources_is_worker_error() -> None:
    # ACQUIRE_LOCKS with no "res" list previously crashed _grantable with a
    # KeyError once the coordinator tried to arbitrate the lock request.
    def bad(proxy) -> None:
        proto.send_msg(proxy._sock, {"t": proto.ACQUIRE_LOCKS, "w": 0})
        # The coordinator rejects the frame and stops reading; the worker's
        # DONE below is simply never consumed.

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=3.0)
    result = coord.explore(
        worker_set=ThreadLauncher([bad]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=10,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "malformed" in (result.failure or "")


def test_access_frame_without_resource_id_is_worker_error() -> None:
    def bad(proxy) -> None:
        proto.send_msg(proxy._sock, {"t": proto.ACCESS, "w": 0, "kind": "read"})

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=3.0)
    result = coord.explore(
        worker_set=ThreadLauncher([bad]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=10,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "malformed" in (result.failure or "")


# ---------------------------------------------------------------------------
# Wire framing: a corrupt length prefix must fail loudly, not allocate 4 GiB
# ---------------------------------------------------------------------------


def test_recv_msg_rejects_oversized_length_prefix() -> None:
    # Legitimate frames are tiny (the largest is an ITER_START dill payload).
    # A desynced/corrupt stream whose next four bytes decode to ~4 GiB must
    # raise a clear OSError — which every coordinator path already routes to a
    # structured failure — instead of buffering an absurd read.
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack(">I", 0xFFFFFFFF))
        a.close()
        with pytest.raises(OSError, match="frame length"):
            proto.recv_msg(b)
    finally:
        b.close()


def test_recv_msg_accepts_normal_frames_unchanged() -> None:
    a, b = socket.socketpair()
    try:
        proto.send_msg(a, {"t": proto.GRANT})
        assert proto.recv_msg(b) == {"t": proto.GRANT}
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# HELLO validation: non-integer, duplicate, and out-of-range worker ids
# ---------------------------------------------------------------------------


def _tcp_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    return listener


def test_accept_hello_rejects_non_integer_worker_id() -> None:
    # {"t": "hello", "w": "abc"} passes the "w" presence check; int() then
    # raised a bare ValueError that escaped explore() and leaked the accepted
    # socket. It must be an OSError (the coordinators' connection-failure
    # type), and the socket must be closed.
    listener = _tcp_listener()
    port = listener.getsockname()[1]
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        proto.send_msg(client, {"t": proto.HELLO, "w": "abc"})
        with pytest.raises(OSError):
            accept_hello(listener, timeout=2.0)
        client.close()
    finally:
        listener.close()


class _ForcedIdLauncher:
    """Launch in-process workers that announce *forced* ids, ignoring targets."""

    def __init__(self, bodies, forced_ids) -> None:
        self._bodies = list(bodies)
        self._forced_ids = list(forced_ids)

    def launch(self, targets):
        threads = []
        for target, wid, body in zip(list(targets), self._forced_ids, self._bodies):
            t = threading.Thread(
                target=_connect_and_serve,
                args=(str(target.args[0]), wid, body),
                daemon=True,
            )
            t.start()
            threads.append(t)
        return threads

    def join(self, handles, timeout):
        for t in handles:
            t.join(timeout)
        return [t for t in handles if t.is_alive()]


def test_duplicate_worker_ids_are_a_structured_failure_not_a_false_pass() -> None:
    # Two workers announcing the same id previously overwrote each other in the
    # coordinator's connection map: one worker was silently never driven, its
    # socket leaked, and the run could complete ok=True over half the workload.
    def body(proxy) -> None:
        proxy.report_and_wait(None, 0)

    coord = CrossProcessCoordinator(num_workers=2, deadlock_timeout=2.0)
    result = coord.explore(
        worker_set=_ForcedIdLauncher([body, body], forced_ids=[0, 0]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=10,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "duplicate worker id" in (result.failure or "")


def test_out_of_range_worker_id_is_a_structured_failure() -> None:
    # A worker announcing an id outside range(num_workers) previously flowed
    # straight into scheduling bookkeeping (and, on the DPOR path, into the
    # Rust engine) under a nonsense id.
    def body(proxy) -> None:
        proxy.report_and_wait(None, 0)

    coord = CrossProcessCoordinator(num_workers=1, deadlock_timeout=2.0)
    result = coord.explore(
        worker_set=_ForcedIdLauncher([body], forced_ids=[5]),
        setup=lambda: None,
        invariant=lambda: True,
        max_iterations=10,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "worker id" in (result.failure or "")


def test_dpor_duplicate_worker_ids_are_a_structured_failure() -> None:
    from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator

    def body(proxy) -> None:
        proxy.report_and_wait(None, 0)

    coord = DporCrossProcessCoordinator(num_workers=2, deadlock_timeout=2.0)
    result = coord.explore(
        worker_set=_ForcedIdLauncher([body, body], forced_ids=[1, 1]),
        setup=lambda: None,
        invariant=lambda: True,
    )
    assert not result.ok
    assert result.failure_kind == "worker_error"
    assert "duplicate worker id" in (result.failure or "")


# ---------------------------------------------------------------------------
# Concurrent proxy use from a threaded worker target must fail loudly
# ---------------------------------------------------------------------------


def test_proxy_rejects_concurrent_use_from_second_thread() -> None:
    # The proxy models ONE logical worker: its frames must form a single
    # sequential stream, and GRANT frames carry no addressee. Two threads
    # blocking in recv on the same socket would be woken arbitrarily —
    # silent nondeterminism. Concurrent use must raise instead.
    worker_sock, coord_sock = socket.socketpair()
    got_first_frame = threading.Event()
    release_grant = threading.Event()
    results: list[object] = []

    def coordinator() -> None:
        msg = proto.recv_msg(coord_sock)
        assert msg is not None and msg["t"] == proto.REPORT_AND_WAIT
        got_first_frame.set()
        release_grant.wait(timeout=5.0)
        proto.send_msg(coord_sock, {"t": proto.GRANT})

    coord_thread = threading.Thread(target=coordinator, daemon=True)
    coord_thread.start()
    try:
        proxy = SchedulerProxy(worker_sock, worker_id=0)

        def first() -> None:
            results.append(proxy.report_and_wait(None, 0))

        t1 = threading.Thread(target=first, daemon=True)
        t1.start()
        assert got_first_frame.wait(timeout=5.0)
        # t1 is now blocked awaiting the grant; a second thread's scheduling
        # call must be rejected, not interleaved onto the same socket.
        with pytest.raises(RuntimeError, match="concurrent"):
            proxy.report_and_wait(None, 0)
        with pytest.raises(RuntimeError, match="concurrent"):
            proxy.io_report("sql:t:k", "read")
        assert "concurrent" in proxy.fatal_error
        release_grant.set()
        t1.join(timeout=5.0)
        assert results == [True]
    finally:
        release_grant.set()
        coord_thread.join(timeout=5.0)
        worker_sock.close()
        coord_sock.close()


def test_proxy_sequential_cross_thread_use_still_works() -> None:
    # Only *concurrent* use is unsound; sequential handoff between threads
    # (thread does one statement, joins, main continues) stays supported.
    worker_sock, coord_sock = socket.socketpair()

    def coordinator() -> None:
        for _ in range(2):
            msg = proto.recv_msg(coord_sock)
            assert msg is not None and msg["t"] == proto.REPORT_AND_WAIT
            proto.send_msg(coord_sock, {"t": proto.GRANT})

    coord_thread = threading.Thread(target=coordinator, daemon=True)
    coord_thread.start()
    try:
        proxy = SchedulerProxy(worker_sock, worker_id=0)
        results: list[bool] = []
        t = threading.Thread(target=lambda: results.append(proxy.report_and_wait(None, 0)), daemon=True)
        t.start()
        t.join(timeout=5.0)
        results.append(proxy.report_and_wait(None, 0))
        assert results == [True, True]
    finally:
        coord_thread.join(timeout=5.0)
        worker_sock.close()
        coord_sock.close()


# ---------------------------------------------------------------------------
# Failure before launch must not leak the coordinator's temp directory
# ---------------------------------------------------------------------------


def test_dpor_engine_construction_failure_cleans_up_socket_tempdir() -> None:
    # An invalid search= reaches the Rust engine constructor, which raises
    # before the coordinator's try/finally is armed; the mkdtemp'd socket
    # directory previously leaked on every such call.
    from frontrun._dpor_runtime.xproc.dpor_coordinator import DporCrossProcessCoordinator

    coord = DporCrossProcessCoordinator(num_workers=1, search="bogus")
    tempdir = os.path.dirname(coord.socket_path)
    assert os.path.isdir(tempdir)

    class _NeverLaunched:
        def launch(self, targets):  # pragma: no cover - engine raises first
            raise AssertionError("worker_set must not be touched")

        def join(self, handles, timeout):  # pragma: no cover
            return []

    with pytest.raises(ValueError, match="search"):
        coord.explore(worker_set=_NeverLaunched(), setup=lambda: None, invariant=lambda: True)
    assert not os.path.exists(tempdir)
