"""Engine-driven cross-process coordinator (Phase 2: real DPOR).

Instead of brute-force enumeration (Phase 1), this drives the Rust DPOR engine
so equivalent interleavings are pruned (partial-order reduction) and row-lock
deadlocks are detected via the wait-for graph. The trick: reuse the in-process
:class:`DporScheduler` verbatim, running one local *relay* thread per remote
worker. Each relay translates its worker's socket frames into the exact
``report_and_wait`` / ``acquire_row_locks`` / pending-io sequence a real worker
thread would produce, so the scheduler and engine cannot tell the "threads" are
proxies for separate processes.

Per execution: ``dpor_exploration_iter`` opens a fresh ``PyExecution``; a fresh
``DporScheduler`` drives it; ``setup`` resets the DB; workers are spawned and
relayed; then the invariant/deadlock are checked and the engine advances to the
next execution until the search tree is exhausted.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
from collections.abc import Callable
from typing import Any

from frontrun._deadlock import DeadlockError, SchedulerAbort, install_wait_for_graph, uninstall_wait_for_graph
from frontrun._dpor_core import dpor_exploration_iter, make_deadline, make_dpor_engine
from frontrun._dpor_runtime._shared import _dpor_tls, _make_object_key
from frontrun._dpor_runtime.scheduler import DporScheduler
from frontrun._opcode_observer import StableObjectIds

from . import protocol as proto
from .coordinator import CrossProcessResult, Launcher, accept_hello


def _setup_relay_tls(scheduler: DporScheduler, worker_id: int) -> list[tuple[int, str, bool]]:
    """Install the minimal per-thread state ``DporScheduler`` reads for this relay.

    Mirrors the subset of ``DporBytecodeRunner._setup_dpor_tls`` the scheduler
    needs — a pending-io buffer shared with ``scheduler._pending_io_by_thread``
    and lock-depth bookkeeping — without any of the worker-side SQL/opcode
    patching (that lives in the worker process).
    """
    pending_io: list[tuple[int, str, bool]] = []
    _dpor_tls.scheduler = scheduler
    _dpor_tls.thread_id = worker_id
    _dpor_tls.engine = scheduler.engine
    _dpor_tls.execution = scheduler.execution
    _dpor_tls._last_path_id = None
    _dpor_tls.lock_depth = 0
    _dpor_tls.pending_io = pending_io
    scheduler._lock_depth_by_thread[worker_id] = 0
    scheduler._pending_io_by_thread[worker_id] = pending_io
    return pending_io


def _flush_orphan_pending_io(
    scheduler: DporScheduler,
    worker_id: int,
    pending_io: list[tuple[int, str, bool]],
) -> None:
    """Report accesses buffered after the worker's last scheduling point.

    A worker's final access (e.g. the write of a read-modify-write) is buffered
    with no subsequent ``report_and_wait`` to flush it, so it would be dropped —
    mirroring ``DporBytecodeRunner._teardown_dpor_tls``, flush it to the engine
    on relay teardown so the DPOR search sees every access.
    """
    if not pending_io:
        return
    for obj_key, io_kind, synced in pending_io:
        with scheduler._engine_lock:
            if synced:
                scheduler.engine.report_synced_io_access(scheduler.execution, worker_id, obj_key, io_kind)
            else:
                scheduler.engine.report_io_access(scheduler.execution, worker_id, obj_key, io_kind)
    pending_io.clear()


def _teardown_relay_tls(scheduler: DporScheduler, worker_id: int) -> None:
    _dpor_tls.scheduler = None
    _dpor_tls.thread_id = None
    _dpor_tls.pending_io = []
    scheduler._lock_depth_by_thread.pop(worker_id, None)
    scheduler._pending_io_by_thread.pop(worker_id, None)


def _relay_loop(
    scheduler: DporScheduler,
    worker_id: int,
    sock: socket.socket,
    accesses: list[tuple[int, str, str]],
    accesses_lock: threading.Lock,
    worker_errors: dict[int, str],
) -> None:
    """Translate one worker's socket frames into scheduler calls."""
    pending_io = _setup_relay_tls(scheduler, worker_id)
    registered_groups: set[int] = set()
    try:
        while True:
            try:
                msg = proto.recv_msg(sock)
            except (TimeoutError, OSError):
                msg = None
            if msg is None:
                break
            kind = msg["t"]
            if kind == proto.ACCESS:
                rid = msg["rid"]
                access_kind = msg["kind"]
                obj_key = _make_object_key(hash(rid), rid)
                pending_io.append((obj_key, access_kind, True))  # synced=True: Python-level SQL/Redis
                with accesses_lock:
                    accesses.append((worker_id, rid, access_kind))
                if rid.startswith("sql:"):
                    parts = rid.split(":")
                    table_group = f"sql:{parts[1]}" if len(parts) >= 2 else rid
                    group_key = hash(table_group) & 0xFFFFFFFFFFFFFFFF
                    if obj_key not in registered_groups:
                        with scheduler._engine_lock:
                            scheduler.engine.register_resource_group(obj_key, group_key)
                        registered_groups.add(obj_key)
            elif kind == proto.REPORT_AND_WAIT:
                granted = scheduler.report_and_wait(None, worker_id)
                _reply(sock, granted)
                if not granted:
                    break
            elif kind == proto.ACQUIRE_LOCKS:
                try:
                    scheduler.acquire_row_locks(worker_id, list(msg["res"]))
                except SchedulerAbort:
                    _reply(sock, False)
                    break
                else:
                    _reply(sock, True)
            elif kind == proto.RELEASE_LOCKS:
                scheduler.release_row_locks(worker_id)
            elif kind == proto.BEFORE_IO:
                scheduler.before_io(worker_id, msg["rid"])
                granted = not (scheduler._finished or scheduler._error)
                _reply(sock, granted)
                if not granted:
                    break
            elif kind == proto.AFTER_IO:
                scheduler.after_io(worker_id, msg["rid"])
            elif kind == proto.DONE:
                break
            elif kind == proto.ERROR:
                worker_errors[worker_id] = str(msg.get("msg", "worker error"))
                scheduler.report_error(RuntimeError(worker_errors[worker_id]))
                break
    finally:
        _flush_orphan_pending_io(scheduler, worker_id, pending_io)
        _teardown_relay_tls(scheduler, worker_id)
        scheduler.mark_done(worker_id)


def _reply(sock: socket.socket, granted: bool) -> None:
    try:
        proto.send_msg(sock, {"t": proto.GRANT if granted else proto.ABORT})
    except OSError:
        pass


class DporCrossProcessCoordinator:
    def __init__(
        self,
        *,
        num_workers: int,
        socket_path: str | None = None,
        deadlock_timeout: float = 5.0,
        preemption_bound: int | None = 2,
        max_executions: int | None = None,
        max_branches: int = 100_000,
        total_timeout: float | None = None,
        stop_on_first: bool = True,
        search: str | None = None,
        reuse_workers: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.deadlock_timeout = deadlock_timeout
        self.preemption_bound = preemption_bound
        self.max_executions = max_executions
        self.max_branches = max_branches
        self.total_timeout = total_timeout
        self.stop_on_first = stop_on_first
        self.search = search
        self.reuse_workers = reuse_workers
        self._own_dir: str | None = None
        if socket_path is None:
            self._own_dir = tempfile.mkdtemp(prefix="frontrun-xproc-")
            socket_path = os.path.join(self._own_dir, "s")
        self.socket_path = socket_path

    def explore(
        self,
        *,
        launch: Launcher,
        setup: Callable[[], Any],
        invariant: Callable[[], bool],
    ) -> CrossProcessResult:
        engine = make_dpor_engine(
            num_threads=self.num_workers,
            preemption_bound=self.preemption_bound,
            max_branches=self.max_branches,
            max_executions=self.max_executions,
            search=self.search,
        )
        engine_lock = threading.Lock()
        stable_ids = StableObjectIds()
        deadline = make_deadline(self.total_timeout)

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        listener.listen(self.num_workers)
        listener.settimeout(self.deadlock_timeout * 2 + 10.0)
        install_wait_for_graph()

        # Reuse mode: spawn persistent workers and accept their connections once.
        persistent_handles: Any = None
        persistent_socks: dict[int, socket.socket] = {}
        if self.reuse_workers:
            persistent_handles = launch.launch(self.socket_path, list(range(self.num_workers)))
            for _ in range(self.num_workers):
                sock, wid = accept_hello(listener, self.deadlock_timeout)
                persistent_socks[wid] = sock

        num_explored = 0
        try:
            for step in dpor_exploration_iter(
                engine=engine,
                engine_lock=engine_lock,
                stable_ids=stable_ids,
                total_deadline=deadline,
            ):
                execution = step.execution
                scheduler = DporScheduler(
                    engine,
                    execution,
                    self.num_workers,
                    engine_lock=engine_lock,
                    deadlock_timeout=self.deadlock_timeout,
                    detect_io=True,
                    stable_ids=stable_ids,
                )
                accesses: list[tuple[int, str, str]] = []
                worker_errors: dict[int, str] = {}
                setup()  # reset external state before each interleaving
                if self.reuse_workers:
                    self._run_reused(persistent_socks, scheduler, accesses, worker_errors)
                else:
                    self._run_spawned(listener, launch, scheduler, accesses, worker_errors)
                num_explored += 1

                result = self._evaluate(
                    execution, scheduler, engine_lock, invariant, worker_errors, accesses, num_explored
                )
                if result is not None:
                    return result
            return CrossProcessResult(ok=True, iterations=num_explored, exhausted=True)
        finally:
            if self.reuse_workers:
                for sock in persistent_socks.values():
                    try:
                        proto.send_msg(sock, {"t": proto.SHUTDOWN})
                    except OSError:
                        pass
                    try:
                        sock.close()
                    except OSError:
                        pass
                if persistent_handles is not None:
                    launch.join(persistent_handles, self.deadlock_timeout)
            uninstall_wait_for_graph()
            listener.close()
            self._cleanup_socket()

    def _drive_relays(
        self,
        scheduler: DporScheduler,
        socks_by_id: dict[int, socket.socket],
        accesses: list[tuple[int, str, str]],
        worker_errors: dict[int, str],
    ) -> None:
        accesses_lock = threading.Lock()
        relays = [
            threading.Thread(
                target=_relay_loop,
                args=(scheduler, wid, sock, accesses, accesses_lock, worker_errors),
                name=f"xproc-relay-{wid}",
                daemon=True,
            )
            for wid, sock in socks_by_id.items()
        ]
        for t in relays:
            t.start()
        join_budget = self.deadlock_timeout * 2 + 10.0
        for t in relays:
            t.join(join_budget)

    def _run_spawned(
        self,
        listener: socket.socket,
        launch: Launcher,
        scheduler: DporScheduler,
        accesses: list[tuple[int, str, str]],
        worker_errors: dict[int, str],
    ) -> None:
        handles = launch.launch(self.socket_path, list(range(self.num_workers)))
        socks_by_id: dict[int, socket.socket] = {}
        try:
            for _ in range(self.num_workers):
                sock, wid = accept_hello(listener, self.deadlock_timeout)
                socks_by_id[wid] = sock
            self._drive_relays(scheduler, socks_by_id, accesses, worker_errors)
        finally:
            for s in socks_by_id.values():
                try:
                    s.close()
                except OSError:
                    pass
            launch.join(handles, self.deadlock_timeout)

    def _run_reused(
        self,
        socks_by_id: dict[int, socket.socket],
        scheduler: DporScheduler,
        accesses: list[tuple[int, str, str]],
        worker_errors: dict[int, str],
    ) -> None:
        for sock in socks_by_id.values():
            proto.send_msg(sock, {"t": proto.ITER_START})
        self._drive_relays(scheduler, socks_by_id, accesses, worker_errors)

    def _evaluate(
        self,
        execution: Any,
        scheduler: DporScheduler,
        engine_lock: threading.Lock,
        invariant: Callable[[], bool],
        worker_errors: dict[int, str],
        accesses: list[tuple[int, str, str]],
        num_explored: int,
    ) -> CrossProcessResult | None:
        """Return a failing result to stop, or None to continue exploring."""
        with engine_lock:
            schedule_trace = list(execution.schedule_trace)
        err = scheduler._error

        if worker_errors:
            wid = min(worker_errors)
            return CrossProcessResult(
                ok=False,
                iterations=num_explored,
                exhausted=False,
                failing_schedule=schedule_trace,
                failure=f"worker {wid} failed: {worker_errors[wid]}",
                failure_kind="worker_error",
                accesses=accesses,
            )
        if isinstance(err, DeadlockError):
            return CrossProcessResult(
                ok=False,
                iterations=num_explored,
                exhausted=False,
                failing_schedule=schedule_trace,
                failure=getattr(err, "cycle_description", None) or str(err),
                failure_kind="deadlock",
                accesses=accesses,
            )
        # A scheduler fallback TimeoutError means the run free-ran unscheduled;
        # its final state describes no DPOR schedule, so skip the invariant.
        if isinstance(err, TimeoutError):
            return None
        if not invariant():
            if self.stop_on_first:
                return CrossProcessResult(
                    ok=False,
                    iterations=num_explored,
                    exhausted=False,
                    failing_schedule=schedule_trace,
                    failure="invariant violated",
                    failure_kind="invariant",
                    accesses=accesses,
                )
        return None

    def _cleanup_socket(self) -> None:
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        if self._own_dir is not None:
            shutil.rmtree(self._own_dir, ignore_errors=True)
