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
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from frontrun._deadlock import DeadlockError, SchedulerAbort, install_wait_for_graph, uninstall_wait_for_graph
from frontrun._dpor_core import (
    IterationCustomizer,
    LivenessProbe,
    WorkerSet,
    dpor_exploration_iter,
    make_deadline,
    make_dpor_engine,
)
from frontrun._dpor_runtime._shared import _dpor_tls, _make_object_key
from frontrun._dpor_runtime.scheduler import DporScheduler
from frontrun._opcode_observer import StableObjectIds

from . import protocol as proto
from .coordinator import CrossProcessResult, accept_hello_live, worker_targets
from .launch import WorkerSerializationError


class _RelayDporScheduler(DporScheduler):
    """DporScheduler variant for socket relays: no cross-thread pending-io flush.

    In-process, a thread's buffered ``pending_io`` describes accesses that
    already physically happened (deferred I/O), so another thread reaching a
    real I/O boundary must make them visible to the engine — that is what
    ``_flush_other_pending_io_for_current_io_unlocked`` does. An xproc relay's
    buffer instead holds ACCESS frames the worker sends BEFORE executing the
    statement (pre-declarations). Cross-flushing those from another relay's
    turn attributes the not-yet-executed access at that relay's step whenever
    the ACCESS frame wins the OS race against the other worker's grant; the
    engine trace desyncs, the write-write reversal is never seeded into the
    wakeup tree, and the search ends early with a false ok=True. Relay buffers
    are flushed only at their own worker's grant (the own-pending flush inside
    ``before_sync_retry`` / ``report_and_wait`` / ``before_io``), so the
    cross-thread flush must be a no-op here.
    """

    def _flush_other_pending_io_for_current_io_unlocked(self, thread_id: int) -> None:
        return


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


def _flush_relay_pending_io(
    scheduler: DporScheduler,
    worker_id: int,
    pending_io: list[tuple[int, str, bool]],
) -> None:
    """Report this relay's buffered accesses to the engine, then clear the buffer.

    Relay-teardown flush only: a worker's trailing access (e.g. one reported
    after its last statement, with no subsequent scheduling point to flush it)
    would otherwise be dropped and the DPOR search would never see it. Every
    in-run flush happens inside the scheduler's own-pending flush at the
    worker's grant (``before_sync_retry`` / ``before_io``), which both
    attributes the access at the worker's own scheduling step and — on the
    ACQUIRE_LOCKS path — records it before ``acquire_row_locks`` reports
    ``lock_acquire``, i.e. OUTSIDE the critical section it precedes (otherwise
    two workers' unlocked writes would look lock-synchronized and DPOR would
    prune the real race).

    Unlike ``DporScheduler._flush_pending_io_for_unlocked`` (whose ``_unlocked``
    contract requires the caller to already hold ``self._condition``), this runs
    from the relay thread without the condition lock — safe because it only
    touches this worker's own buffer and serialises the engine calls on
    ``_engine_lock``.
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
    unclean: set[int],
    progress: list[float] | None = None,
    recv_timeout: float | None = None,
) -> None:
    """Translate one worker's socket frames into scheduler calls.

    Records the worker id in *unclean* unless the worker reached a clean
    terminal (DONE/ERROR). An abort mid-handshake or a recv timeout leaves the
    worker still running and its socket at an unknown frame boundary; in reuse
    mode the coordinator uses *unclean* to avoid reusing a poisoned connection.

    *progress* is a shared single-element heartbeat: the relay bumps it
    whenever a frame arrives or a grant is issued, and ``_drive_relays`` treats
    relays as hung only when NO relay has made progress for the whole join
    budget (deadlock_timeout bounds silence between steps, not iteration wall
    time). *recv_timeout* (normally ``deadlock_timeout``) bounds each recv:
    when it fires the worker is alive but silent — a statement slower than
    deadlock_timeout or blocking outside frontrun's model — which is reported
    through the scheduler's error slot so ``_evaluate`` diagnoses
    ``failure_kind="timeout"`` with the raise-deadlock_timeout advice, not a
    misleading worker disconnect.
    """
    if recv_timeout is not None and recv_timeout > 0:
        try:
            sock.settimeout(recv_timeout)
        except OSError:
            pass

    def _bump_progress() -> None:
        if progress is not None:
            progress[0] = time.monotonic()

    pending_io = _setup_relay_tls(scheduler, worker_id)
    registered_groups: set[int] = set()
    clean = False
    # True while this worker owns the scheduler turn granted by
    # before_sync_retry: the worker is executing its real driver call, and the
    # arrival of its next frame (or its disconnect) marks that call complete.
    holding_sync_turn = False
    try:
        while True:
            try:
                msg = proto.recv_msg(sock)
            except TimeoutError:
                # Socket still open but silent past deadlock_timeout: the
                # worker is alive but its statement overran (or blocked at the
                # DB level, outside frontrun's model). Latch a TimeoutError on
                # the scheduler BEFORE releasing the held turn (the finally
                # does that) so no other relay gets granted and _evaluate's
                # failure_kind="timeout" diagnosis wins over any disconnect /
                # relay-join error.
                report_error = getattr(scheduler, "report_error", None)
                if callable(report_error):
                    report_error(
                        TimeoutError(f"worker {worker_id} sent no frame for {recv_timeout}s while still connected")
                    )
                break
            except OSError:
                msg = None
            if msg is not None:
                _bump_progress()
                if msg["t"] == proto.ACCESS:
                    # ACCESS frames declare the next operation. Keep the previous
                    # sync turn until the following non-ACCESS boundary so every
                    # access from one SQL/Redis operation is recorded contiguously.
                    # The pending buffer is flushed when that boundary is granted.
                    rid = msg["rid"]
                    access_kind = msg["kind"]
                    obj_key = _make_object_key(hash(rid), rid)
                    pending_io.append((obj_key, access_kind, True))  # synced=True: Python-level SQL/Redis
                    with accesses_lock:
                        accesses.append((worker_id, rid, access_kind))
                    # Register the resource's table group with the engine once
                    # per obj_key (Defect #15). Gate the parse/hash on the
                    # membership check so a recurring access doesn't re-split
                    # and re-hash.
                    if rid.startswith("sql:") and obj_key not in registered_groups:
                        parts = rid.split(":")
                        table_group = f"sql:{parts[1]}" if len(parts) >= 2 else rid
                        group_key = hash(table_group) & 0xFFFFFFFFFFFFFFFF
                        with scheduler._engine_lock:
                            scheduler.engine.register_resource_group(obj_key, group_key)
                        registered_groups.add(obj_key)
                    continue
            if holding_sync_turn:
                # The statement the last grant covered has finished executing
                # (the worker sent its next frame); release the held turn so
                # the precomputed next worker can run. Holding it through the
                # driver call mirrors the in-process SQL path
                # (_sql_cursor._dpor_schedule_and_suppress_sync) and keeps two
                # workers' real DB calls from overlapping nondeterministically.
                scheduler.after_sync_retry(worker_id)
                holding_sync_turn = False
            if msg is None:
                with accesses_lock:
                    worker_errors[worker_id] = "worker disconnected"
                break
            kind = msg["t"]
            if kind == proto.REPORT_AND_WAIT:
                # Take the turn (flushing this worker's buffered pre-declared
                # accesses at its own step, inside before_sync_retry) and HOLD
                # it: the worker executes its real statement after the grant,
                # and the turn is released only when its next frame arrives.
                granted = scheduler.before_sync_retry(worker_id)
                _bump_progress()
                _reply(sock, granted)
                if not granted:
                    break
                holding_sync_turn = True
            elif kind == proto.ACQUIRE_LOCKS:
                # Take and hold the scheduling turn through the modeled row-lock
                # acquire. A plain report_and_wait() schedules the next worker
                # before acquire_row_locks() records lock_acquire, letting two
                # relays race around the acquire and desynchronizing the engine
                # trace from the executor. before_sync_retry also flushes any
                # access buffered before this acquire at the worker's own step,
                # BEFORE acquire_row_locks reports 'lock_acquire' — so it is
                # recorded with the thread's PRE-lock happens-before rather
                # than inside the critical section (which would make two
                # workers' unlocked writes look lock-synchronized and prune
                # the race).
                if not scheduler.before_sync_retry(worker_id):
                    _reply(sock, False)
                    break
                try:
                    scheduler.acquire_row_locks(worker_id, list(msg["res"]))
                except SchedulerAbort:
                    scheduler.after_sync_retry(worker_id)
                    _reply(sock, False)
                    break
                # Keep the turn: the worker performs the real (locked) driver
                # call after this grant; released when its next frame arrives.
                _bump_progress()
                _reply(sock, True)
                holding_sync_turn = True
            elif kind == proto.RELEASE_LOCKS:
                scheduler.release_row_locks(worker_id, msg.get("res"))
            elif kind == proto.BEFORE_IO:
                scheduler.before_io(worker_id, msg["rid"])
                # The turn was granted iff before_io made this worker the active
                # IO thread; read it under the condition lock rather than racing
                # on _finished/_error, which other relays mutate.
                with scheduler._condition:
                    granted = scheduler._active_io_thread == worker_id
                _bump_progress()
                _reply(sock, granted)
                if not granted:
                    break
            elif kind == proto.AFTER_IO:
                scheduler.after_io(worker_id, msg["rid"])
            elif kind == proto.DONE:
                clean = True
                break
            elif kind == proto.ERROR:
                message = str(msg.get("msg", "worker error"))
                with accesses_lock:
                    worker_errors[worker_id] = message
                scheduler.report_error(RuntimeError(message))
                clean = True
                break
    finally:
        if holding_sync_turn:
            scheduler.after_sync_retry(worker_id)
        if not clean:
            with accesses_lock:
                unclean.add(worker_id)
        _flush_relay_pending_io(scheduler, worker_id, pending_io)
        _teardown_relay_tls(scheduler, worker_id)
        scheduler.mark_done(worker_id)


def _reply(sock: socket.socket, granted: bool) -> None:
    try:
        proto.send_msg(sock, {"t": proto.GRANT if granted else proto.ABORT})
    except OSError:
        pass


class _WorkerLaunchError(OSError):
    """A worker failed to connect; its message already carries child diagnostics."""


class _TotalTimeoutExpired(Exception):
    """total_timeout expired while an execution was in flight.

    Deliberately NOT a TimeoutError subclass: the exploration loop's
    ``except (TimeoutError, OSError)`` maps those to a ``worker_error``
    connection-failure result, whereas a total_timeout expiry is a clean
    truncation of the search (``exhausted=False``), not a workload failure.
    """


def _launch_error(worker_set: WorkerSet, handles: Any, exc: Exception) -> Exception:
    """Enrich a connect failure with the WorkerSet's diagnosis of dead children.

    Turns a bare ``TimeoutError`` (worker never sent HELLO) into a message naming
    the real cause — e.g. a child that exited with ``ModuleNotFoundError`` for a
    bad ``module:callable`` target — when the WorkerSet can recover it.
    """
    detail = worker_set.diagnose(handles) if isinstance(worker_set, LivenessProbe) else None
    if not detail:
        return exc
    return _WorkerLaunchError(f"{type(exc).__name__}: {exc}; {detail}")


def _connection_failure(exc: Exception, iterations: int) -> CrossProcessResult:
    """A worker never connected (or its socket broke): report a clean result."""
    detail = str(exc) if isinstance(exc, _WorkerLaunchError) else f"{type(exc).__name__}: {exc}"
    return CrossProcessResult(
        ok=False,
        iterations=iterations,
        exhausted=False,
        failure=f"worker connection failed: {detail}",
        failure_kind="worker_error",
    )


def _serialization_failure(exc: Exception, iterations: int) -> CrossProcessResult:
    """A worker/state couldn't be serialised for a subprocess: report cleanly.

    ``_dumps_worker`` raises :class:`WorkerSerializationError` (payload even
    dill cannot serialise, or dill missing); it surfaces from
    ``worker_set.launch(...)`` / ``iter_start_message(...)`` inside the
    exploration loop. Return the same structured ``worker_error`` result the
    connection-failure path uses instead of letting a bare exception escape,
    while still surfacing the clear message.
    """
    return CrossProcessResult(
        ok=False,
        iterations=iterations,
        exhausted=False,
        failure=f"worker serialization failed: {type(exc).__name__}: {exc}",
        failure_kind="worker_error",
    )


class DporCrossProcessCoordinator:
    """Engine-driven cross-process DPOR coordinator.

    Reuse limitation: with ``reuse_workers=True`` the first iteration that ends
    unclean (a deadlock or an aborted worker) leaves a poisoned persistent
    socket, so the search stops early — reported honestly as
    ``exhausted=False`` — rather than re-spawning workers. Use the default
    respawn mode (``reuse_workers=False``) for exhaustive multi-bug search over
    deadlock-bearing workloads.
    """

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
        # Overall budget for a worker to connect and send HELLO; also the
        # no-progress backstop in _drive_relays. Generous vs deadlock_timeout
        # because process spawn is slow.
        self._connect_budget = deadlock_timeout * 2 + 10.0
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
        self._relay_no_progress_budget = self._connect_budget
        self.socket_path = socket_path

    def explore(
        self,
        *,
        worker_set: WorkerSet,
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
        listener.settimeout(self._connect_budget)
        install_wait_for_graph()

        persistent_handles: Any = None
        persistent_socks: dict[int, socket.socket] = {}
        num_explored = 0
        exhausted = True
        first_failure: CrossProcessResult | None = None
        # Every failing execution as (execution_number, schedule), mirroring
        # thread-mode InterleavingResult.failures — with stop_on_first=False
        # later failing schedules must not be discarded.
        failures: list[tuple[int, list[int]]] = []
        try:
            # Reuse mode: spawn persistent workers and accept their connections once.
            if self.reuse_workers:
                try:
                    persistent_handles = worker_set.launch(
                        worker_targets(self.socket_path, list(range(self.num_workers)))
                    )
                except WorkerSerializationError as exc:
                    return _serialization_failure(exc, 0)
                try:
                    for _ in range(self.num_workers):
                        sock, wid = accept_hello_live(listener, worker_set, persistent_handles, self._connect_budget)
                        persistent_socks[wid] = sock
                except (TimeoutError, OSError) as exc:
                    return _connection_failure(_launch_error(worker_set, persistent_handles, exc), 0)

            for step in dpor_exploration_iter(
                engine=engine,
                engine_lock=engine_lock,
                stable_ids=stable_ids,
                total_deadline=deadline,
            ):
                execution = step.execution
                scheduler = _RelayDporScheduler(
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
                unclean: set[int] = set()
                setup()  # reset external state before each interleaving
                try:
                    if self.reuse_workers:
                        self._run_reused(
                            persistent_socks, worker_set, scheduler, accesses, worker_errors, unclean, deadline
                        )
                    else:
                        self._run_spawned(
                            listener, worker_set, scheduler, accesses, worker_errors, unclean, deadline
                        )
                except _TotalTimeoutExpired:
                    # The in-flight execution was truncated by the user's time
                    # budget: nothing about it was verified, so it neither
                    # counts as explored nor as a failure — but the search can
                    # no longer claim exhaustion.
                    exhausted = False
                    break
                except (TimeoutError, OSError) as exc:
                    return replace(_connection_failure(exc, num_explored + 1), failures=failures)
                except WorkerSerializationError as exc:
                    # A dill serialisation failure in worker_set.launch(...) /
                    # iter_start_message(...) — surface it as a structured
                    # worker_error rather than a bare exception. Anything
                    # broader (a generic TypeError from launch machinery) is a
                    # bug and must propagate rather than be mislabeled.
                    return replace(_serialization_failure(exc, num_explored + 1), failures=failures)
                num_explored += 1

                result = self._evaluate(
                    execution, scheduler, engine_lock, invariant, worker_errors, accesses, num_explored
                )
                if result is not None:
                    if result.failing_schedule is not None:
                        failures.append((num_explored, list(result.failing_schedule)))
                    if self.stop_on_first:
                        return replace(result, failures=failures)
                    if first_failure is None:
                        first_failure = result
                    # An aborted execution (deadlock or worker error) unwinds its
                    # workers via SchedulerAbort before their remaining accesses
                    # are reported, so the engine never seeds the wakeup tree from
                    # this trace and next_execution() can return False with
                    # interleavings still unexplored. Demote exhausted rather than
                    # over-claim coverage (mirrors _evaluate building these results
                    # with exhausted=False; an invariant failure completes fully so
                    # it does NOT demote). Only max_executions/total_timeout were
                    # previously handled, in the for..else below.
                    if result.failure_kind in ("deadlock", "worker_error", "timeout"):
                        exhausted = False

                # In reuse mode a worker aborted mid-iteration leaves stray
                # frames on its persistent socket; sending the next ITER_START
                # would desync it. Stop reusing rather than corrupt the search.
                if self.reuse_workers and unclean:
                    exhausted = False
                    break
            else:
                # The iterator ended on its own. That is only genuine exhaustion
                # if no bound truncated the search: next_execution() returns False
                # identically for an empty wakeup tree and a hit max_executions cap
                # (engine.rs), and the iterator also stops on total_timeout. Report
                # exhausted honestly so a bounded run doesn't over-claim coverage.
                if self.max_executions is not None and num_explored >= self.max_executions:
                    exhausted = False
                elif deadline is not None and time.monotonic() > deadline:
                    exhausted = False
            if first_failure is not None:
                return replace(first_failure, iterations=num_explored, exhausted=exhausted, failures=failures)
            return CrossProcessResult(ok=True, iterations=num_explored, exhausted=exhausted)
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
                    worker_set.join(persistent_handles, self.deadlock_timeout)
            uninstall_wait_for_graph()
            listener.close()
            self._cleanup_socket()

    def _drive_relays(
        self,
        scheduler: DporScheduler,
        socks_by_id: dict[int, socket.socket],
        accesses: list[tuple[int, str, str]],
        worker_errors: dict[int, str],
        unclean: set[int],
        total_deadline: float | None = None,
    ) -> None:
        accesses_lock = threading.Lock()
        # Relays update this heartbeat on every received frame or issued grant.
        progress = [time.monotonic()]
        relays = [
            threading.Thread(
                target=_relay_loop,
                args=(
                    scheduler,
                    wid,
                    sock,
                    accesses,
                    accesses_lock,
                    worker_errors,
                    unclean,
                    progress,
                    self.deadlock_timeout,
                ),
                name=f"xproc-relay-{wid}",
                daemon=True,
            )
            for wid, sock in socks_by_id.items()
        ]
        for t in relays:
            t.start()
        # Socket and scheduler timeouts handle normal stalls. This backstop
        # prevents a stuck relay from surviving into the next iteration.
        no_progress_budget = self._relay_no_progress_budget
        pending = list(relays)
        timeout_error: Exception | None = None
        while pending:
            pending[0].join(timeout=0.05)
            pending = [t for t in pending if t.is_alive()]
            if not pending:
                break
            now = time.monotonic()
            if total_deadline is not None and now > total_deadline:
                # total_timeout must bound the search even mid-execution: a
                # worker that keeps sending frames resets both the per-recv
                # timeout and this heartbeat, so neither liveness guard would
                # ever end the run (the engine's max_branches step cap would,
                # eventually — a step count, not the user's time budget).
                timeout_error = _TotalTimeoutExpired(
                    f"total_timeout={self.total_timeout}s expired while an execution was in flight"
                )
                break
            if now - progress[0] > no_progress_budget:
                names = ", ".join(t.name for t in pending)
                timeout_error = TimeoutError(
                    f"cross-process relay thread(s) {names} made no progress within "
                    f"{no_progress_budget}s and did not terminate; aborting to avoid a "
                    f"concurrent-engine data race"
                )
                break
        if timeout_error is None:
            return

        report_error = getattr(scheduler, "report_error", None)
        if callable(report_error):
            report_error(timeout_error)
        for sock in socks_by_id.values():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        cleanup_deadline = time.monotonic() + max(1.0, min(self.deadlock_timeout, 5.0))
        for t in relays:
            t.join(max(0.0, cleanup_deadline - time.monotonic()))
        raise timeout_error

    def _run_spawned(
        self,
        listener: socket.socket,
        worker_set: WorkerSet,
        scheduler: DporScheduler,
        accesses: list[tuple[int, str, str]],
        worker_errors: dict[int, str],
        unclean: set[int],
        total_deadline: float | None = None,
    ) -> None:
        handles = worker_set.launch(worker_targets(self.socket_path, list(range(self.num_workers))))
        socks_by_id: dict[int, socket.socket] = {}
        try:
            try:
                for _ in range(self.num_workers):
                    sock, wid = accept_hello_live(listener, worker_set, handles, self._connect_budget)
                    socks_by_id[wid] = sock
            except (TimeoutError, OSError) as exc:
                # Reap dead children so the WorkerSet can read their exit/stderr,
                # then surface the real cause instead of a bare connect timeout.
                worker_set.join(handles, self.deadlock_timeout)
                raise _launch_error(worker_set, handles, exc) from exc
            self._drive_relays(scheduler, socks_by_id, accesses, worker_errors, unclean, total_deadline)
        finally:
            for s in socks_by_id.values():
                try:
                    s.close()
                except OSError:
                    pass
            worker_set.join(handles, self.deadlock_timeout)

    def _run_reused(
        self,
        socks_by_id: dict[int, socket.socket],
        worker_set: WorkerSet,
        scheduler: DporScheduler,
        accesses: list[tuple[int, str, str]],
        worker_errors: dict[int, str],
        unclean: set[int],
        total_deadline: float | None = None,
    ) -> None:
        customizer = worker_set if isinstance(worker_set, IterationCustomizer) else None
        for wid, sock in socks_by_id.items():
            if customizer is not None:
                msg = customizer.iter_start_message(wid)
            else:
                msg = {"t": proto.ITER_START}
            proto.send_msg(sock, msg)
        self._drive_relays(scheduler, socks_by_id, accesses, worker_errors, unclean, total_deadline)

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
        """Return a failing result for this execution, or None if it held.

        The caller decides whether to stop (``stop_on_first``) or keep
        exploring; this never silently drops a failure.
        """
        with engine_lock:
            schedule_trace = list(execution.schedule_trace)
        err = scheduler._error

        def _fail(failure: str, kind: str) -> CrossProcessResult:
            return CrossProcessResult(
                ok=False,
                iterations=num_explored,
                exhausted=False,
                failing_schedule=schedule_trace,
                failure=failure,
                failure_kind=kind,
                accesses=accesses,
            )

        # Deadlock first: a row-lock cycle aborts the holder, whose worker then
        # often raises too, so checking worker_errors first would mask the
        # deadlock behind that induced crash (mirrors the in-process priority).
        if isinstance(err, DeadlockError):
            return _fail(getattr(err, "cycle_description", None) or str(err), "deadlock")
        if worker_errors:
            wid = min(worker_errors)
            return _fail(f"worker {wid} failed: {worker_errors[wid]}", "worker_error")
        # A scheduler fallback TimeoutError means the run free-ran unscheduled
        # (unmodeled DB-level blocking, or a statement slower than
        # deadlock_timeout). Its final state describes no DPOR schedule, so the
        # invariant is skipped — but the execution must count as a failure, not
        # a clean pass: the schedule was never driven to completion and nothing
        # was verified.
        if isinstance(err, TimeoutError):
            return _fail(
                f"scheduler timed out (deadlock_timeout={self.deadlock_timeout}s expired): {err}. "
                "A worker blocked outside frontrun's model (e.g. database-level locking) or a "
                "statement ran longer than deadlock_timeout; raise deadlock_timeout if the "
                "workload is just slow.",
                "timeout",
            )
        if not invariant():
            return _fail("invariant violated", "invariant")
        return None

    def _cleanup_socket(self) -> None:
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        if self._own_dir is not None:
            shutil.rmtree(self._own_dir, ignore_errors=True)
