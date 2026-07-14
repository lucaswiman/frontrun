"""Coordinator for cross-process DPOR exploration (Phase 1: no Rust engine).

The coordinator owns the exploration: it accepts worker connections over an
``AF_UNIX`` socket, and drives one interleaving per iteration by granting a
single worker at a time at each external-access scheduling point. Because
separate workers share no memory, the only ordering that matters is the order
of external accesses — so the coordinator enumerates interleavings at that
(coarse) granularity.

Phase 1 enumerates the interleaving space *exhaustively* by depth-first search
with spawn-per-iteration replay: each iteration re-runs ``setup`` then all
workers, replaying a chosen decision prefix and then diverging to an unexplored
branch. Row-lock (``SELECT FOR UPDATE``) arbitration reuses the shared
:class:`RowLockRegistry`. Feeding accesses into the Rust DPOR engine (to prune
equivalent interleavings) is a later slice; the protocol and worker side are
already engine-agnostic.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from frontrun._dpor_core import LivenessProbe, RowLockRegistry, WorkerSet, WorkerTarget

from . import protocol as proto


def worker_targets(socket_path: str, worker_ids: list[int]) -> list[WorkerTarget]:
    """Build backend-neutral targets carrying the coordinator socket path."""
    return [WorkerTarget(worker_id=wid, args=(socket_path,)) for wid in worker_ids]


class _WorkerLaunchError(OSError):
    """A worker failed to connect; its message already carries child diagnostics."""


def _launch_error(worker_set: WorkerSet, handles: Any, exc: Exception) -> Exception:
    """Enrich a connect failure with the WorkerSet's diagnosis of dead children.

    Turns a bare ``TimeoutError`` (worker never sent HELLO) into a message naming
    the real cause — e.g. a child that exited with ``ModuleNotFoundError`` for a
    bad ``module:callable`` target — when the WorkerSet can recover it.  Shared
    by both coordinators: the launchers capture each child's stderr precisely so
    ``diagnose()`` can surface it, and which strategy the user picked must not
    decide whether they see it.
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


def bind_coordination_listener(
    socket_path: str,
    num_workers: int,
    timeout: float,
    on_error: Callable[[], None],
) -> socket.socket:
    """Create, bind, and listen the AF_UNIX coordination socket.

    Shared by the exhaustive and DPOR coordinators.  Binding happens before
    the coordinators enter their cleanup try/finally, so a bind failure —
    realistically ``socket_path`` exceeding the AF_UNIX ~108-byte limit under
    a deep ``$TMPDIR`` — must run *on_error* (the coordinator's socket/tempdir
    cleanup) here rather than leak the mkdtemp'd working directory on every
    call, and must say which path failed and what to do about it.
    """
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(socket_path)
        listener.listen(num_workers)
        listener.settimeout(timeout)
    except OSError as exc:
        listener.close()
        on_error()
        raise OSError(
            f"cannot bind cross-process coordination socket {socket_path!r}: {exc} "
            "(AF_UNIX socket paths are limited to ~108 bytes on Linux; set a shorter "
            "$TMPDIR or pass an explicit short socket_path=)"
        ) from exc
    return listener


def accept_hello(listener: socket.socket, timeout: float) -> tuple[socket.socket, int]:
    """Accept one worker connection and read its HELLO frame, returning (sock, worker_id).

    Shared by the exhaustive and DPOR coordinators so accept order is decoupled
    from worker ids.
    """
    sock, _addr = listener.accept()
    sock.settimeout(timeout)
    # A worker can connect and then die (crash/OOM/os._exit) or send garbage
    # before its HELLO.  Treat that as a *connection* failure (OSError, which
    # accept_hello_live and the coordinators already catch and route through
    # the liveness diagnostics), not an internal RuntimeError that escapes
    # explore().  Always close the accepted socket on the error path so the fd
    # is not leaked.
    try:
        hello = proto.recv_msg(sock)
    except BaseException:
        sock.close()
        raise
    if hello is None or hello.get("t") != proto.HELLO or "w" not in hello:
        sock.close()
        raise OSError(f"expected HELLO frame, got {hello!r}")
    try:
        worker_id = int(hello["w"])
    except (TypeError, ValueError):
        # A non-integer id must be a connection failure (OSError, which the
        # coordinators catch and structure), not a bare ValueError escaping
        # explore() with the accepted socket leaked.
        sock.close()
        raise OSError(f"HELLO frame carries a non-integer worker id: {hello!r}") from None
    return sock, worker_id


def check_worker_id(worker_id: int, num_workers: int, taken: Collection[int], sock: socket.socket) -> None:
    """Validate a HELLO-announced worker id against the expected dense range.

    A duplicate id would silently overwrite the previous worker's connection
    (that worker is then never driven — a false ``ok=True`` over half the
    workload); an out-of-range id would flow into scheduling bookkeeping (and,
    on the DPOR path, into the Rust engine) under a nonsense thread id.  Both
    close the accepted socket and raise OSError so the coordinators' existing
    connection-failure handling reports the real cause.
    """
    if worker_id in taken:
        sock.close()
        raise OSError(f"duplicate worker id {worker_id} announced by a second HELLO")
    if not 0 <= worker_id < num_workers:
        sock.close()
        raise OSError(f"worker id {worker_id} out of range for num_workers={num_workers}")


def accept_hello_live(
    listener: socket.socket,
    worker_set: Any,
    handles: Any,
    connect_budget: float,
) -> tuple[socket.socket, int]:
    """Accept one HELLO, failing fast if a launched child dies before connecting.

    A worker that crashes at startup (bad ``module:callable`` target, import
    error) never connects, so a plain ``accept`` would block the full connect
    budget before timing out. Poll ``worker_set.diagnose(handles)`` between short
    accept attempts and raise as soon as a child has exited, so the coordinator
    surfaces the real cause in ~a poll interval instead of tens of seconds.
    """
    deadline = time.monotonic() + connect_budget
    prev = listener.gettimeout()
    listener.settimeout(min(0.5, connect_budget))
    try:
        while True:
            try:
                return accept_hello(listener, connect_budget)
            except (TimeoutError, OSError):
                # accept() timed out with no connection yet (the accepted socket
                # keeps the full connect_budget, so this is not a slow HELLO).
                # any_exited / all_exited are non-destructive; the stderr-reading
                # diagnose() is left for the failure path so it is not consumed
                # here.
                if isinstance(worker_set, LivenessProbe):
                    if worker_set.any_exited(handles):
                        raise TimeoutError("worker exited before connecting") from None
                    # A child can also exit *cleanly* (code 0) before ever sending
                    # HELLO — e.g. a target that calls sys.exit(0) at import. That
                    # is invisible to any_exited (nonzero-only), so without this
                    # the accept loop would block the whole connect budget.
                    if worker_set.all_exited(handles):
                        raise TimeoutError("all workers exited before connecting") from None
                if time.monotonic() >= deadline:
                    raise TimeoutError("workers did not connect within the deadlock timeout") from None
    finally:
        listener.settimeout(prev)


@dataclass
class CrossProcessResult:
    """Outcome of a cross-process exploration."""

    ok: bool
    iterations: int
    # True only when the search space was genuinely fully covered: any
    # truncating bound (max_iterations / max_executions / total_timeout, or a
    # non-None preemption_bound on the DPOR strategy) demotes this to False.
    exhausted: bool
    failing_schedule: list[int] | None = None
    failure: str | None = None
    # One of: "invariant", "worker_error", "deadlock", "timeout",
    # "nondeterministic", "step_limit" (exhaustive: max_steps_per_run hit),
    # "branch_limit" (DPOR: max_branches hit), None.
    failure_kind: str | None = None
    accesses: list[tuple[int, str, str]] | None = None
    # Mapping-input labels keyed by the dense numeric ids used in schedules and
    # access traces. Empty for sequence/single-worker inputs and direct
    # coordinator use.
    worker_labels: dict[int, str] = field(default_factory=dict)
    # Every failing execution as (execution_number, schedule) pairs, mirroring
    # thread-mode InterleavingResult.failures. Both strategies populate it for
    # any failure that carries a failing_schedule; the DPOR coordinator with
    # stop_on_first=False accumulates ALL failing executions instead of only
    # the first.
    failures: list[tuple[int, list[int]]] = field(default_factory=list)


class _Conn:
    """Coordinator-side view of one connected worker."""

    def __init__(self, worker_id: int, sock: socket.socket) -> None:
        self.worker_id = worker_id
        self.sock = sock
        self.pending: dict[str, Any] | None = None  # blocking request awaiting a grant
        self.done = False
        self.error: str | None = None
        # recv timed out with the socket still open: the worker is alive but
        # silent past deadlock_timeout (distinct from a disconnect/EOF).
        self.timed_out = False


@dataclass
class _Outcome:
    schedule: list[int]
    branch_points: list[list[int]]
    accesses: list[tuple[int, str, str]]
    # Why _drive stopped short of finishing every worker, if it did:
    #   None             — all workers finished cleanly
    #   "deadlock"       — a genuine cross-worker deadlock (no runnable worker)
    #   "nondeterministic" — the recorded prefix choice is no longer grantable
    #                        during replay (a nondeterministic workload)
    #   "step_limit"     — the run exceeded max_steps_per_run scheduling points
    #                      without finishing (nonterminating worker)
    stop: str | None
    errors: dict[int, str]
    # Workers whose recv timed out while still connected (alive but silent past
    # deadlock_timeout), mapped to a human-facing diagnosis.
    timeouts: dict[int, str]


class CrossProcessCoordinator:
    def __init__(
        self,
        *,
        num_workers: int,
        socket_path: str | None = None,
        deadlock_timeout: float = 10.0,
        max_steps_per_run: int = 100_000,
    ) -> None:
        self.num_workers = num_workers
        self.deadlock_timeout = deadlock_timeout
        # Per-run bound on scheduling points. A nonterminating worker
        # (``while True`` around scheduled statements) keeps frames arriving,
        # so the per-recv deadlock_timeout never fires and max_iterations only
        # bounds *completed* iterations — without this cap explore() would hang
        # forever. Generous by default (matching the DPOR path's max_branches
        # spirit); raise it if a workload genuinely has more scheduling points.
        self.max_steps_per_run = max_steps_per_run
        self._own_dir: str | None = None
        if socket_path is None:
            self._own_dir = tempfile.mkdtemp(prefix="frontrun-xproc-")
            socket_path = os.path.join(self._own_dir, "s")
        self.socket_path = socket_path

    def explore(
        self,
        *,
        worker_set: WorkerSet,
        setup: Callable[[], Any],
        invariant: Callable[[], bool],
        max_iterations: int = 4096,
    ) -> CrossProcessResult:
        listener = bind_coordination_listener(
            self.socket_path, self.num_workers, self.deadlock_timeout, self._cleanup_socket
        )
        try:
            return self._explore(listener, worker_set, setup, invariant, max_iterations)
        finally:
            listener.close()
            self._cleanup_socket()

    # -- exploration loop ---------------------------------------------------

    def _explore(
        self,
        listener: socket.socket,
        worker_set: WorkerSet,
        setup: Callable[[], Any],
        invariant: Callable[[], bool],
        max_iterations: int,
    ) -> CrossProcessResult:
        stack: list[list[int]] = [[]]
        seen: set[tuple[int, ...]] = set()
        iterations = 0
        exhausted = True
        while stack:
            if iterations >= max_iterations:
                exhausted = False
                break
            prefix = stack.pop()
            key = tuple(prefix)
            if key in seen:
                continue
            seen.add(key)

            setup()
            try:
                outcome = self._run_once(listener, worker_set, prefix)
            except (TimeoutError, OSError) as exc:
                return _connection_failure(exc, iterations + 1)
            iterations += 1

            if outcome.errors:
                wid, msg = next(iter(sorted(outcome.errors.items())))
                return CrossProcessResult(
                    ok=False,
                    iterations=iterations,
                    exhausted=False,
                    failing_schedule=outcome.schedule,
                    failure=f"worker {wid} failed: {msg}",
                    failure_kind="worker_error",
                    accesses=outcome.accesses,
                    failures=[(iterations, list(outcome.schedule))],
                )
            if outcome.timeouts:
                _wid, msg = next(iter(sorted(outcome.timeouts.items())))
                return CrossProcessResult(
                    ok=False,
                    iterations=iterations,
                    exhausted=False,
                    failing_schedule=outcome.schedule,
                    failure=msg,
                    failure_kind="timeout",
                    accesses=outcome.accesses,
                    failures=[(iterations, list(outcome.schedule))],
                )
            if outcome.stop == "step_limit":
                return CrossProcessResult(
                    ok=False,
                    iterations=iterations,
                    exhausted=False,
                    failure=(
                        f"run exceeded max_steps_per_run={self.max_steps_per_run} scheduling points without "
                        "completing; a worker may be nonterminating (e.g. an unbounded loop around scheduled "
                        "statements). Raise max_steps_per_run if the workload genuinely runs this long."
                    ),
                    failure_kind="step_limit",
                )
            if outcome.stop == "deadlock":
                return CrossProcessResult(
                    ok=False,
                    iterations=iterations,
                    exhausted=False,
                    failing_schedule=outcome.schedule,
                    failure="cross-worker deadlock (no runnable worker)",
                    failure_kind="deadlock",
                    accesses=outcome.accesses,
                    failures=[(iterations, list(outcome.schedule))],
                )
            if outcome.stop == "nondeterministic":
                return CrossProcessResult(
                    ok=False,
                    iterations=iterations,
                    exhausted=False,
                    failing_schedule=outcome.schedule,
                    failure="recorded schedule no longer reproducible (nondeterministic workload?)",
                    failure_kind="nondeterministic",
                    accesses=outcome.accesses,
                    failures=[(iterations, list(outcome.schedule))],
                )
            if not invariant():
                return CrossProcessResult(
                    ok=False,
                    iterations=iterations,
                    exhausted=False,
                    failing_schedule=outcome.schedule,
                    failure="invariant violated",
                    failure_kind="invariant",
                    accesses=outcome.accesses,
                    failures=[(iterations, list(outcome.schedule))],
                )

            # Push a fresh prefix for every unexplored alternative at each
            # decision point at or beyond this run's divergence depth. Each
            # pushed prefix diverges from the greedy continuation at its last
            # element, so every interleaving is reached exactly once.
            for i in range(len(prefix), len(outcome.schedule)):
                chosen = outcome.schedule[i]
                stack.extend([*outcome.schedule[:i], alt] for alt in outcome.branch_points[i] if alt != chosen)

        return CrossProcessResult(ok=True, iterations=iterations, exhausted=exhausted)

    # -- one interleaving ---------------------------------------------------

    def _run_once(self, listener: socket.socket, worker_set: WorkerSet, prefix: list[int]) -> _Outcome:
        handles = worker_set.launch(worker_targets(self.socket_path, list(range(self.num_workers))))
        registry = RowLockRegistry()
        conns: dict[int, _Conn] = {}
        accesses: list[tuple[int, str, str]] = []
        try:
            try:
                for _ in range(self.num_workers):
                    sock, wid = accept_hello_live(listener, worker_set, handles, self.deadlock_timeout)
                    check_worker_id(wid, self.num_workers, conns, sock)
                    conn = _Conn(wid, sock)
                    conns[wid] = conn
                    self._advance(conn, accesses, registry)
            except (TimeoutError, OSError) as exc:
                # Reap dead children so the WorkerSet can read their exit/stderr,
                # then surface the real cause instead of a bare connect timeout
                # (mirrors the DPOR coordinator's _run_spawned).
                worker_set.join(handles, self.deadlock_timeout)
                raise _launch_error(worker_set, handles, exc) from exc
            schedule, branch_points, stop = self._drive(conns, prefix, accesses, registry)
            errors = {wid: c.error for wid, c in conns.items() if c.error is not None}
            timeouts: dict[int, str] = {}
            for wid, c in sorted(conns.items()):
                if not c.timed_out:
                    continue
                # A recv timeout means the socket stayed open but silent — the
                # worker is (almost certainly) alive. Confirm with the
                # WorkerSet's liveness probe when it has one: a child that
                # crashed without closing the socket is a worker_error, not a
                # too-small deadlock_timeout.
                if isinstance(worker_set, LivenessProbe) and worker_set.any_exited(handles):
                    # any_exited/diagnose are fleet-wide: the process that died
                    # need not be *wid* (which merely went silent), so the
                    # message must not claim wid exited — diagnose names the
                    # worker(s) that actually did.
                    detail = worker_set.diagnose(handles)
                    errors.setdefault(
                        wid,
                        f"sent no frame within deadlock_timeout={self.deadlock_timeout}s during a run in "
                        f"which a worker process exited ({detail or 'nonzero exit'})",
                    )
                    continue
                timeouts[wid] = (
                    f"worker {wid} sent no frame within deadlock_timeout={self.deadlock_timeout}s but is still "
                    "running: it blocked outside frontrun's model (e.g. database-level locking) or a statement "
                    "ran longer than deadlock_timeout; raise deadlock_timeout if the workload is just slow"
                )
            return _Outcome(schedule, branch_points, accesses, stop, errors, timeouts)
        finally:
            for c in conns.values():
                try:
                    c.sock.close()
                except OSError:
                    pass
            worker_set.join(handles, self.deadlock_timeout)

    def _drive(
        self,
        conns: dict[int, _Conn],
        prefix: list[int],
        accesses: list[tuple[int, str, str]],
        registry: RowLockRegistry,
    ) -> tuple[list[int], list[list[int]], str | None]:
        schedule: list[int] = []
        branch_points: list[list[int]] = []
        step = 0
        while True:
            grantable = self._grantable(conns, registry)
            if not grantable:
                if all(c.done for c in conns.values()):
                    return schedule, branch_points, None
                return schedule, branch_points, "deadlock"  # some worker stuck: deadlock
            if step >= self.max_steps_per_run:
                return schedule, branch_points, "step_limit"
            branch_points.append(grantable)
            if step < len(prefix):
                choice = prefix[step]
                if choice not in grantable:
                    # Nondeterministic replay: the recorded prefix choice is no
                    # longer grantable, so the schedule cannot be reproduced.
                    # This is a divergent (nondeterministic) workload, NOT a
                    # cross-worker deadlock — surface it as its own stop reason
                    # rather than silently exploring a different tree.
                    return schedule, branch_points, "nondeterministic"
            else:
                choice = grantable[0]
            schedule.append(choice)
            self._grant(conns[choice], accesses, registry)
            step += 1

    def _grantable(self, conns: dict[int, _Conn], registry: RowLockRegistry) -> list[int]:
        out: list[int] = []
        for wid in sorted(conns):
            conn = conns[wid]
            if conn.done or conn.pending is None:
                continue
            kind = conn.pending["t"]
            if kind == proto.REPORT_AND_WAIT:
                out.append(wid)
            elif kind == proto.ACQUIRE_LOCKS:
                resources = conn.pending["res"]
                if all(registry.active_lock_owner(r) in (None, wid) for r in resources):
                    out.append(wid)
        return out

    def _grant(self, conn: _Conn, accesses: list[tuple[int, str, str]], registry: RowLockRegistry) -> None:
        assert conn.pending is not None
        if conn.pending["t"] == proto.ACQUIRE_LOCKS:
            for res in conn.pending["res"]:
                registry.record_acquire(conn.worker_id, res, None)
        conn.pending = None
        proto.send_msg(conn.sock, {"t": proto.GRANT})
        self._advance(conn, accesses, registry)

    def _advance(
        self,
        conn: _Conn,
        accesses: list[tuple[int, str, str]],
        registry: RowLockRegistry,
    ) -> None:
        """Read frames from *conn* until it blocks on a request or finishes."""
        while True:
            try:
                msg = proto.recv_msg(conn.sock)
            except TimeoutError:
                # The socket is still open but silent past deadlock_timeout:
                # the worker is alive but slow/blocked, NOT disconnected.
                # Record that distinctly so _run_once can diagnose
                # failure_kind="timeout" (with raise-deadlock_timeout advice)
                # instead of a misleading "worker disconnected" worker_error.
                conn.done = True
                conn.pending = None
                conn.timed_out = True
                registry.pop_all(conn.worker_id, None)
                return
            except OSError:
                msg = None
            if msg is None:
                conn.done = True
                conn.pending = None
                if conn.error is None:
                    conn.error = "worker disconnected"
                registry.pop_all(conn.worker_id, None)
                return
            kind = msg.get("t")
            if kind == proto.ACCESS:
                rid = msg.get("rid")
                access_kind = msg.get("kind")
                if not isinstance(rid, str) or not isinstance(access_kind, str):
                    self._fail_malformed(conn, msg, registry)
                    return
                accesses.append((conn.worker_id, rid, access_kind))
            elif kind == proto.RELEASE_LOCKS:
                registry.pop(conn.worker_id, None, msg.get("res"))
            elif kind == proto.ACQUIRE_LOCKS:
                if not isinstance(msg.get("res"), list):
                    self._fail_malformed(conn, msg, registry)
                    return
                conn.pending = msg
                return
            elif kind == proto.REPORT_AND_WAIT:
                conn.pending = msg
                return
            elif kind == proto.AFTER_SYNC:
                # The exhaustive scheduler grants one worker at a time and
                # reads that worker until its next blocking request.  The
                # explicit completion frame is therefore only a boundary
                # marker here; unlike DPOR, there is no held engine turn to
                # release.
                continue
            elif kind == proto.DONE:
                conn.done = True
                conn.pending = None
                registry.pop_all(conn.worker_id, None)
                return
            elif kind == proto.ERROR:
                conn.done = True
                conn.pending = None
                conn.error = msg.get("msg", "worker error")
                registry.pop_all(conn.worker_id, None)
                return
            elif kind is None:
                # A frame with no "t" at all — a buggy or desynchronised
                # worker. Structure it like any other worker failure instead
                # of dying on a KeyError that escapes explore().
                self._fail_malformed(conn, msg, registry)
                return
            else:
                # An unexpected frame — e.g. BEFORE_IO/AFTER_IO from a Redis
                # worker, which the exhaustive coordinator does not support.
                # Surface it as a worker error instead of silently swallowing
                # the frame and leaving the worker blocked awaiting a grant.
                conn.done = True
                conn.pending = None
                conn.error = f"unsupported frame {kind!r} (use strategy='dpor' for Redis workers)"
                registry.pop_all(conn.worker_id, None)
                return

    @staticmethod
    def _fail_malformed(conn: _Conn, msg: dict[str, Any], registry: RowLockRegistry) -> None:
        """Mark *conn* failed on a structurally invalid frame (missing/typed-wrong keys)."""
        conn.done = True
        conn.pending = None
        conn.error = f"malformed frame {msg!r}"
        registry.pop_all(conn.worker_id, None)

    def _cleanup_socket(self) -> None:
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        if self._own_dir is not None:
            shutil.rmtree(self._own_dir, ignore_errors=True)
