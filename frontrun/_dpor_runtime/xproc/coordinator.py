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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from frontrun._dpor_core import LivenessProbe, RowLockRegistry, WorkerSet, WorkerTarget

from . import protocol as proto


def worker_targets(socket_path: str, worker_ids: list[int]) -> list[WorkerTarget]:
    """Build backend-neutral targets carrying the coordinator socket path."""
    return [WorkerTarget(worker_id=wid, args=(socket_path,)) for wid in worker_ids]


def accept_hello(listener: socket.socket, timeout: float) -> tuple[socket.socket, int]:
    """Accept one worker connection and read its HELLO frame, returning (sock, worker_id).

    Shared by the exhaustive and DPOR coordinators so accept order is decoupled
    from worker ids.
    """
    sock, _addr = listener.accept()
    sock.settimeout(timeout)
    hello = proto.recv_msg(sock)
    if hello is None or hello.get("t") != proto.HELLO:
        raise RuntimeError(f"expected HELLO frame, got {hello!r}")
    return sock, int(hello["w"])


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
    exhausted: bool
    failing_schedule: list[int] | None = None
    failure: str | None = None
    # One of: "invariant", "worker_error", "deadlock", "nondeterministic", None.
    failure_kind: str | None = None
    accesses: list[tuple[int, str, str]] | None = None


class _Conn:
    """Coordinator-side view of one connected worker."""

    def __init__(self, worker_id: int, sock: socket.socket) -> None:
        self.worker_id = worker_id
        self.sock = sock
        self.pending: dict[str, Any] | None = None  # blocking request awaiting a grant
        self.done = False
        self.error: str | None = None


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
    stop: str | None
    errors: dict[int, str]


class CrossProcessCoordinator:
    def __init__(
        self,
        *,
        num_workers: int,
        socket_path: str | None = None,
        deadlock_timeout: float = 10.0,
    ) -> None:
        self.num_workers = num_workers
        self.deadlock_timeout = deadlock_timeout
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
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        listener.listen(self.num_workers)
        listener.settimeout(self.deadlock_timeout)
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
                return CrossProcessResult(
                    ok=False,
                    iterations=iterations + 1,
                    exhausted=False,
                    failure=f"worker connection failed: {type(exc).__name__}: {exc}",
                    failure_kind="worker_error",
                )
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
            for _ in range(self.num_workers):
                sock, wid = accept_hello_live(listener, worker_set, handles, self.deadlock_timeout)
                conn = _Conn(wid, sock)
                conns[wid] = conn
                self._advance(conn, accesses, registry)
            schedule, branch_points, stop = self._drive(conns, prefix, accesses, registry)
            errors = {wid: c.error for wid, c in conns.items() if c.error is not None}
            return _Outcome(schedule, branch_points, accesses, stop, errors)
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
            except (TimeoutError, OSError):
                msg = None
            if msg is None:
                conn.done = True
                conn.pending = None
                if conn.error is None:
                    conn.error = "worker disconnected or timed out"
                registry.pop_all(conn.worker_id, None)
                return
            kind = msg["t"]
            if kind == proto.ACCESS:
                accesses.append((conn.worker_id, msg["rid"], msg["kind"]))
            elif kind == proto.RELEASE_LOCKS:
                registry.pop_all(conn.worker_id, None)
            elif kind in (proto.REPORT_AND_WAIT, proto.ACQUIRE_LOCKS):
                conn.pending = msg
                return
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

    def _cleanup_socket(self) -> None:
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        if self._own_dir is not None:
            shutil.rmtree(self._own_dir, ignore_errors=True)
