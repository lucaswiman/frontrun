"""
Deadlock detection for frontrun schedulers.

Provides two complementary detection mechanisms:

1. **All-threads-waiting detection** (option 1): When every non-done thread
   is parked inside ``wait_for_turn()`` and none of them is the scheduled
   thread, no progress is possible.  The scheduler can detect this instantly
   instead of waiting for a timeout.

2. **Wait-for graph cycle detection** (option 3): Cooperative primitives
   register ``(thread, lock)`` waiting/holding edges.  A cycle in the
   resulting directed graph means a true lock-ordering deadlock.

Both mechanisms fire instantly when the deadlock condition is met.  A
fallback wall-clock timeout is still kept for threads stuck in C extensions
or other unmanaged blocking calls.
"""

from __future__ import annotations

from typing import Literal

from frontrun._real_threading import lock as _real_lock
from frontrun._real_threading import rlock as _real_rlock

LockKind = Literal["lock", "row_lock"]


class DeadlockError(Exception):
    """Raised when WaitForGraph detects a lock-ordering deadlock cycle."""

    def __init__(self, message: str, cycle_description: str) -> None:
        super().__init__(message)
        self.cycle_description = cycle_description


class SchedulerAbortError(Exception):
    """Raised inside cooperative spin loops when the scheduler has errored.

    Cooperative primitives catch this internally and bail out of their
    spin-yield loops.  Thread entry points in BytecodeShuffler /
    DporBytecodeRunner catch it and propagate via ``scheduler.report_error``.
    """


# Convenience alias used throughout the codebase.
SchedulerAbort = SchedulerAbortError


# ---------------------------------------------------------------------------
# Wait-for graph
# ---------------------------------------------------------------------------


class WaitForGraph:
    """Thread-safe directed graph for deadlock (cycle) detection.

    Edges represent "thread T is waiting for lock L" and "lock L is held by
    thread H".  A cycle ``T1 -> L1 -> T2 -> L2 -> T1`` means threads T1 and
    T2 are in a lock-ordering deadlock.

    Nodes are ``("thread", thread_id)``, ``("lock", lock_object_id)``, or
    ``("row_lock", row_lock_int_id)``.
    """

    def __init__(self) -> None:
        # Use RLock to allow reentrant access when GC-triggered __del__
        # chains fire during graph operations.  See defect #7.
        self._lock = _real_rlock()
        # adjacency: node -> set of successor nodes
        self._edges: dict[tuple[str, int], set[tuple[str, int]]] = {}

    def add_waiting(self, thread_id: int, lock_id: int, kind: LockKind = "lock") -> list[tuple[str, int]] | None:
        """Record that *thread_id* is waiting for *lock_id*.

        Returns the cycle path if adding this edge creates a cycle,
        otherwise ``None``.
        """
        src = ("thread", thread_id)
        dst = (kind, lock_id)
        with self._lock:
            self._edges.setdefault(src, set()).add(dst)
            cycle = self._find_cycle_from(src)
            if cycle is not None:
                return cycle
            return None

    def remove_waiting(self, thread_id: int, lock_id: int, kind: LockKind = "lock") -> None:
        """Remove the waiting edge (thread acquired or gave up)."""
        src = ("thread", thread_id)
        dst = (kind, lock_id)
        with self._lock:
            succs = self._edges.get(src)
            if succs is not None:
                succs.discard(dst)
                if not succs:
                    del self._edges[src]

    def add_holding(self, thread_id: int, lock_id: int, kind: LockKind = "lock") -> None:
        """Record that *lock_id* is held by *thread_id*."""
        src = (kind, lock_id)
        dst = ("thread", thread_id)
        with self._lock:
            self._edges.setdefault(src, set()).add(dst)

    def remove_holding(self, thread_id: int, lock_id: int, kind: LockKind = "lock") -> None:
        """Remove the holding edge (thread released the lock)."""
        src = (kind, lock_id)
        dst = ("thread", thread_id)
        with self._lock:
            succs = self._edges.get(src)
            if succs is not None:
                succs.discard(dst)
                if not succs:
                    del self._edges[src]

    def clear(self) -> None:
        """Remove all edges."""
        with self._lock:
            self._edges.clear()

    def reverse_reachable_threads_from(self, thread_id: int) -> set[int]:
        """Return threads transitively blocked behind *thread_id*'s held locks.

        Traverses the wait-for graph with edges reversed:

        - ``lock -> holder`` becomes ``holder -> lock``
        - ``thread -> waited_lock`` becomes ``waited_lock -> thread``

        Starting from ``("thread", thread_id)``, the reachable thread nodes are
        exactly the threads that are directly or transitively waiting on locks
        currently held (possibly indirectly) by *thread_id*.
        """
        start = ("thread", thread_id)
        with self._lock:
            reverse_edges: dict[tuple[str, int], set[tuple[str, int]]] = {}
            for src, dsts in self._edges.items():
                for dst in dsts:
                    reverse_edges.setdefault(dst, set()).add(src)

            visited: set[tuple[str, int]] = {start}
            stack: list[tuple[str, int]] = [start]
            reachable_threads: set[int] = set()

            while stack:
                node = stack.pop()
                for neighbor in reverse_edges.get(node, ()):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
                    if neighbor[0] == "thread":
                        reachable_threads.add(neighbor[1])

            return reachable_threads

    # -- internal ----------------------------------------------------------

    def _find_cycle_from(self, start: tuple[str, int]) -> list[tuple[str, int]] | None:
        """DFS from *start* looking for a cycle back to *start*.

        Must be called with ``self._lock`` held.
        """
        visited: set[tuple[str, int]] = {start}  # include start so loop-back is detected
        path: list[tuple[str, int]] = []

        def dfs(node: tuple[str, int]) -> bool:
            if node in visited:
                if node == start:
                    return True
                return False
            visited.add(node)
            path.append(node)
            for neighbor in self._edges.get(node, ()):
                if dfs(neighbor):
                    return True
            path.pop()
            return False

        for neighbor in self._edges.get(start, ()):
            if dfs(neighbor):
                return [start, *path, start]
        return None


def format_cycle(
    cycle: list[tuple[str, int]],
    lock_names: dict[int, str] | None = None,
    lock_id_map: dict[int, int] | None = None,
) -> str:
    """Human-readable description of a deadlock cycle.

    *lock_id_map* maps raw ``id(lock_obj)`` → stable integer ID (from
    :class:`StableObjectIds`).  When provided, regular locks are formatted as
    ``lock <stable_id>`` instead of ``lock 0x<hex_addr>``, keeping the cycle
    description consistent with the lock-event data in HTML reports.
    """
    parts: list[str] = []
    for kind, ident in cycle:
        if kind == "thread":
            parts.append(f"thread {ident}")
        elif kind == "row_lock" and lock_names and ident in lock_names:
            parts.append(f"row_lock {lock_names[ident]}")
        elif kind == "row_lock":
            parts.append(f"row_lock 0x{ident:x}")
        elif kind == "lock" and lock_id_map and ident in lock_id_map:
            parts.append(f"lock {lock_id_map[ident]}")
        else:
            parts.append(f"lock 0x{ident:x}")
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Global wait-for graph instance (shared across all cooperative primitives)
# ---------------------------------------------------------------------------

_global_graph: WaitForGraph | None = None
_graph_lock = _real_lock()


def get_wait_for_graph() -> WaitForGraph | None:
    """Return the active wait-for graph, or ``None`` if not installed."""
    return _global_graph


def install_wait_for_graph() -> WaitForGraph:
    """Install (or return existing) global wait-for graph."""
    global _global_graph  # noqa: PLW0603
    with _graph_lock:
        if _global_graph is None:
            _global_graph = WaitForGraph()
        return _global_graph


def uninstall_wait_for_graph() -> None:
    """Remove the global wait-for graph."""
    global _global_graph  # noqa: PLW0603
    with _graph_lock:
        if _global_graph is not None:
            _global_graph.clear()
        _global_graph = None
