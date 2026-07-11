"""Shared row-lock registry used by sync (DporScheduler) and async (AsyncDporScheduler)."""

from __future__ import annotations

from typing import Any


class RowLockRegistry:
    """Tracks SQL SELECT FOR UPDATE row-lock ownership and integer IDs.

    Both ``DporScheduler`` (sync) and ``AsyncDporScheduler`` (async) maintain
    identical state for row-lock tracking:

    * ``_active_row_locks``  — resource_id → holder thread/task ID
    * ``_task_row_locks``    — thread/task ID → set of held resource IDs
    * ``_row_lock_ids``      — resource_id → stable integer ID for WaitForGraph
    * ``_row_lock_next_id``  — monotonic counter for ID allocation

    This class holds that shared state and exposes three shared operations:

    * ``_row_lock_int_id(res_id)`` — stable monotonic int ID, byte-for-byte
      identical in both schedulers.
    * ``record_acquire(owner_id, res_id, graph)`` — update ownership dicts and
      call ``graph.add_holding`` after the caller decides to proceed.  Both
      schedulers run this after their (divergent) blocking/non-blocking decisions.
    * ``pop(owner_id, graph, resources)`` — release selected locks (or all when
      ``resources`` is ``None``), call ``graph.remove_holding`` for each, and
      return ``(res_id, int_id)`` pairs for sync-engine release events.
    * ``pop_all(owner_id, graph)`` — compatibility shorthand for releasing all.
    * ``id_to_resource()`` — inverse mapping passed to ``format_cycle``.

    The blocking-vs-non-blocking acquire loop remains scheduler-specific because
    the async scheduler cannot block (single event-loop thread) while the sync
    scheduler waits on a condition variable.
    """

    def __init__(self) -> None:
        # resource_id → holding thread/task ID (exclusive row-lock ownership).
        self._active_row_locks: dict[str, int] = {}
        # Reverse index: thread/task ID → set of held resource IDs.
        # Avoids O(n) scan when releasing all locks for a finished thread/task.
        self._task_row_locks: dict[int, set[str]] = {}
        # resource_id → stable integer ID for WaitForGraph nodes.
        # String resource IDs are assigned monotonically increasing integers so
        # row-lock nodes ("row_lock", int) are disjoint from cooperative-lock
        # nodes ("lock", id(obj)) in the WaitForGraph.
        self._row_lock_ids: dict[str, int] = {}
        self._row_lock_next_id: int = 0

    def _row_lock_int_id(self, res_id: str) -> int:
        """Return a stable monotonic integer ID for *res_id* (allocated on first call)."""
        lid = self._row_lock_ids.get(res_id)
        if lid is None:
            lid = self._row_lock_next_id
            self._row_lock_next_id += 1
            self._row_lock_ids[res_id] = lid
        return lid

    def active_lock_owner(self, res_id: str) -> int | None:
        """Return the worker/task ID currently holding *res_id*, or ``None`` if free.

        Public read-only accessor over ``_active_row_locks`` so schedulers (e.g.
        the cross-process coordinator's grantability check) can test row-lock
        ownership without reaching into private state.
        """
        return self._active_row_locks.get(res_id)

    def id_to_resource(self) -> dict[int, str]:
        """Return the inverse of ``_row_lock_ids`` for :func:`~frontrun._deadlock.format_cycle`.

        Passed as the second argument to ``format_cycle`` so deadlock messages
        display human-readable resource strings rather than opaque integers.
        """
        return {v: k for k, v in self._row_lock_ids.items()}

    def record_acquire(self, owner_id: int, res_id: str, graph: Any) -> None:
        """Record that *owner_id* now holds *res_id* and notify *graph*.

        Call this **after** any blocking/non-blocking decision has been made
        (i.e. the caller has confirmed it is safe to proceed).

        Updates ``_active_row_locks`` and ``_task_row_locks``, and calls
        ``graph.add_holding(owner_id, int_id, kind="row_lock")`` if *graph*
        is not ``None``.
        """
        lid = self._row_lock_int_id(res_id)
        prev_owner = self._active_row_locks.get(res_id)
        if prev_owner is not None and prev_owner != owner_id:
            # Ownership transfer: scrub the old holder's bookkeeping so a later
            # pop_all(prev_owner) cannot remove this resource from
            # _active_row_locks and corrupt the new owner's state.
            prev_set = self._task_row_locks.get(prev_owner)
            if prev_set is not None:
                prev_set.discard(res_id)
            if graph is not None:
                graph.remove_holding(prev_owner, lid, kind="row_lock")
        self._active_row_locks[res_id] = owner_id
        self._task_row_locks.setdefault(owner_id, set()).add(res_id)
        if graph is not None:
            graph.add_holding(owner_id, lid, kind="row_lock")

    def pop(
        self,
        owner_id: int,
        graph: Any,
        resources: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        """Release selected row locks, or all locks when *resources* is ``None``.

        Only locks currently held by *owner_id* are affected. Remaining locks
        stay in both ownership indexes and in the wait-for graph.

        Returns:
            A list of ``(res_id, int_id)`` pairs for every released resource,
            so the caller can pass each to ``engine.report_sync`` if needed
            (the sync scheduler does; the async scheduler does not).
            Returns ``[]`` if none of the requested locks were held.
        """
        held = self._task_row_locks.get(owner_id)
        if not held:
            return []
        selected = set(held) if resources is None else held.intersection(resources)
        if not selected:
            return []
        held.difference_update(selected)
        if not held:
            self._task_row_locks.pop(owner_id, None)
        released: list[tuple[str, int]] = []
        for res_id in selected:
            if self._active_row_locks.get(res_id) == owner_id:
                self._active_row_locks.pop(res_id, None)
            lid = self._row_lock_ids.get(res_id)
            if lid is not None:
                released.append((res_id, lid))
        # The sync scheduler emits DPOR events in this order, so canonicalize
        # the set iteration by the stable allocation ID.
        released.sort(key=lambda pair: pair[1])
        if graph is not None:
            for _res_id, lid in released:
                graph.remove_holding(owner_id, lid, kind="row_lock")
        return released

    def pop_all(self, owner_id: int, graph: Any) -> list[tuple[str, int]]:
        """Release all row locks held by *owner_id*."""
        return self.pop(owner_id, graph)
