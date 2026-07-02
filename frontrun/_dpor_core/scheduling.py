"""Shared scheduling-decision helpers for the sync and async DPOR schedulers.

Pure functions lifted from the duplicated tail of ``_schedule_next`` in
``DporScheduler`` (sync) and ``AsyncDporScheduler`` (async). They operate only
on plain dicts/sets — no threading or asyncio — so both backends call them
identically; the schedulers differ only in the block/wake primitive that lives
*around* these decisions.
"""

from __future__ import annotations


def apply_lock_blocked_override(
    scheduled: int | None,
    blocked: dict[int, int],
    done: set[int],
) -> int | None:
    """Redirect the engine's choice when it picks a lock-blocked worker.

    When the DPOR engine selects ``scheduled`` but that worker is blocked
    waiting on a lock held by another worker, run the holder instead so it can
    make progress and release the lock — otherwise the scheduler cycles between
    the blocked worker and its holder, manufacturing a false deadlock timeout.
    If the holder has already finished, the mapping is stale: drop it and let
    the engine's original choice proceed.

    Args:
        scheduled: The worker id the engine chose (``None`` if nothing runnable).
        blocked: Maps a blocked worker id to the id of the worker holding the
            lock it waits on. Mutated in place to drop stale holder entries.
        done: The set of finished worker ids.

    Returns:
        The worker id to actually run (the holder, the original choice, or
        ``None``).
    """
    if scheduled is not None and scheduled in blocked:
        holder = blocked[scheduled]
        if holder not in done:
            return holder
        # Holder is done — the lock should already be released. Drop the stale
        # entry so the originally-scheduled worker can proceed.
        blocked.pop(scheduled, None)
    return scheduled
