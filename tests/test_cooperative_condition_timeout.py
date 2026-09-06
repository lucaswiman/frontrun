"""Tests for CooperativeCondition ticket bookkeeping under timeout/abort.

Finding 1: A waiter takes a ticket before blocking.  On timeout (or
SchedulerAbort), the ticket is abandoned.  Because ``notify(n)`` computes the
number of un-served tickets as ``_next_ticket - _served``, a dead ticket
permanently absorbs one future notification, so a later genuine waiter can
hang (lost wakeup).

Finding 9c: ``CooperativeCondition.wait`` must fully release a reentrant lock
(all recursion levels) and restore the count afterwards, matching
``threading.Condition._release_save``.
"""

import pytest

from frontrun._cooperative import (
    CooperativeCondition,
    CooperativeLock,
    CooperativeRLock,
)


@pytest.mark.parametrize("expected_served", [[True], [True, False]])
def test_cancelled_ticket_does_not_absorb_notification(expected_served: list[bool]) -> None:
    """notify(1) skips a cancelled ticket and serves exactly one live waiter."""
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    with lock:
        cond._next_ticket = 1
        cond._cancel_ticket(0)
        cond._next_ticket += len(expected_served)
        cond.notify(1)

    assert [cond._ticket_served(ticket) for ticket in range(1, cond._next_ticket)] == expected_served


def test_rlock_wait_fully_releases_and_restores():
    """wait() on a CooperativeRLock acquired twice must release ALL levels.

    Real ``threading.Condition`` calls ``lock._release_save()`` which fully
    releases the reentrant lock so a notifier on another thread can acquire it.
    Releasing only one recursion level leaves count >= 1, so the notifier can
    never acquire and the program stalls.
    """
    observed_counts: list[int] = []

    rlock = CooperativeRLock()
    cond = CooperativeCondition(rlock)

    # Spy on the real fallback condition: when wait() blocks there, the
    # reentrant lock must have been fully released (count == 0).
    class _SpyCond:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def wait(self, timeout=None):
            observed_counts.append(rlock._count)
            return False

    cond._real_cond = _SpyCond()

    rlock.acquire()
    rlock.acquire()
    assert rlock._count == 2

    # No scheduler context -> wait() falls into the (spied) real-condition path.
    cond.wait(timeout=0.01)

    assert observed_counts == [0], (
        f"reentrant lock not fully released during wait(): count seen by "
        f"notifier path was {observed_counts}, expected [0]"
    )
    assert rlock._count == 2, f"recursion count not restored after wait(): {rlock._count}"
    assert rlock._is_owned(), "lock ownership not restored after wait()"
    rlock.release()
    rlock.release()
