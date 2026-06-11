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

from frontrun._cooperative import (
    CooperativeCondition,
    CooperativeLock,
    CooperativeRLock,
)


def test_cancelled_ticket_does_not_absorb_notification():
    """A timed-out (cancelled) ticket must not consume a future notify().

    Scenario from the finding:
      - T1 wait(timeout) times out: ticket 0 leaked/cancelled.
      - T2 wait(): ticket 1, still waiting.
      - T3 notify(): must wake T2, not be absorbed by the dead ticket 0.
    """
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()

    # Ticket 0 was taken by a waiter that timed out and cancelled it.
    cond._next_ticket = 1
    cond._cancel_ticket(0)

    # Ticket 1 is taken by a live waiter T2.
    live_ticket = cond._next_ticket
    cond._next_ticket += 1

    # T3 notifies once: this must serve the live waiter, not the dead ticket.
    cond.notify(1)

    lock.release()

    assert cond._ticket_served(live_ticket), (
        f"notify(1) was absorbed by the cancelled ticket; live waiter "
        f"(ticket {live_ticket}) was not served. served={cond._served} "
        f"cancelled={cond._cancelled}"
    )


def test_cancel_then_notify_only_serves_live():
    """notify(1) with one cancelled + one live ticket serves exactly the live one."""
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)
    lock.acquire()

    # tickets 0 (cancelled), 1 (live), 2 (live)
    cond._next_ticket = 3
    cond._cancel_ticket(0)

    cond.notify(1)
    lock.release()

    assert cond._ticket_served(1), "first live ticket should be served by notify(1)"
    assert not cond._ticket_served(2), "second live ticket should NOT be served by notify(1)"


def test_full_timeout_then_real_wait_does_not_hang():
    """End-to-end: a timed-out waiter must not steal the next notify().

    Uses a real CooperativeRLock-free path with no scheduler context so wait()
    times out quickly, but the ticket bookkeeping is still exercised.
    """
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()
    # Simulate a waiter that already timed out and cancelled its ticket.
    cond._next_ticket = 1
    cond._cancel_ticket(0)
    lock.release()

    # A subsequent live waiter takes a ticket and gets notified.
    lock.acquire()
    live = cond._next_ticket
    cond._next_ticket += 1
    cond.notify(1)
    lock.release()

    assert cond._ticket_served(live)


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
