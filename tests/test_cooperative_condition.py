"""Cooperative condition notification, cancellation, and wait-for contracts."""

import threading
import time

import pytest

from frontrun._cooperative import CooperativeCondition, CooperativeLock, CooperativeRLock


@pytest.mark.parametrize(
    ("waiters", "notifications", "served"),
    [
        pytest.param(3, [1], [1], id="notify-one"),
        pytest.param(5, [2], [2], id="notify-two"),
        pytest.param(4, [None], [4], id="notify-all"),
        pytest.param(0, [None], [0], id="notify-all-empty"),
        pytest.param(0, [1], [0], id="notify-empty"),
        pytest.param(2, [5], [2], id="cap-to-waiters"),
        pytest.param(3, [1, 1], [1, 2], id="accumulate"),
        pytest.param(3, [2, 2], [2, 3], id="cap-repeated-notify"),
        pytest.param(3, [1, None], [1, 3], id="notify-all-after-notify"),
    ],
)
def test_notifications_serve_only_existing_tickets(waiters, notifications, served):
    # Already-notified waiters remain counted until they reacquire the lock.
    cond = CooperativeCondition(CooperativeLock())
    with cond:
        cond._waiters = cond._next_ticket = waiters
        for count, expected in zip(notifications, served, strict=True):
            if count is None:
                cond.notify_all()
            else:
                cond.notify(count)
            assert cond._served == expected
            assert [cond._ticket_served(ticket) for ticket in range(waiters)] == [
                ticket < expected for ticket in range(waiters)
            ]


def test_notify_does_not_overwake_unmanaged_waiter():
    """notify(1) for a managed waiter must not also wake an unmanaged waiter.

    A managed waiter M holds the lowest ticket and spins on the ticket
    system; an unmanaged waiter U (no scheduler context) holds a higher
    ticket and blocks in the fallback ``real_cond.wait()``.  ``notify(1)``
    must wake exactly one waiter — M, the longest-waiting — and must NOT also
    wake U via ``real_cond``.  Waking both violates the "notify(n) wakes at
    most n waiters" contract and leaves U's ticket as an un-served zombie
    that later absorbs a genuine notification (a lost wakeup).
    """
    import threading
    import time

    cond = CooperativeCondition(CooperativeLock())

    # Simulate a managed waiter M holding ticket 0 (spinning on the ticket
    # system): advance _next_ticket past ticket 0 and record the waiter.
    with cond:
        cond._next_ticket = 1
        cond._waiters = 1

    # U: unmanaged (no scheduler context) — takes ticket 1 and blocks in the
    # fallback real_cond.wait().
    out: dict[str, bool] = {}

    def u() -> None:
        with cond:
            out["served"] = cond.wait(timeout=2.0)

    t = threading.Thread(target=u)
    t.start()
    time.sleep(0.3)  # let U enter real_cond.wait()

    with cond:
        cond.notify(1)  # meant for managed ticket 0 — must NOT wake U

    t.join(0.5)
    assert t.is_alive(), (
        "notify(1) for the managed waiter (ticket 0) spuriously woke the "
        "unmanaged waiter U (ticket 1) via real_cond — over-waking beyond n."
    )

    # Clean up: a second notify(1) serves U's ticket 1 and wakes it.
    with cond:
        cond.notify(1)
    t.join(2.5)
    assert not t.is_alive(), "U should wake once its own ticket is served."
    assert out.get("served") is True


def test_notify_caps_real_cond_to_actual():
    """notify(n) must cap real_cond.notify() to the served *unmanaged* tickets.

    Bug: CooperativeCondition.notify() passes the raw `n` (or the total served
    count) to self._real_cond.notify(), waking more non-cooperative threads
    than there are unmanaged waiters whose tickets were served.  Only tickets
    held by unmanaged (real_cond) waiters should drive real_cond.notify().
    """
    from unittest.mock import MagicMock

    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()

    # Two waiters: ticket 0 is unmanaged (blocked in real_cond), ticket 1 is a
    # managed spinner.  Only the unmanaged one should drive real_cond.notify().
    cond._waiters = 2
    cond._next_ticket = 2
    cond._real_cond_tickets = {0}

    mock_real_cond = MagicMock()
    mock_real_cond.__enter__ = MagicMock(return_value=mock_real_cond)
    mock_real_cond.__exit__ = MagicMock(return_value=False)

    cond._real_cond = mock_real_cond

    cond.notify(5)

    # Both tickets served, but only 1 belongs to an unmanaged waiter.
    mock_real_cond.notify.assert_called_once_with(1)

    lock.release()


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


def test_notify_wakes_unmanaged_waiter():
    cond = CooperativeCondition()
    ready = threading.Event()
    result = []

    def waiter():
        with cond:
            ready.set()
            result.append(cond.wait(timeout=3.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    try:
        assert ready.wait(timeout=1.0)
        with cond:
            cond.notify()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert result == [True]  # A timeout must not masquerade as notification.
    finally:
        with cond:
            cond.notify_all()
        thread.join(timeout=3.5)


def test_notify_without_lock_raises():
    with pytest.raises(RuntimeError, match="cannot notify on un-acquired lock"):
        CooperativeCondition().notify()


@pytest.mark.parametrize("ready", [True, False], ids=["already-ready", "timeout"])
def test_wait_for_predicate_and_timeout(ready):
    cond = CooperativeCondition()
    start = time.monotonic()
    with cond:
        assert cond.wait_for(lambda: ready, timeout=0.1) is ready
    if not ready:
        assert time.monotonic() - start >= 0.09


def test_wait_for_retries_after_spurious_wakeup(monkeypatch):
    # A first unsuccessful wake must not exhaust the whole timeout.
    cond = CooperativeCondition()
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        return True

    monkeypatch.setattr(cond, "wait", wait)
    with cond:
        assert cond.wait_for(lambda: len(waits) == 2, timeout=5.0)
    assert len(waits) == 2
