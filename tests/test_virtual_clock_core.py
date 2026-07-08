"""Unit-level tests for virtual-clock core plumbing.

These exercise ``frontrun._virtual_clock`` and ``frontrun._cooperative``
internals directly (patch/unpatch bookkeeping, active-clock resolution, the
deadline coordinator's thread-safety) rather than going through a full
``frontrun.explore()`` run.
"""

from __future__ import annotations

import datetime as dt
import threading
import time

import pytest

from frontrun import _cooperative
from frontrun._virtual_clock import (
    VIRTUAL_EPOCH,
    DeadlineCoordinator,
    VirtualClock,
    _active_virtual_clock,
    _thread_clocks,
    clock_scope,
    patch_time,
    unpatch_time,
)


# ---------------------------------------------------------------------------
# Fix 1: patch/unpatch must not stomp a pre-existing third-party time patch.
# ---------------------------------------------------------------------------


def test_patch_time_preserves_preexisting_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A freezegun-style pre-existing ``time.*`` patch survives patch/unpatch.

    ``clock_scope`` (patch_time/unpatch_time) must save whatever is *currently*
    installed at the 0->1 transition and restore that at 1->0 — not the pristine
    C functions captured at import.  The gated fallback (no active clock) must
    also call the pre-existing patch, so a non-registered thread inside the
    scope keeps seeing the third-party fake.
    """
    sentinel = 424242.0

    def fake_time() -> float:
        return sentinel

    def fake_monotonic() -> float:
        return sentinel

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(time, "monotonic", fake_monotonic)

    other_thread_saw: dict[str, float] = {}
    clock = VirtualClock()
    with clock_scope(clock):
        # The registered (current) thread sees virtual time.
        assert time.time() == clock.now()
        assert time.monotonic() == clock.now()

        # A non-registered thread hits the gated fallback -> the pre-existing patch.
        def other() -> None:
            other_thread_saw["time"] = time.time()
            other_thread_saw["monotonic"] = time.monotonic()

        th = threading.Thread(target=other)
        th.start()
        th.join()

    # After the scope, the pre-existing patch is restored (not the C function).
    assert time.time is fake_time
    assert time.monotonic is fake_monotonic
    assert time.time() == sentinel
    assert other_thread_saw["time"] == sentinel
    assert other_thread_saw["monotonic"] == sentinel


# ---------------------------------------------------------------------------
# Fix 6: virtual datetime/date instances must have the real concrete type.
# ---------------------------------------------------------------------------


def test_virtual_datetime_instances_have_real_concrete_type() -> None:
    """``now()``/``utcnow()``/``today()`` return plain real datetime/date.

    Only the *class* on the ``datetime`` module is swapped during the patch
    window; instances must be exactly ``datetime.datetime`` / ``datetime.date``
    so values stored into user state keep working after unpatch (``type(x) is
    datetime.datetime``, pydantic strict, sqlite adapters).
    """
    real_datetime = dt.datetime
    real_date = dt.date
    clock = VirtualClock()
    with clock_scope(clock):
        now = dt.datetime.now()
        utc = dt.datetime.utcnow()
        aware = dt.datetime.now(dt.timezone.utc)
        today = dt.date.today()
        # Inside the scope the module attribute is the virtual class, but the
        # instances themselves must already be the real concrete type.
        assert type(now) is real_datetime
        assert type(utc) is real_datetime
        assert type(aware) is real_datetime
        assert type(today) is real_date
        # Values reflect virtual time.
        assert now.timestamp() == pytest.approx(VIRTUAL_EPOCH)

    # After the scope, the stashed instances are still the real type.
    assert type(now) is dt.datetime
    assert type(today) is dt.date
    assert aware.tzinfo is dt.timezone.utc


# ---------------------------------------------------------------------------
# Fix 3: an active scheduler context is authoritative for clock resolution.
# ---------------------------------------------------------------------------


def test_scheduler_context_with_none_clock_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scheduler ctx whose virtual_clock is None means *real* time.

    A ``clock="real"`` exploration nested inside an outer ``clock_scope`` /
    contextvar registration must not leak the outer virtual clock: when a
    scheduler context is present, its clock (even ``None``) wins over the TLS
    registration and the contextvar.
    """

    class FakeScheduler:
        virtual_clock = None

    sentinel = 4242.0
    monkeypatch.setattr(time, "time", lambda: sentinel)

    clock = VirtualClock()
    ident = threading.get_ident()
    _thread_clocks[ident] = clock  # simulate an outer clock_scope registration
    _cooperative.set_context(FakeScheduler(), 1)
    patch_time()
    try:
        # The scheduler ctx with virtual_clock=None wins -> no active clock.
        assert _active_virtual_clock() is None
        # ... so the patched time.time() falls through to the (saved) real fn,
        # not the virtual epoch.
        assert time.time() == sentinel
    finally:
        unpatch_time()
        _cooperative.clear_context()
        _thread_clocks.pop(ident, None)


# ---------------------------------------------------------------------------
# Fix 5: time.sleep under an active clock (no scheduler ctx) advances the clock.
# ---------------------------------------------------------------------------


def test_cooperative_sleep_advances_active_clock_without_scheduler_ctx() -> None:
    """Sleeping during setup()/invariant (no worker turn) must age the clock.

    Under ``clock_scope`` the driver thread has an active virtual clock but no
    cooperative scheduler context.  ``time.sleep(n)`` must advance the clock by
    ``n`` virtual seconds (deterministic — the driver is the only clock user at
    that moment), not silently no-op.
    """
    _cooperative.clear_context()
    assert _cooperative.get_context() is None
    clock = VirtualClock()
    with clock_scope(clock):
        before = time.monotonic()
        _cooperative._cooperative_sleep(5.0)
        after = time.monotonic()
    assert after - before == pytest.approx(5.0)
    assert clock.now() == pytest.approx(VIRTUAL_EPOCH + 5.0)
