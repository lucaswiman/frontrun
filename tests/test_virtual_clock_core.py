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
