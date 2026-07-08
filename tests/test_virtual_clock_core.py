"""Unit-level tests for virtual-clock core plumbing.

These exercise ``frontrun._virtual_clock`` and ``frontrun._cooperative``
internals directly (patch/unpatch bookkeeping, active-clock resolution, the
deadline coordinator's thread-safety) rather than going through a full
``frontrun.explore()`` run.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import threading
import time
import warnings
from typing import Any

import pytest

import frontrun
from frontrun import _cooperative
from frontrun import _virtual_clock as vc
from frontrun._virtual_clock import (
    VIRTUAL_EPOCH,
    DeadlineCoordinator,
    VirtualClock,
    _active_virtual_clock,
    _clock_var,
    clock_scope,
    patch_time,
    unpatch_time,
    validate_clock_options,
    warn_if_captured_time_reference,
)


def _reset_diag_caches() -> None:
    vc._warned_captured_refs.clear()
    scanned = getattr(vc, "_scanned_code_objects", None)
    if scanned is not None:
        scanned.clear()


class _FakeFrame:
    """Minimal stand-in for a Python frame for the diagnostic scanner."""

    def __init__(self, code: Any, locals_: dict[str, Any], globals_: dict[str, Any], lineno: int) -> None:
        self.f_code = code
        self.f_locals = locals_
        self.f_globals = globals_
        self.f_lineno = lineno


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


def test_virtual_datetime_fromtimestamp_is_input_deterministic_and_real_type() -> None:
    real_datetime = dt.datetime
    real_date = dt.date
    timestamp = 123_456.0

    with clock_scope(VirtualClock()):
        aware = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)
        naive = dt.datetime.fromtimestamp(timestamp)
        today = dt.date.fromtimestamp(timestamp)

    assert type(aware) is real_datetime
    assert type(naive) is real_datetime
    assert type(today) is real_date
    assert aware.timestamp() == pytest.approx(timestamp)
    assert naive.timestamp() == pytest.approx(timestamp)
    assert today == real_datetime.fromtimestamp(timestamp).date()


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
    token = _clock_var.set((threading.get_ident(), clock))  # simulate an outer clock_scope registration
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
        _clock_var.reset(token)


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


# ---------------------------------------------------------------------------
# Fix 2: DeadlineCoordinator must be safe under concurrent mutation/iteration.
# ---------------------------------------------------------------------------


def test_deadline_coordinator_survives_concurrent_hammering() -> None:
    """Concurrent add / cancel / advance must not raise or corrupt state.

    Exploration mutators hold the scheduler engine lock while replay advance
    paths hold only the scheduler condition, so both can touch the coordinator's
    deadline dict at once.  Without an internal lock, an insert during an
    ``advance_to`` iteration raises ``RuntimeError: dictionary changed size
    during iteration`` (near-certain on free-threaded builds) and torpedoes the
    reproduction.  The coordinator must serialise its own mutation/iteration.
    """
    coord = DeadlineCoordinator()
    clock = VirtualClock()
    errors: list[BaseException] = []
    stop = threading.Event()

    def adder() -> None:
        i = 0
        try:
            while not stop.is_set():
                base = clock.now()
                coord.add_sleep(i % 16, base + (i % 7) + 0.001, None)
                coord.add_timeout((i % 16) + 100, base + (i % 5) + 0.001, object())
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def canceller() -> None:
        try:
            while not stop.is_set():
                for actor in range(16):
                    coord.cancel(actor)
                    coord.cancel_sleep(actor)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def advancer() -> None:
        try:
            while not stop.is_set():
                coord.next_deadline()
                coord.has_pending()
                coord.advance_to_next(clock)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=adder),
        threading.Thread(target=adder),
        threading.Thread(target=canceller),
        threading.Thread(target=advancer),
    ]
    for t in threads:
        t.start()
    time.sleep(1.5)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"coordinator raised under concurrency: {errors!r}"
    # Clock only ever moves forward.
    assert clock.now() >= VIRTUAL_EPOCH


def test_deadline_coordinator_advance_to_next_selects_and_advances_atomically() -> None:
    class InsertingCoordinator(DeadlineCoordinator):
        def __init__(self) -> None:
            super().__init__()
            self.inserted = False

        def next_deadline(self) -> float | None:
            deadline = super().next_deadline()
            if not self.inserted:
                self.inserted = True
                self.add_sleep(2, VIRTUAL_EPOCH + 1.0, None, token=object())
            return deadline

    coord = InsertingCoordinator()
    clock = VirtualClock()
    coord.add_sleep(1, VIRTUAL_EPOCH + 10.0, None, token=object())

    due = coord.advance_to_next(clock)

    assert [event.actor_id for event in due] == [1]
    assert clock.now() == pytest.approx(VIRTUAL_EPOCH + 10.0)


# ---------------------------------------------------------------------------
# Fix 4: make clock_diagnostics usable.
# ---------------------------------------------------------------------------


def _probe() -> None:  # a stable, distinct code object for scanner tests
    pass


def test_clock_diagnostics_warns_once_per_capture_across_lines() -> None:
    """(a)/(b) The dedup key drops f_lineno and each code object scans once.

    Two frames sharing one code object but differing in f_lineno (as a per-line
    tracer would supply) must yield exactly one warning, not one per line.
    """
    _reset_diag_caches()
    captured = time.monotonic
    code = _probe.__code__
    globs: dict[str, Any] = {}
    frame_a = _FakeFrame(code, {"cap": captured}, globs, 10)
    frame_b = _FakeFrame(code, {"cap": captured}, globs, 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_captured_time_reference(frame_a)
        warn_if_captured_time_reference(frame_b)
    captured_warnings = [w for w in caught if "captured" in str(w.message)]
    assert len(captured_warnings) == 1


def test_clock_diagnostics_scans_each_code_object_at_most_once() -> None:
    """(b) A cache hit early-returns before touching frame locals/globals."""
    _reset_diag_caches()

    class CountingDict(dict[str, Any]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.items_calls = 0

        def items(self):  # type: ignore[override]
            self.items_calls += 1
            return super().items()

    captured = time.monotonic
    code = _probe.__code__
    locals_a = CountingDict({"cap": captured})
    globals_a = CountingDict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warn_if_captured_time_reference(_FakeFrame(code, locals_a, globals_a, 1))
        # Second frame, same code object: must early-return without scanning.
        locals_b = CountingDict({"cap": captured})
        globals_b = CountingDict()
        warn_if_captured_time_reference(_FakeFrame(code, locals_b, globals_b, 2))
    assert locals_a.items_calls == 1
    assert globals_a.items_calls == 1
    assert locals_b.items_calls == 0
    assert globals_b.items_calls == 0


def test_clock_diagnostics_survives_racing_dict_mutation() -> None:
    """(c) A RuntimeError from concurrent global stores is retried, not raised."""
    _reset_diag_caches()

    class FlakyDict(dict[str, Any]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0

        def items(self):  # type: ignore[override]
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("dictionary changed size during iteration")
            return super().items()

    captured = time.monotonic
    code = _probe.__code__
    flaky_globals = FlakyDict({"cap": captured})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Must not raise despite the first items() blowing up.
        warn_if_captured_time_reference(_FakeFrame(code, {}, flaky_globals, 1))
    assert flaky_globals.calls == 2  # retried once
    captured_warnings = [w for w in caught if "captured" in str(w.message)]
    assert len(captured_warnings) == 1


def test_validate_rejects_clock_diagnostics_with_real_clock() -> None:
    """(d) clock_diagnostics=True with clock='real' is a ValueError, centrally."""
    with pytest.raises(ValueError, match="clock_diagnostics"):
        validate_clock_options("real", clock_diagnostics=True)


def test_explore_rejects_clock_diagnostics_with_real_clock() -> None:
    """(d) The public explore() surface rejects diagnostics under clock='real'."""

    class State:
        pass

    with pytest.raises(ValueError, match="clock_diagnostics"):
        frontrun.explore(
            setup=State,
            workers=[lambda s: None],
            invariant=lambda s: True,
            clock="real",
            clock_diagnostics=True,
        )


def test_async_random_warns_clock_diagnostics_unsupported() -> None:
    """(e) The async random strategy warns that clock_diagnostics is ignored."""

    class State:
        def __init__(self) -> None:
            self.value = 0

    async def worker(s: State) -> None:
        s.value += 1

    with pytest.warns(RuntimeWarning, match="clock_diagnostics is not supported"):
        asyncio.run(
            frontrun.explore(
                setup=State,
                workers=[worker],
                invariant=lambda s: True,
                strategy="random",
                clock="virtual",
                clock_diagnostics=True,
                max_attempts=2,
                reproduce_on_failure=0,
            )
        )


# ---------------------------------------------------------------------------
# Threads spawned inside a clock scope are not part of the exploration.
# ---------------------------------------------------------------------------


def test_thread_spawned_inside_clock_scope_sees_real_time() -> None:
    """A thread spawned inside ``clock_scope`` must see real time.

    Regression (3.14t): the free-threaded build starts ``threading.Thread``
    with a copy of the caller's contextvars context (GIL builds start threads
    with an empty context), so a bare contextvar registration leaked the
    driver's virtual clock into threads spawned inside the scope.  The
    registration is ident-gated: only the registering thread resolves it.
    """
    real_before = time.monotonic()
    clock = VirtualClock()
    seen: dict[str, float] = {}
    with clock_scope(clock):
        assert time.monotonic() == clock.now()

        def other() -> None:
            seen["monotonic"] = time.monotonic()

        th = threading.Thread(target=other)
        th.start()
        th.join()
    # The helper saw real monotonic time (close to our pre-scope sample), not
    # the virtual epoch.
    assert abs(seen["monotonic"] - real_before) < 60.0
