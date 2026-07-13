"""Unit-level tests for virtual-clock core plumbing.

These exercise ``frontrun._virtual_clock`` and ``frontrun._cooperative``
internals directly (patch/unpatch bookkeeping, active-clock resolution, the
deadline coordinator's thread-safety) rather than going through a full
``frontrun.explore()`` run.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import math
import threading
import time
import warnings
from typing import Any

import pytest

import frontrun
from frontrun import _async_virtual_timeouts as async_virtual_timeouts
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
# Third-party time patch preservation.
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

    assert time.time is fake_time
    assert time.monotonic is fake_monotonic
    assert time.time() == sentinel
    assert other_thread_saw["time"] == sentinel
    assert other_thread_saw["monotonic"] == sentinel


def test_async_sleep_patch_is_reference_counted() -> None:
    """One overlapping owner must not unpatch ``asyncio.sleep`` for another."""
    async_virtual_timeouts._patch_asyncio_sleep()
    async_virtual_timeouts._patch_asyncio_sleep()
    try:
        async_virtual_timeouts._unpatch_asyncio_sleep()
        assert asyncio.sleep is async_virtual_timeouts._cooperative_async_sleep
    finally:
        async_virtual_timeouts._unpatch_asyncio_sleep()


def test_async_timeout_patch_is_reference_counted() -> None:
    """The virtual timeout shims must survive teardown of one overlapping scope."""
    async_virtual_timeouts._patch_asyncio_timeouts()
    async_virtual_timeouts._patch_asyncio_timeouts()
    try:
        async_virtual_timeouts._unpatch_asyncio_timeouts()
        assert asyncio.wait_for is async_virtual_timeouts._virtual_asyncio_wait_for
    finally:
        async_virtual_timeouts._unpatch_asyncio_timeouts()


def test_async_sleep_patch_preserves_preexisting_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An outer asyncio instrumentation patch must survive frontrun's scope."""

    async def fake_sleep(delay: float, result: Any = None) -> Any:  # noqa: ANN401
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    async_virtual_timeouts._patch_asyncio_sleep()
    try:
        assert asyncio.sleep is async_virtual_timeouts._cooperative_async_sleep
    finally:
        async_virtual_timeouts._unpatch_asyncio_sleep()
    assert asyncio.sleep is fake_sleep


def test_async_timeout_patch_preserves_preexisting_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An outer ``asyncio.wait_for`` patch must be restored after exploration."""

    async def fake_wait_for(awaitable: Any, timeout: float | None) -> Any:  # noqa: ANN401
        return await awaitable

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    async_virtual_timeouts._patch_asyncio_timeouts()
    try:
        assert asyncio.wait_for is async_virtual_timeouts._virtual_asyncio_wait_for
    finally:
        async_virtual_timeouts._unpatch_asyncio_timeouts()
    assert asyncio.wait_for is fake_wait_for


def test_unmanaged_sync_sleep_keeps_its_real_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-wide shim must delegate sleeps from unrelated threads."""
    observed: list[float] = []
    monkeypatch.setattr(_cooperative, "_real_time_sleep", observed.append)

    _cooperative._cooperative_sleep(0.25)

    assert observed == [0.25]


def test_sync_sleep_patch_preserves_preexisting_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An outer time.sleep instrumentation patch must survive exploration."""
    observed: list[float] = []

    def fake_sleep(delay: float) -> None:
        observed.append(delay)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    _cooperative.patch_sleep()
    try:
        assert time.sleep is _cooperative._cooperative_sleep
    finally:
        _cooperative.unpatch_sleep()

    assert time.sleep is fake_sleep


def test_active_sync_sleep_patch_preserves_unrelated_thread_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ownership must not collapse a background thread's sleep."""
    observed: list[float] = []

    def fake_sleep(delay: float) -> None:
        observed.append(delay)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    _cooperative.patch_sleep()
    try:
        thread = threading.Thread(target=lambda: time.sleep(0.25))
        thread.start()
        thread.join()
    finally:
        _cooperative.unpatch_sleep()

    assert observed == [0.25]


def test_unmanaged_async_sleep_keeps_its_real_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-wide shim must delegate sleeps from unrelated async tasks."""
    observed: list[float] = []

    async def fake_sleep(delay: float, result: Any = None) -> Any:  # noqa: ANN401
        observed.append(delay)
        return result

    monkeypatch.setattr(async_virtual_timeouts, "_real_asyncio_sleep", fake_sleep)
    result = asyncio.run(async_virtual_timeouts._cooperative_async_sleep(0.25, "sentinel"))

    assert result == "sentinel"
    assert observed == [0.25]


@pytest.mark.parametrize(
    ("delay", "error"),
    [
        (-1.0, ValueError),
        (math.nan, ValueError),
        (math.inf, OverflowError),
    ],
)
def test_virtual_time_sleep_rejects_invalid_delay(
    monkeypatch: pytest.MonkeyPatch, delay: float, error: type[Exception]
) -> None:
    """Virtual ``time.sleep`` preserves the stdlib's invalid-delay contract."""
    monkeypatch.setattr(_cooperative, "_active_virtual_clock", lambda: VirtualClock())

    with pytest.raises(error):
        _cooperative._cooperative_sleep(delay)


# ---------------------------------------------------------------------------
# Concrete datetime/date values.
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


def test_virtual_datetime_direct_constructors_return_real_concrete_types() -> None:
    real_datetime = dt.datetime
    real_date = dt.date

    with clock_scope(VirtualClock()):
        timestamp = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
        day = dt.date(2026, 1, 2)

    assert type(timestamp) is real_datetime
    assert type(day) is real_date
    assert timestamp == real_datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    assert day == real_date(2026, 1, 2)


# ---------------------------------------------------------------------------
# User subclasses of the patched datetime/date classes.
# ---------------------------------------------------------------------------


def test_datetime_subclass_defined_inside_scope_behaves_like_stdlib_subclass() -> None:
    """A ``datetime.datetime`` subclass defined inside the patch scope must work.

    Real libraries subclass datetime (e.g. ``pandas.Timestamp``); if such a
    library is imported lazily inside a clock scope its subclass bases the
    virtual class.  Construction must preserve the subclass (not silently
    return a plain real datetime), and isinstance/issubclass against the
    subclass must stay exact — only the virtual base class itself is
    transparent to the real type.
    """
    with clock_scope(VirtualClock()):

        class MyDT(dt.datetime):
            def tag(self) -> str:
                return f"tagged-{self.year}"

        value = MyDT(2020, 1, 1)
        assert type(value) is MyDT
        assert value.tag() == "tagged-2020"
        assert (value.year, value.month, value.day) == (2020, 1, 1)

        # Keyword construction goes through the same __new__ path.
        kw = MyDT(year=2021, month=2, day=3)
        assert type(kw) is MyDT
        assert kw.tag() == "tagged-2021"

        # isinstance/issubclass against the *subclass* must be exact: a plain
        # datetime is not a MyDT, and the patched base is not a MyDT subclass.
        assert not isinstance(dt.datetime(1999, 1, 1), MyDT)
        assert not issubclass(dt.datetime, MyDT)
        # ...but subclass instances are instances of the (patched) base.
        assert isinstance(value, dt.datetime)
        assert issubclass(MyDT, dt.datetime)

        # copy/deepcopy round-trip via __reduce_ex__ (the pickle path) must
        # preserve the subclass, like stdlib datetime subclasses do.
        assert type(copy.copy(value)) is MyDT
        deep = copy.deepcopy(value)
        assert type(deep) is MyDT
        assert deep == value
        # The pickle bytes-state constructor form directly.
        reduce_cls, reduce_args = value.__reduce__()
        rebuilt = reduce_cls(*reduce_args)
        assert type(rebuilt) is MyDT
        assert rebuilt == value

    # After the scope unwinds: instances remain valid, comparable, and exact.
    assert value.tag() == "tagged-2020"
    assert isinstance(value, MyDT)
    assert isinstance(value, dt.datetime)
    assert not isinstance(dt.datetime(1999, 1, 1), MyDT)
    assert value == dt.datetime(2020, 1, 1)


def test_date_subclass_defined_inside_scope_behaves_like_stdlib_subclass() -> None:
    with clock_scope(VirtualClock()):

        class MyDate(dt.date):
            def tag(self) -> str:
                return f"tagged-{self.year}"

        value = MyDate(2020, 1, 1)
        assert type(value) is MyDate
        assert value.tag() == "tagged-2020"
        assert (value.year, value.month, value.day) == (2020, 1, 1)

        kw = MyDate(year=2021, month=2, day=3)
        assert type(kw) is MyDate

        assert not isinstance(dt.date(1999, 1, 1), MyDate)
        assert not issubclass(dt.date, MyDate)
        assert isinstance(value, dt.date)
        assert issubclass(MyDate, dt.date)

        assert type(copy.copy(value)) is MyDate
        assert type(copy.deepcopy(value)) is MyDate

    assert value.tag() == "tagged-2020"
    assert isinstance(value, MyDate)
    assert value == dt.date(2020, 1, 1)


def test_datetime_subclass_classmethods_return_subclass_and_virtual_time() -> None:
    """``MyDT.now()`` etc. must return the subclass *and* virtual time.

    Matches stdlib semantics: alternate constructors called on a subclass
    return the subclass.  Under an active virtual clock they must still read
    virtual (not wall) time.
    """
    clock = VirtualClock()
    with clock_scope(clock):

        class MyDT(dt.datetime):
            pass

        now = MyDT.now()
        assert type(now) is MyDT
        assert now.timestamp() == pytest.approx(VIRTUAL_EPOCH)

        aware = MyDT.now(dt.timezone.utc)
        assert type(aware) is MyDT
        assert aware.timestamp() == pytest.approx(VIRTUAL_EPOCH)
        assert aware.tzinfo is dt.timezone.utc

        utc = MyDT.utcnow()
        assert type(utc) is MyDT
        assert utc.tzinfo is None

        ts = MyDT.fromtimestamp(123_456.0, dt.timezone.utc)
        assert type(ts) is MyDT
        assert ts.timestamp() == pytest.approx(123_456.0)
        assert type(MyDT.utcfromtimestamp(123_456.0)) is MyDT

        class MyDate(dt.date):
            pass

        today = MyDate.today()
        assert type(today) is MyDate
        assert today == dt.datetime.fromtimestamp(clock.now()).date()
        assert type(MyDate.fromtimestamp(123_456.0)) is MyDate


def test_plain_datetime_behavior_unchanged_by_subclass_support() -> None:
    """Regression guard: the base virtual classes keep their existing contract.

    Plain construction inside the scope still yields exactly the real concrete
    type (no virtual type leaking out of the scope), ``now()`` still reads
    virtual time, and instances created inside the scope stay valid and
    comparable after the scope exits.
    """
    real_datetime = dt.datetime
    real_date = dt.date
    clock = VirtualClock()
    with clock_scope(clock):
        made = dt.datetime(2020, 1, 1)
        day = dt.date(2020, 1, 1)
        now = dt.datetime.now()
        assert type(made) is real_datetime
        assert type(day) is real_date
        assert type(now) is real_datetime
        assert now.timestamp() == pytest.approx(VIRTUAL_EPOCH)
        # The transparent-base isinstance behavior is preserved.
        assert isinstance(made, dt.datetime)
        assert isinstance(day, dt.date)

    assert made == dt.datetime(2020, 1, 1)
    assert day == dt.date(2020, 1, 1)
    assert made < dt.datetime(2021, 1, 1)
    assert isinstance(made, dt.datetime)


# ---------------------------------------------------------------------------
# Active scheduler context precedence.
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
# Driver-thread sleep under an active clock.
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
# DeadlineCoordinator concurrency.
# ---------------------------------------------------------------------------


def test_deadline_coordinator_survives_concurrent_hammering() -> None:
    """Concurrent add / cancel / advance must not raise or corrupt state.

    Exploration mutators hold the scheduler engine lock while replay advance
    paths hold only the scheduler condition, so both can touch the coordinator's
    deadline dict at once.  The coordinator serializes its own
    mutation/iteration.
    """
    coord = DeadlineCoordinator()
    clock = VirtualClock()
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)
    iterations = 2_000

    def adder(offset: int) -> None:
        try:
            barrier.wait()
            for i in range(iterations):
                base = clock.now()
                coord.add_sleep((i + offset) % 16, base + (i % 7) + 0.001, None)
                coord.add_timeout(((i + offset) % 16) + 100, base + (i % 5) + 0.001, object())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def canceller() -> None:
        try:
            barrier.wait()
            for _ in range(iterations):
                for actor in range(16):
                    coord.cancel(actor)
                    coord.cancel_sleep(actor)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def advancer() -> None:
        try:
            barrier.wait()
            for _ in range(iterations):
                coord.next_deadline()
                coord.has_pending()
                coord.advance_to_next(clock)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=adder, args=(0,)),
        threading.Thread(target=adder, args=(8,)),
        threading.Thread(target=canceller),
        threading.Thread(target=advancer),
    ]
    for t in threads:
        t.start()
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
# Clock diagnostics.
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


def test_timed_acquire_state_refuses_wall_clock_fallback_under_virtual_clock() -> None:
    """A scheduler with a virtual clock but no ``add_timed_wait`` must fail loudly.

    ``_timed_acquire_state`` used to degrade to a *wall-clock* deadline when the
    active scheduler exposed ``virtual_clock`` but not ``add_timed_wait`` —
    silently making the timeout host-speed-dependent under a clock that
    promises determinism.  No shipped scheduler hits this (both implement
    ``add_timed_wait``), so the mismatch is an internal contract violation and
    must raise rather than quietly hand back real time.
    """

    class _ClockOnlyScheduler:
        virtual_clock = VirtualClock()
        # Deliberately no add_timed_wait.

    with pytest.raises(RuntimeError, match="add_timed_wait"):
        _cooperative._timed_acquire_state(1.0, _ClockOnlyScheduler(), thread_id=0)


def test_ns_time_functions_read_virtual_time() -> None:
    """The ``*_ns`` variants must return virtual time, not just be restored.

    Coverage previously only asserted that ``time_ns``/``monotonic_ns``/
    ``perf_counter_ns`` are saved and restored around exploration; nothing
    checked their *reads* reflect the virtual clock (the float variants were
    checked, the ``_ns`` ones were not).
    """
    clock = VirtualClock()
    with clock_scope(clock):
        expected = round(clock.now() * 1e9)
        assert time.time_ns() == expected
        assert time.monotonic_ns() == expected
        assert time.perf_counter_ns() == expected
        clock.advance_to(clock.now() + 1.5)
        expected = round(clock.now() * 1e9)
        assert time.time_ns() == expected
        assert time.monotonic_ns() == expected
        assert time.perf_counter_ns() == expected


def test_nested_clock_scope_restores_outer_virtual_clock() -> None:
    """An inner ``clock_scope`` must resolve to *its* clock and, on exit,
    restore the outer scope's clock (refcounted patch + contextvar stack),
    not real time and not the inner clock."""
    outer = VirtualClock(epoch=1_000_000.0)
    inner = VirtualClock(epoch=2_000_000.0)
    real_before = vc.real_monotonic()
    with clock_scope(outer):
        assert time.monotonic() == outer.now()
        with clock_scope(inner):
            assert time.monotonic() == inner.now()
            inner.advance_to(inner.now() + 5.0)
            assert time.monotonic() == inner.now()
        # Inner exit: the outer virtual clock is active again (the patch must
        # not have been torn down by the inner unpatch).
        assert time.monotonic() == outer.now()
    # Full exit: real time restored.
    assert abs(time.monotonic() - real_before) < 60.0
