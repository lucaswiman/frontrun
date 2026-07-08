"""Virtual clock for exploring timeout / retry / TTL races.

Implements the design in ``ideas/virtual_clock.md``: during exploration the
scheduler owns a :class:`VirtualClock`; patched ``time.time`` /
``time.monotonic`` / ``time.perf_counter`` (and their ``_ns`` variants) return
virtual time for explored code, ``time.sleep`` / ``asyncio.sleep`` become
*timed blocks* (deadline registration + scheduler yield, zero wall time), and
the clock only advances when the scheduler decides it should:

- ``clock="virtual"`` — **autojump**: the clock jumps to the earliest pending
  deadline only when no worker is runnable (Trio's ``MockClock`` model).
- ``clock="explored"`` — the clock advance is a schedulable step of a
  synthetic DPOR *clock actor*, so the engine explores orderings of "timer
  fires" against workers' steps like any other interleaving choice.

Patching is gated per-thread/per-context so that non-explored threads (pytest
machinery, unrelated code) always see real time:

1. threads with a cooperative scheduler context in TLS whose scheduler holds a
   virtual clock (worker threads under sync exploration);
2. threads explicitly registered via :func:`clock_scope` (the exploration
   driver thread while running ``setup()`` / the invariant);
3. asyncio task contexts where :data:`_clock_var` is set (async exploration —
   contextvars propagate to tasks, but not to the event loop's own
   ``_run_once`` machinery, which keeps loop timers on wall-clock time).
"""

from __future__ import annotations

import datetime as _datetime
import threading
import time
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import CodeType
from typing import Any, Literal

from frontrun import _real_threading as _rt

ClockMode = Literal["real", "virtual", "explored"]

_CLOCK_MODES: tuple[str, ...] = ("real", "virtual", "explored")

#: Arbitrary virtual epoch (seconds).  Large enough to be visibly "not wall
#: clock" in traces, small enough that ``now * 1e9`` fits comfortably in an int.
VIRTUAL_EPOCH = 1_000_000.0

_real_time = time.time
_real_monotonic = time.monotonic
_real_perf_counter = time.perf_counter
_real_time_ns = time.time_ns
_real_monotonic_ns = time.monotonic_ns
_real_perf_counter_ns = time.perf_counter_ns
_real_datetime = _datetime.datetime
_real_date = _datetime.date
_REAL_TIME_FUNCTIONS = {
    _real_time: "time.time",
    _real_monotonic: "time.monotonic",
    _real_perf_counter: "time.perf_counter",
    _real_time_ns: "time.time_ns",
    _real_monotonic_ns: "time.monotonic_ns",
    _real_perf_counter_ns: "time.perf_counter_ns",
}
_warned_captured_refs: set[tuple[str, str, str, str]] = set()

# Values that were *installed* on the ``time`` / ``datetime`` modules when
# ``patch_time`` last transitioned 0->1.  Unlike the ``_real_*`` C functions
# above (captured once at import), these track whatever third-party patcher
# (freezegun, time-machine, ...) was already active when we patched.  The
# gated fallback (no active virtual clock) calls *these*, and ``unpatch_time``
# restores *these*, so an outer freeze survives a nested ``clock_scope``.
_saved_time = _real_time
_saved_monotonic = _real_monotonic
_saved_perf_counter = _real_perf_counter
_saved_time_ns = _real_time_ns
_saved_monotonic_ns = _real_monotonic_ns
_saved_perf_counter_ns = _real_perf_counter_ns
_saved_datetime: type[_datetime.datetime] = _real_datetime
_saved_date: type[_datetime.date] = _real_date


def real_monotonic() -> float:
    """The saved, never-patched ``time.monotonic`` (for scheduler machinery)."""
    return _real_monotonic()


def validate_clock(clock: str) -> ClockMode:
    """Validate a user-supplied ``clock=`` value, returning it typed."""
    if clock not in _CLOCK_MODES:
        raise ValueError(f"unknown clock={clock!r}; must be one of 'real', 'virtual', 'explored'")
    return clock  # type: ignore[return-value]


def validate_clock_options(
    clock: str,
    *,
    patch_sleep: bool = True,
    serializable_invariant: object = False,
    clock_diagnostics: bool = False,
) -> ClockMode:
    """Validate ``clock=`` against the options it constrains.

    Shared by every exploration entry point so the error text is identical
    everywhere.  Returns the typed mode.
    """
    mode = validate_clock(clock)
    if mode != "real":
        if not patch_sleep:
            raise ValueError("clock='virtual'/'explored' requires patch_sleep=True (sleeps become virtual deadlines)")
        if serializable_invariant is not False:
            raise ValueError(
                "clock='virtual'/'explored' cannot be combined with serializable_invariant: "
                "the sequential baseline runs execute outside the scheduler, so their sleeps "
                "and clock reads would use real wall-clock time"
            )
    elif clock_diagnostics:
        raise ValueError(
            "clock_diagnostics=True requires clock='virtual' or clock='explored' "
            "(there is no virtual clock to diagnose captured time references against under clock='real')"
        )
    return mode


#: Code objects already scanned by :func:`warn_if_captured_time_reference`.
#: Scanning is O(globals+locals), so we do it at most once per code object per
#: process.  ``id``-reuse across freed code objects is possible but harmless
#: here (code objects for explored workers stay alive for the run's duration).
_scanned_code_objects: set[CodeType] = set()


def warn_if_captured_time_reference(frame: Any) -> None:
    """Warn once when a frame holds a pre-patch real ``time.*`` function.

    This is an opt-in diagnostic called by tracers.  It intentionally does
    not rewrite the reference: function objects captured before frontrun's
    patch scope cannot be safely swapped out in general.

    Cost control: each code object is scanned at most once per process (the
    tracer fires per opcode, so re-scanning would be O(globals) per
    instruction).  The cache hit early-returns *before* touching the frame's
    locals/globals.  The dedup key omits the line number so a capture warns
    once, not once per executed line.
    """
    code = frame.f_code
    if code in _scanned_code_objects:
        return
    _scanned_code_objects.add(code)
    qualname = getattr(code, "co_qualname", code.co_name)
    for scope_name, mapping in (("local", frame.f_locals), ("global", frame.f_globals)):
        # Iterating a live f_globals can race concurrent module-global stores on
        # free-threaded builds; snapshot into a list, retrying once on the
        # transient "dictionary changed size during iteration".
        try:
            items = list(mapping.items())
        except RuntimeError:
            try:
                items = list(mapping.items())
            except RuntimeError:
                continue
        for name, value in items:
            label = next((candidate for func, candidate in _REAL_TIME_FUNCTIONS.items() if value is func), None)
            if label is None:
                continue
            key = (code.co_filename, qualname, name, label)
            if key in _warned_captured_refs:
                continue
            _warned_captured_refs.add(key)
            warnings.warn(
                f"virtual clock diagnostic: captured real {label} reference in {scope_name} {name!r}; "
                f"call through the time module inside explored code instead",
                RuntimeWarning,
                stacklevel=2,
            )


class VirtualClock:
    """Monotonic virtual clock owned by an exploration scheduler.

    Only the scheduler advances it (autojump on idle, or the explored clock
    actor's step).  Reads are side-effect-free and are deliberately *not*
    scheduling points (see the proposal's tractability trade-off).
    """

    __slots__ = ("_now", "_lock")

    def __init__(self, epoch: float = VIRTUAL_EPOCH) -> None:
        self._now = epoch
        # A *real* lock, never the cooperative replacement: advance_to() runs
        # inside scheduler machinery (often while holding the scheduler
        # condition), and a patched threading.Lock would report sync events
        # back into the scheduler — a reentrant self-deadlock.
        self._lock = _rt.lock()

    def now(self) -> float:
        return self._now

    def advance_to(self, deadline: float) -> None:
        """Jump the clock forward to *deadline* (never backwards)."""
        with self._lock:
            if deadline > self._now:
                self._now = deadline


@dataclass(frozen=True)
class WakeEvent:
    """A deadline that became due after a virtual-clock advance."""

    actor_id: int
    deadline: float
    token: object
    kind: str
    wake_id: int | None


@dataclass(frozen=True)
class _Deadline:
    actor_id: int
    deadline: float
    order: int
    token: object
    kind: str
    wake_id: int | None


_SLEEP_TOKEN = object()


class DeadlineCoordinator:
    """Own virtual deadline ordering for a scheduler.

    The scheduler still owns engine blocking/unblocking and synchronization
    reporting.  This helper only tracks deadlines, selects the next virtual
    time, and returns the events that became due.  It supports more than one
    deadline per actor, which is required for ``asyncio.wait_for`` wrapping an
    awaitable that also has its own virtual sleep deadline.

    Thread-safety: every method that mutates or iterates ``_deadlines`` (and
    the ``_next_order`` counter) is serialised on an internal *real* lock.
    Callers already hold a scheduler lock (``_engine_lock`` for exploration
    mutators, ``_condition`` for replay advance paths), but those are two
    different locks — so an insert on one and an ``advance_to`` iteration on the
    other could race the dict without this internal guard.  The lock is a real,
    never-cooperative lock (like :class:`VirtualClock._lock`) because these
    methods run inside scheduler machinery; a patched ``threading.Lock`` would
    report sync events back into the scheduler.  Nesting order is always
    scheduler-lock -> coordinator-lock -> ``clock._lock``; no method calls back
    into scheduler code while holding ``self._lock``, so there is no inversion.
    """

    def __init__(self) -> None:
        self._deadlines: dict[tuple[int, object], _Deadline] = {}
        self._next_order = 0
        self._lock = _rt.lock()

    def _new_deadline(self, actor_id: int, deadline: float, token: object, kind: str, wake_id: int | None) -> _Deadline:
        """Allocate a deadline with the next order token.  Caller holds ``_lock``."""
        self._next_order += 1
        return _Deadline(actor_id, deadline, self._next_order, token, kind, wake_id)

    def add_sleep(self, actor_id: int, deadline: float, wake_id: int | None, token: object = _SLEEP_TOKEN) -> None:
        with self._lock:
            self._deadlines[(actor_id, token)] = self._new_deadline(actor_id, deadline, token, "sleep", wake_id)

    def add_timeout(self, actor_id: int, deadline: float, token: object, wake_id: int | None = None) -> None:
        with self._lock:
            self._deadlines[(actor_id, token)] = self._new_deadline(actor_id, deadline, token, "timeout", wake_id)

    def cancel(self, actor_id: int, token: object | None = None) -> None:
        with self._lock:
            if token is not None:
                self._deadlines.pop((actor_id, token), None)
                return
            for key in [key for key in self._deadlines if key[0] == actor_id]:
                del self._deadlines[key]

    def cancel_sleep(self, actor_id: int) -> None:
        self.cancel(actor_id, _SLEEP_TOKEN)

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._deadlines)

    def next_deadline(self) -> float | None:
        with self._lock:
            if not self._deadlines:
                return None
            return min(entry.deadline for entry in self._deadlines.values())

    def advance_to_next(self, clock: VirtualClock) -> list[WakeEvent]:
        deadline = self.next_deadline()
        if deadline is None:
            return []
        return self.advance_to(clock, deadline)

    def advance_to(self, clock: VirtualClock, deadline: float) -> list[WakeEvent]:
        with self._lock:
            clock.advance_to(deadline)
            now = clock.now()
            due = [entry for entry in self._deadlines.values() if entry.deadline <= now]
            due.sort(key=lambda entry: (entry.deadline, entry.actor_id, entry.order))
            for entry in due:
                self._deadlines.pop((entry.actor_id, entry.token), None)
            return [
                WakeEvent(
                    actor_id=entry.actor_id,
                    deadline=entry.deadline,
                    token=entry.token,
                    kind=entry.kind,
                    wake_id=entry.wake_id,
                )
                for entry in due
            ]


class _VirtualDateTimeMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _real_datetime)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, _real_datetime)


class _VirtualDateMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _real_date)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, _real_date)


class _VirtualDateTime(_real_datetime, metaclass=_VirtualDateTimeMeta):
    # Only the *class* on the datetime module is virtual; the values these
    # constructors return are plain instances of the real datetime class saved
    # when we patched (Fix 6), so ``type(x) is datetime.datetime`` holds and the
    # objects survive unpatch (pydantic strict, sqlite adapters, ...).
    @classmethod
    def now(cls, tz: _datetime.tzinfo | None = None) -> _datetime.datetime:
        clock = _active_virtual_clock()
        real = _saved_datetime
        if clock is None:
            return real.now(tz)
        return real.fromtimestamp(clock.now(), tz)

    @classmethod
    def utcnow(cls) -> _datetime.datetime:
        clock = _active_virtual_clock()
        real = _saved_datetime
        if clock is None:
            return real.now(_datetime.timezone.utc).replace(tzinfo=None)
        return real.fromtimestamp(clock.now(), _datetime.timezone.utc).replace(tzinfo=None)


class _VirtualDate(_real_date, metaclass=_VirtualDateMeta):
    @classmethod
    def today(cls) -> _datetime.date:
        clock = _active_virtual_clock()
        if clock is None:
            return _saved_date.today()
        return _saved_datetime.fromtimestamp(clock.now()).date()


# ---------------------------------------------------------------------------
# Active-clock resolution
# ---------------------------------------------------------------------------

# Threads registered explicitly (exploration driver during setup()/invariant()).
_thread_clocks: dict[int, VirtualClock] = {}

#: Async exploration sets this contextvar around setup/tasks/invariant; task
#: contexts inherit it, the event loop's base context does not.
_clock_var: ContextVar[VirtualClock | None] = ContextVar("frontrun_virtual_clock", default=None)


# One-time lazy bind of _cooperative.get_context (function-level import would
# cost ~1µs per patched time.* call; module-level would be an import cycle).
_get_context: Callable[[], tuple[Any, int] | None] | None = None


def _active_virtual_clock() -> VirtualClock | None:
    """Return the virtual clock for the calling thread/context, if any."""
    global _get_context  # noqa: PLW0603
    if _get_context is None:
        from frontrun._cooperative import get_context as _get_context_impl

        _get_context = _get_context_impl

    ctx = _get_context()
    if ctx is not None:
        # An active scheduler context is authoritative: return its clock even
        # when None (a clock="real" exploration).  Falling through would let an
        # outer clock_scope / contextvar leak its virtual clock into a nested
        # real-clock exploration.
        return getattr(ctx[0], "virtual_clock", None)  # type: ignore[no-any-return]
    clock = _thread_clocks.get(threading.get_ident())
    if clock is not None:
        return clock
    return _clock_var.get()


@contextmanager
def clock_scope(clock: VirtualClock | None) -> Generator[None, None, None]:
    """Register *clock* for the current thread for the duration of the block.

    Used by exploration drivers around ``setup()`` and invariant evaluation so
    that state created / inspected on the driver thread sees the same virtual
    time as the workers.  A ``None`` clock makes this a no-op.

    Owns the ``time.*`` patch for its duration (reference-counted), so it works
    even outside a runner's patch scope — invariant evaluation happens after
    the workers' patch scope has already been unwound.
    """
    if clock is None:
        yield
        return
    ident = threading.get_ident()
    prev = _thread_clocks.get(ident)
    _thread_clocks[ident] = clock
    patch_time()
    try:
        yield
    finally:
        unpatch_time()
        if prev is None:
            _thread_clocks.pop(ident, None)
        else:
            _thread_clocks[ident] = prev


@contextmanager
def clock_context(clock: VirtualClock | None) -> Generator[None, None, None]:
    """Set :data:`_clock_var` for the current *context* (async exploration).

    asyncio tasks created inside the block inherit the contextvar (task
    contexts copy the creating context), while the event loop's own
    ``_run_once`` machinery — which runs in the loop's base context — keeps
    seeing real time, so loop timers stay on the wall clock.

    Like :func:`clock_scope`, owns the ``time.*`` patch for its duration.
    """
    if clock is None:
        yield
        return
    token = _clock_var.set(clock)
    patch_time()
    try:
        yield
    finally:
        unpatch_time()
        _clock_var.reset(token)


# ---------------------------------------------------------------------------
# time.* patching (reference-counted, like patch_sleep in _cooperative)
# ---------------------------------------------------------------------------


def _virtual_time() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _saved_time()


def _virtual_monotonic() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _saved_monotonic()


def _virtual_perf_counter() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _saved_perf_counter()


def _virtual_time_ns() -> int:
    clock = _active_virtual_clock()
    return round(clock.now() * 1e9) if clock is not None else _saved_time_ns()


def _virtual_monotonic_ns() -> int:
    clock = _active_virtual_clock()
    return round(clock.now() * 1e9) if clock is not None else _saved_monotonic_ns()


def _virtual_perf_counter_ns() -> int:
    clock = _active_virtual_clock()
    return round(clock.now() * 1e9) if clock is not None else _saved_perf_counter_ns()


_time_patch_count = 0
_time_patch_lock = _rt.lock()


def patch_time() -> None:
    """Route ``time.{time,monotonic,perf_counter}`` (+ ``_ns``) through the
    active virtual clock.  Reference-counted; unaffected threads see real time.
    """
    global _time_patch_count  # noqa: PLW0603
    global _saved_time, _saved_monotonic, _saved_perf_counter  # noqa: PLW0603
    global _saved_time_ns, _saved_monotonic_ns, _saved_perf_counter_ns  # noqa: PLW0603
    global _saved_datetime, _saved_date  # noqa: PLW0603
    with _time_patch_lock:
        _time_patch_count += 1
        if _time_patch_count > 1:
            return
        # Snapshot whatever is currently installed (possibly a third-party
        # fake) *before* overwriting, so the gated fallback and unpatch_time
        # honor it rather than the pristine import-time C functions.
        _saved_time = time.time
        _saved_monotonic = time.monotonic
        _saved_perf_counter = time.perf_counter
        _saved_time_ns = time.time_ns
        _saved_monotonic_ns = time.monotonic_ns
        _saved_perf_counter_ns = time.perf_counter_ns
        _saved_datetime = _datetime.datetime
        _saved_date = _datetime.date
        time.time = _virtual_time
        time.monotonic = _virtual_monotonic
        time.perf_counter = _virtual_perf_counter
        time.time_ns = _virtual_time_ns
        time.monotonic_ns = _virtual_monotonic_ns
        time.perf_counter_ns = _virtual_perf_counter_ns
        _datetime.datetime = _VirtualDateTime
        _datetime.date = _VirtualDate


def unpatch_time() -> None:
    """Restore the original ``time.*`` functions once all patchers release."""
    global _time_patch_count  # noqa: PLW0603
    with _time_patch_lock:
        if _time_patch_count <= 0:
            return
        _time_patch_count -= 1
        if _time_patch_count > 0:
            return
        # Restore whatever was installed when we patched (Fix 1): an outer
        # freezegun/time-machine patch must survive our scope.
        time.time = _saved_time
        time.monotonic = _saved_monotonic
        time.perf_counter = _saved_perf_counter
        time.time_ns = _saved_time_ns
        time.monotonic_ns = _saved_monotonic_ns
        time.perf_counter_ns = _saved_perf_counter_ns
        _datetime.datetime = _saved_datetime
        _datetime.date = _saved_date


__all__ = [
    "VIRTUAL_EPOCH",
    "ClockMode",
    "DeadlineCoordinator",
    "VirtualClock",
    "WakeEvent",
    "clock_context",
    "clock_scope",
    "patch_time",
    "real_monotonic",
    "unpatch_time",
    "validate_clock",
    "validate_clock_options",
    "warn_if_captured_time_reference",
]
