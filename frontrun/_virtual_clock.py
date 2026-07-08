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

# Objects that were *installed* on the ``time`` / ``datetime`` modules when
# ``patch_time`` last transitioned 0->1, keyed by attribute name.  Unlike the
# ``_real_*`` C functions above (captured once at import), these track whatever
# third-party patcher (freezegun, time-machine, ...) was already active when we
# patched.  The gated fallback (no active virtual clock) and the datetime shims
# call *these*, and ``unpatch_time`` restores *these*, so an outer freeze
# survives a nested ``clock_scope``.  See ``_PATCHES`` for the driving table.
_saved_originals: dict[str, Any] = {
    "time": _real_time,
    "monotonic": _real_monotonic,
    "perf_counter": _real_perf_counter,
    "time_ns": _real_time_ns,
    "monotonic_ns": _real_monotonic_ns,
    "perf_counter_ns": _real_perf_counter_ns,
    "datetime": _real_datetime,
    "date": _real_date,
}


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
class ClockConfig:
    """The static clock options plus the per-execution derivations built on them.

    Every exploration entry point stamped the same trio: derive a fresh
    per-execution :class:`VirtualClock` when the mode is not ``"real"``, derive
    the synthetic clock-actor id (``== num_actors``), and validate the clock
    against the options it constrains.  ``ClockConfig`` centralizes those so the
    entry points construct one and read the derivations off it.  It intentionally
    holds only the *static* options (``mode`` / ``diagnostics``); the mutable,
    per-execution clock comes from :meth:`new_clock`.
    """

    mode: ClockMode = "real"
    diagnostics: bool = False

    def validate(self, *, patch_sleep: bool = True, serializable_invariant: object = False) -> ClockConfig:
        """Validate ``mode`` against the options it constrains; return self.

        Wraps :func:`validate_clock_options` so the error text is identical
        everywhere.  Raises ``ValueError`` on an invalid or conflicting mode.
        """
        validate_clock_options(
            self.mode,
            patch_sleep=patch_sleep,
            serializable_invariant=serializable_invariant,
            clock_diagnostics=self.diagnostics,
        )
        return self

    @property
    def active(self) -> bool:
        """Whether a virtual clock is in effect (``mode != "real"``)."""
        return self.mode != "real"

    def new_clock(self) -> VirtualClock | None:
        """A fresh per-execution :class:`VirtualClock`, or ``None`` under ``"real"``."""
        return VirtualClock() if self.active else None

    def actor_id(self, num_actors: int) -> int | None:
        """The synthetic clock-actor id (``== num_actors``), or ``None`` if inactive."""
        return num_actors if self.active else None


@dataclass(frozen=True)
class WakeEvent:
    """A registered virtual deadline; also the record returned once it is due.

    :class:`DeadlineCoordinator` stores these and, on advance, returns the ones
    that became due directly (no field-by-field copy).  ``order`` is a
    per-coordinator monotonic tiebreaker that makes the due ordering
    deterministic when several deadlines share a virtual time; it is internal
    bookkeeping that consumers of the wake event ignore.
    """

    actor_id: int
    deadline: float
    token: object
    kind: str
    wake_id: int | None
    order: int = 0


_SLEEP_TOKEN = object()

#: Token identifying an actor's single timed-lock-acquire deadline (kind
#: ``"timeout"``).  Shared by the sync DPOR and random schedulers so their
#: ``add_timed_wait`` / ``remove_timed_wait`` / ``give_up_timed_wait`` all key
#: the same coordinator entry.  (The async side keeps its own constant until
#: the later sync/async unification wave dedups it.)
_TIMED_WAIT_TOKEN = "timed_wait"


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
        self._deadlines: dict[tuple[int, object], WakeEvent] = {}
        self._next_order = 0
        self._lock = _rt.lock()

    def _new_deadline(self, actor_id: int, deadline: float, token: object, kind: str, wake_id: int | None) -> WakeEvent:
        """Allocate a deadline with the next order token.  Caller holds ``_lock``."""
        self._next_order += 1
        return WakeEvent(
            actor_id=actor_id, deadline=deadline, token=token, kind=kind, wake_id=wake_id, order=self._next_order
        )

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

    def is_sleeping(self, actor_id: int) -> bool:
        """Whether *actor_id* has a pending ``sleep``-kind deadline."""
        with self._lock:
            return any(e.actor_id == actor_id and e.kind == "sleep" for e in self._deadlines.values())

    def in_timed_wait(self, actor_id: int) -> bool:
        """Whether *actor_id* has a pending ``timeout``-kind deadline."""
        with self._lock:
            return any(e.actor_id == actor_id and e.kind == "timeout" for e in self._deadlines.values())

    def sleep_deadline(self, actor_id: int) -> float | None:
        """The earliest ``sleep``-kind deadline for *actor_id* (``None`` if none)."""
        with self._lock:
            deadlines = [e.deadline for e in self._deadlines.values() if e.actor_id == actor_id and e.kind == "sleep"]
            return min(deadlines) if deadlines else None

    def timed_wait_deadline(self, actor_id: int) -> float | None:
        """The earliest ``timeout``-kind deadline for *actor_id* (``None`` if none)."""
        with self._lock:
            deadlines = [e.deadline for e in self._deadlines.values() if e.actor_id == actor_id and e.kind == "timeout"]
            return min(deadlines) if deadlines else None

    def sleeping_actors(self) -> list[int]:
        """Sorted actor ids with a pending ``sleep``-kind deadline (diagnostics)."""
        with self._lock:
            return sorted({e.actor_id for e in self._deadlines.values() if e.kind == "sleep"})

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
            return due


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
        real = _saved_originals["datetime"]
        if clock is None:
            return real.now(tz)
        return real.fromtimestamp(clock.now(), tz)

    @classmethod
    def utcnow(cls) -> _datetime.datetime:
        clock = _active_virtual_clock()
        real = _saved_originals["datetime"]
        if clock is None:
            return real.now(_datetime.timezone.utc).replace(tzinfo=None)
        return real.fromtimestamp(clock.now(), _datetime.timezone.utc).replace(tzinfo=None)


class _VirtualDate(_real_date, metaclass=_VirtualDateMeta):
    @classmethod
    def today(cls) -> _datetime.date:
        clock = _active_virtual_clock()
        if clock is None:
            return _saved_originals["date"].today()
        return _saved_originals["datetime"].fromtimestamp(clock.now()).date()


# ---------------------------------------------------------------------------
# Active-clock resolution
# ---------------------------------------------------------------------------

#: The exploration driver sets this contextvar around setup/tasks/invariant.
#: contextvars are per-thread *and* per-context: the sync driver reads it on the
#: thread it set it on, and async task contexts inherit it (task contexts copy
#: the creating context) while the event loop's base context does not — so loop
#: timers stay on the wall clock.
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
    return _clock_var.get()


@contextmanager
def clock_context(clock: VirtualClock | None) -> Generator[None, None, None]:
    """Register *clock* on :data:`_clock_var` for the duration of the block.

    Sets the contextvar and owns the ``time.*`` patch (reference-counted) so it
    works even outside a runner's patch scope — invariant evaluation happens
    after the workers' patch scope has already been unwound.  A ``None`` clock
    makes this a no-op.

    contextvars are per-thread and per-context, so this serves both callers:

    - sync exploration drivers wrap ``setup()`` / invariant evaluation, so state
      created / inspected on the driver thread sees the same virtual time as the
      workers;
    - async exploration wraps setup/tasks/invariant — asyncio tasks created
      inside the block inherit the contextvar (task contexts copy the creating
      context), while the event loop's own ``_run_once`` machinery runs in the
      loop's base context and keeps seeing real time, so loop timers stay on the
      wall clock.
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


#: The sync exploration drivers historically used a distinct ``clock_scope``
#: backed by a thread-local dict; it now shares the contextvar implementation
#: (contextvars are per-thread, so the driver reads back what it set).  Kept as
#: a separate public name for the sync call sites.
clock_scope = clock_context


# ---------------------------------------------------------------------------
# time.* patching (reference-counted, like patch_sleep in _cooperative)
# ---------------------------------------------------------------------------


def _virtual_time() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _saved_originals["time"]()


def _virtual_monotonic() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _saved_originals["monotonic"]()


def _virtual_perf_counter() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _saved_originals["perf_counter"]()


def _virtual_time_ns() -> int:
    clock = _active_virtual_clock()
    return round(clock.now() * 1e9) if clock is not None else _saved_originals["time_ns"]()


def _virtual_monotonic_ns() -> int:
    clock = _active_virtual_clock()
    return round(clock.now() * 1e9) if clock is not None else _saved_originals["monotonic_ns"]()


def _virtual_perf_counter_ns() -> int:
    clock = _active_virtual_clock()
    return round(clock.now() * 1e9) if clock is not None else _saved_originals["perf_counter_ns"]()


#: The single table driving :func:`patch_time` / :func:`unpatch_time` and the
#: 0->1 save of currently-installed values: ``(module, attr, virtual_obj)``.
_PATCHES: tuple[tuple[Any, str, Any], ...] = (
    (time, "time", _virtual_time),
    (time, "monotonic", _virtual_monotonic),
    (time, "perf_counter", _virtual_perf_counter),
    (time, "time_ns", _virtual_time_ns),
    (time, "monotonic_ns", _virtual_monotonic_ns),
    (time, "perf_counter_ns", _virtual_perf_counter_ns),
    (_datetime, "datetime", _VirtualDateTime),
    (_datetime, "date", _VirtualDate),
)

_time_patch_count = 0
_time_patch_lock = _rt.lock()


def patch_time() -> None:
    """Route ``time.{time,monotonic,perf_counter}`` (+ ``_ns``) through the
    active virtual clock.  Reference-counted; unaffected threads see real time.
    """
    global _time_patch_count  # noqa: PLW0603
    with _time_patch_lock:
        _time_patch_count += 1
        if _time_patch_count > 1:
            return
        # Snapshot whatever is currently installed (possibly a third-party
        # fake) *before* overwriting, so the gated fallback and unpatch_time
        # honor it rather than the pristine import-time C functions.
        for module, attr, virtual_obj in _PATCHES:
            _saved_originals[attr] = getattr(module, attr)
            setattr(module, attr, virtual_obj)


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
        for module, attr, _ in _PATCHES:
            setattr(module, attr, _saved_originals[attr])


__all__ = [
    "VIRTUAL_EPOCH",
    "ClockConfig",
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
