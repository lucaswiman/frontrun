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

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

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


def real_monotonic() -> float:
    """The saved, never-patched ``time.monotonic`` (for scheduler machinery)."""
    return _real_monotonic()


def validate_clock(clock: str) -> ClockMode:
    """Validate a user-supplied ``clock=`` value, returning it typed."""
    if clock not in _CLOCK_MODES:
        raise ValueError(f"unknown clock={clock!r}; must be one of 'real', 'virtual', 'explored'")
    return clock  # type: ignore[return-value]


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


# ---------------------------------------------------------------------------
# Active-clock resolution
# ---------------------------------------------------------------------------

# Threads registered explicitly (exploration driver during setup()/invariant()).
_thread_clocks: dict[int, VirtualClock] = {}

#: Async exploration sets this contextvar around setup/tasks/invariant; task
#: contexts inherit it, the event loop's base context does not.
_clock_var: ContextVar[VirtualClock | None] = ContextVar("frontrun_virtual_clock", default=None)


def _active_virtual_clock() -> VirtualClock | None:
    """Return the virtual clock for the calling thread/context, if any."""
    from frontrun._cooperative import get_context

    ctx = get_context()
    if ctx is not None:
        clock = getattr(ctx[0], "virtual_clock", None)
        if clock is not None:
            return clock  # type: ignore[no-any-return]
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
    """
    if clock is None:
        yield
        return
    ident = threading.get_ident()
    prev = _thread_clocks.get(ident)
    _thread_clocks[ident] = clock
    try:
        yield
    finally:
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
    """
    if clock is None:
        yield
        return
    token = _clock_var.set(clock)
    try:
        yield
    finally:
        _clock_var.reset(token)


# ---------------------------------------------------------------------------
# time.* patching (reference-counted, like patch_sleep in _cooperative)
# ---------------------------------------------------------------------------


def _virtual_time() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _real_time()


def _virtual_monotonic() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _real_monotonic()


def _virtual_perf_counter() -> float:
    clock = _active_virtual_clock()
    return clock.now() if clock is not None else _real_perf_counter()


def _virtual_time_ns() -> int:
    clock = _active_virtual_clock()
    return int(clock.now() * 1e9) if clock is not None else _real_time_ns()


def _virtual_monotonic_ns() -> int:
    clock = _active_virtual_clock()
    return int(clock.now() * 1e9) if clock is not None else _real_monotonic_ns()


def _virtual_perf_counter_ns() -> int:
    clock = _active_virtual_clock()
    return int(clock.now() * 1e9) if clock is not None else _real_perf_counter_ns()


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
        time.time = _virtual_time
        time.monotonic = _virtual_monotonic
        time.perf_counter = _virtual_perf_counter
        time.time_ns = _virtual_time_ns
        time.monotonic_ns = _virtual_monotonic_ns
        time.perf_counter_ns = _virtual_perf_counter_ns


def unpatch_time() -> None:
    """Restore the original ``time.*`` functions once all patchers release."""
    global _time_patch_count  # noqa: PLW0603
    with _time_patch_lock:
        if _time_patch_count <= 0:
            return
        _time_patch_count -= 1
        if _time_patch_count > 0:
            return
        time.time = _real_time
        time.monotonic = _real_monotonic
        time.perf_counter = _real_perf_counter
        time.time_ns = _real_time_ns
        time.monotonic_ns = _real_monotonic_ns
        time.perf_counter_ns = _real_perf_counter_ns


__all__ = [
    "VIRTUAL_EPOCH",
    "ClockMode",
    "VirtualClock",
    "clock_context",
    "clock_scope",
    "patch_time",
    "real_monotonic",
    "unpatch_time",
    "validate_clock",
]
