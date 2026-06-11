"""Regression: closure-cell lost update under burst schedules on the settrace path.

A burst schedule lets one worker run several opcodes while another is paused
mid-function.  On CPython 3.10-3.11 (``sys.settrace`` opcode tracing) the
runtime writes a frame's ``f_locals`` snapshot back into closure cells via
``PyFrame_LocalsToFast`` after the trace callback returns.  If the paused
frame's snapshot is stale (another thread mutated the shared cell while it was
parked), that write clobbers the concurrent update — a lost update that
surfaces as a *false positive* on lock-protected closures.

``BytecodeShuffler._on_opcode`` must therefore signal "yielded" so
``make_settrace_callback`` refreshes the snapshot.  Lockstep round-robin never
exposed this; variable-length bursts (see ``_random_schedules``) do.

These tests are meaningful on 3.10/3.11; on 3.12+ (``sys.monitoring``) the same
code is exercised but the LocalsToFast hazard does not exist, so they simply
pass there too.
"""

from __future__ import annotations

import threading

import pytest

from frontrun.bytecode import run_with_schedule
from frontrun.cli import is_active as _frontrun_active

pytestmark = pytest.mark.skipif(
    not _frontrun_active(),
    reason="requires the `frontrun` CLI wrapper (cooperative lock patching)",
)


class _ClosureLockState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        count = 0

        def get() -> int:
            return count

        def inc() -> None:
            nonlocal count
            with self.lock:
                temp = count
                count = temp + 1

        self.get = get
        self.inc = inc


def _burst_schedule() -> list[int]:
    # Thread 1 enters its lock acquire, then thread 0 gets a long burst to run
    # its entire critical section (read+write the shared cell), then thread 1
    # resumes.  This is exactly the window that triggers a stale-snapshot
    # write-back of the closure cell on the settrace path.
    return [1] * 3 + [0] * 60 + [1] * 60 + [0] * 60 + [1] * 60


def test_closure_cell_under_lock_not_corrupted_by_burst() -> None:
    state = run_with_schedule(
        _burst_schedule(),
        _ClosureLockState,
        [lambda s: s.inc(), lambda s: s.inc()],
        timeout=5.0,
        detect_io=True,
    )
    assert state.get() == 2, "lock-protected closure cell lost an update under a burst schedule"


def test_closure_cell_burst_no_io() -> None:
    state = run_with_schedule(
        _burst_schedule(),
        _ClosureLockState,
        [lambda s: s.inc(), lambda s: s.inc()],
        timeout=5.0,
        detect_io=False,
    )
    assert state.get() == 2
