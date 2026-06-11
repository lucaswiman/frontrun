"""Finding 3: sys.monitoring tool-id stealing must respect ownership.

setup_opcode_monitoring previously force-freed a tool id slot on any ValueError
from use_tool_id, assuming the holder was a stale frontrun run.  If the holder
is a live external tool (a profiler) or a concurrent frontrun run, this
destroyed its instrumentation.  The fix only force-frees slots the ownership
registry shows belong to a finished/dead frontrun run; otherwise it raises a
clear error.
"""

from __future__ import annotations

import sys

import pytest

_USE_SYS_MONITORING = sys.version_info[:2] >= (3, 12)

pytestmark = pytest.mark.skipif(
    not _USE_SYS_MONITORING,
    reason="tool-id ownership only applies to the sys.monitoring backend (3.12+)",
)


def test_refuses_to_steal_external_tool():
    """An external (non-frontrun) tool holding the slot must not be stolen."""
    from frontrun._opcode_observer import setup_opcode_monitoring, teardown_opcode_monitoring

    mon = sys.monitoring
    tool_id = mon.OPTIMIZER_ID

    # Simulate an external tool holding the optimizer slot.
    mon.use_tool_id(tool_id, "external-profiler")
    try:
        with pytest.raises(RuntimeError, match="not owned by frontrun|already in use"):
            setup_opcode_monitoring(
                tool_name="frontrun-bytecode",
                handle_py_start=lambda *a: None,
                handle_py_return=lambda *a: None,
                handle_instruction=lambda *a: None,
                tool_kind="optimizer",
                monitor_returns=False,
            )
        # The external tool must still hold the slot.
        assert mon.get_tool(tool_id) == "external-profiler"
    finally:
        # Clean up: if frontrun erroneously stole it this free will fail loudly.
        try:
            teardown_opcode_monitoring(tool_id)
        except Exception:
            mon.free_tool_id(tool_id)


def test_reclaims_stale_frontrun_slot():
    """A leaked frontrun slot whose owner thread is gone is reclaimable."""
    from frontrun import _opcode_observer as obs

    mon = sys.monitoring
    tool_id = mon.OPTIMIZER_ID

    # Set up once (registers frontrun ownership) but DON'T tear down — simulate
    # an interrupted run.  Then forge the owner ident to a dead thread id so the
    # liveness check treats it as stale, and set up again.
    obs.setup_opcode_monitoring(
        tool_name="frontrun-bytecode",
        handle_py_start=lambda *a: None,
        handle_py_return=lambda *a: None,
        handle_instruction=lambda *a: None,
        tool_kind="optimizer",
        monitor_returns=False,
    )
    # Forge a dead owner ident (0 is never a live thread ident).
    obs._TOOL_OWNERS[tool_id] = 0

    tid = None
    try:
        tid = obs.setup_opcode_monitoring(
            tool_name="frontrun-bytecode",
            handle_py_start=lambda *a: None,
            handle_py_return=lambda *a: None,
            handle_instruction=lambda *a: None,
            tool_kind="optimizer",
            monitor_returns=False,
        )
        assert tid == tool_id
    finally:
        obs.teardown_opcode_monitoring(tool_id)
