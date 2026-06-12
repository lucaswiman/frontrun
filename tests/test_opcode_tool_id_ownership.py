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


def test_teardown_does_not_free_other_owners_slot():
    """teardown_opcode_monitoring must not free a slot owned by another thread.

    If a run dies and its tool_id is reclaimed by a new run on a different
    thread, the dead run's teardown must not destroy the new owner's
    monitoring callbacks.
    """
    import threading

    from frontrun import _opcode_observer as obs

    mon = sys.monitoring
    tool_id = mon.OPTIMIZER_ID

    # Set up monitoring normally.
    tid = obs.setup_opcode_monitoring(
        tool_name="frontrun-bytecode",
        handle_py_start=lambda *a: None,
        handle_py_return=lambda *a: None,
        handle_instruction=lambda *a: None,
        tool_kind="optimizer",
        monitor_returns=False,
    )
    assert tid == tool_id

    # Simulate another thread taking over the slot: forge the owner to a
    # different (but live) thread ident.  Use the main thread ident since
    # we're calling teardown from a *different* thread.
    main_ident = threading.get_ident()
    # We'll run teardown from a helper thread.  First, forge ownership to
    # the main thread so the helper thread's teardown should NOT free it.
    obs._TOOL_OWNERS[tool_id] = main_ident

    freed = []

    def _teardown_from_other_thread():
        obs.teardown_opcode_monitoring(tool_id)
        # After teardown, check if the tool is still registered.
        tool_name = mon.get_tool(tool_id)
        freed.append(tool_name)

    t = threading.Thread(target=_teardown_from_other_thread)
    t.start()
    t.join(timeout=5.0)

    try:
        # The tool should still be registered because the other thread
        # should not have freed our slot.
        assert freed[0] == "frontrun-bytecode", (
            f"Expected tool to still be 'frontrun-bytecode' but got {freed[0]!r}; "
            "teardown_opcode_monitoring freed a slot owned by another thread"
        )
    finally:
        # Clean up properly.
        obs._TOOL_OWNERS[tool_id] = threading.get_ident()
        obs.teardown_opcode_monitoring(tool_id)
