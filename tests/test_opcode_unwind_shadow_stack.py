"""Finding 4: sys.monitoring backend must clean up shadow stacks on PY_UNWIND.

The monitoring backend registers PY_START | INSTRUCTION | PY_RETURN.  A frame
that exits via an exception fires PY_UNWIND, NOT PY_RETURN, so
``remove_shadow_stack(id(frame))`` was skipped.  A later frame that reuses the
same id() then inherits the dead frame's operand stack.

This test registers monitoring callbacks (with a recording
``remove_shadow_stack``) and runs a traced function that raises; the unwinding
frame's id must be removed.
"""

from __future__ import annotations

import sys

import pytest

from frontrun._opcode_observer import (
    make_monitoring_callbacks,
    setup_opcode_monitoring,
    teardown_opcode_monitoring,
)

_USE_SYS_MONITORING = sys.version_info[:2] >= (3, 12)

pytestmark = pytest.mark.skipif(
    not _USE_SYS_MONITORING,
    reason="PY_UNWIND cleanup only applies to the sys.monitoring backend (3.12+)",
)


def _raises() -> None:
    x = 1 + 1  # noqa: F841  (ensure at least one traced instruction)
    raise ValueError("boom")


def test_py_unwind_removes_shadow_stack():
    removed: list[int] = []

    # get_thread_id must be non-None so handle_py_unwind does its cleanup.
    callbacks = make_monitoring_callbacks(
        get_thread_id=lambda: 0,
        on_opcode=lambda code, offset, frame, tid: None,
        remove_shadow_stack=removed.append,
        detect_io=False,
        is_active=lambda: True,
    )
    # New 4-tuple: start, return, unwind, instruction.
    handle_py_start, handle_py_return, handle_py_unwind, handle_instruction = callbacks

    tool_id = setup_opcode_monitoring(
        tool_name="frontrun-test-unwind",
        handle_py_start=handle_py_start,
        handle_py_return=handle_py_return,
        handle_py_unwind=handle_py_unwind,
        handle_instruction=handle_instruction,
        tool_kind="optimizer",
        monitor_returns=True,
    )
    try:
        with pytest.raises(ValueError, match="boom"):
            _raises()
    finally:
        teardown_opcode_monitoring(tool_id)

    assert removed, "PY_UNWIND did not remove the shadow stack of the unwinding frame"
