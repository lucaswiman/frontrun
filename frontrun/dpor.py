"""Re-exports of sync DPOR internals used by the test suite.

The canonical implementation lives in :mod:`frontrun._dpor_runtime`; this
module exists only as a stable import path for the symbols that tests reach
into directly.  Code outside the tests should use :func:`frontrun.explore`.
"""

from __future__ import annotations

from frontrun._dpor_runtime._shared import (
    ShadowStack,
    StableObjectIds,
    _append_unique_lock_event,
    _dpor_tls,
    _make_object_key,
    _process_opcode,
)
from frontrun._dpor_runtime.preload_bridge import _PreloadBridge
from frontrun._dpor_runtime.runner import DporBytecodeRunner
from frontrun._dpor_runtime.scheduler import (
    DporScheduler,
    _IOAnchoredReplayScheduler,
)

__all__ = [
    "DporBytecodeRunner",
    "DporScheduler",
    "ShadowStack",
    "StableObjectIds",
    "_IOAnchoredReplayScheduler",
    "_PreloadBridge",
    "_append_unique_lock_event",
    "_dpor_tls",
    "_make_object_key",
    "_process_opcode",
]
