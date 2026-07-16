"""Shared imports and TLS state for sync DPOR internals."""

# ruff: noqa: F401
# pyright: reportUnusedFunction=false, reportUnsupportedDunderAll=false

from __future__ import annotations

import linecache
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from frontrun._cooperative import (
    clear_context,
    patch_locks,
    patch_sleep,
    real_condition,
    real_lock,
    set_context,
    set_sync_reporter,
    unpatch_locks,
    unpatch_sleep,
)
from frontrun._deadlock import DeadlockError, SchedulerAbort, install_wait_for_graph, uninstall_wait_for_graph
from frontrun._io_detection import (
    patch_io,
    set_io_reporter,
    unpatch_io,
)
from frontrun._opcode_observer import (
    ShadowStack,
    StableObjectIds,
    _get_instructions,
    _make_object_key,
    _process_opcode,
    clear_instr_cache,
    get_object_key_reverse_map,
    process_opcode_with_coarsening,
    set_object_key_reverse_map,
)
from frontrun._redis_client import (
    is_redis_tid_suppressed,
    patch_redis,
    unpatch_redis,
)
from frontrun._sql_anomaly import classify_sql_anomaly
from frontrun._sql_cursor import (
    _clear_active_sql_io_context,
    clear_sql_metadata,
    get_active_sql_io_context,
    is_sql_endpoint_suppressed,
    is_sql_write_suppressed,
    is_tid_suppressed,
    patch_sql,
    unpatch_sql,
)
from frontrun._sql_insert_tracker import clear_insert_tracker, ensure_no_uncaptured_inserts
from frontrun._trace_format import TraceRecorder, build_call_chain, format_trace
from frontrun._tracing import TraceFilter as _TraceFilter
from frontrun._tracing import set_active_trace_filter as _set_active_trace_filter
from frontrun._tracing import should_trace_file as _should_trace_file
from frontrun.cli import require_active as _require_frontrun_env
from frontrun.common import (
    InterleavingResult,
    _call_sync_setup,
    check_invariant,
    check_serializability_violation,
)

try:
    from frontrun._dpor import PyDporEngine, PyExecution  # type: ignore[reportAttributeAccessIssue]
except ModuleNotFoundError as _err:
    raise ModuleNotFoundError(
        "frontrun.explore requires the frontrun._dpor Rust extension.\n"
        "Build it with:  make build-dpor-3.14t   (or build-dpor-3.10 / build-dpor-3.14)\n"
        "Or install from source:  pip install -e ."
    ) from _err

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Thread-local state for the DPOR scheduler
# ---------------------------------------------------------------------------

_dpor_tls = threading.local()


def _append_unique_lock_event(lock_events: list[Any], event: Any) -> None:
    """Append a lock event unless it duplicates the immediately previous one."""
    if lock_events:
        last = lock_events[-1]
        if (
            last.schedule_index == event.schedule_index
            and last.thread_id == event.thread_id
            and last.event_type == event.event_type
            and last.lock_id == event.lock_id
        ):
            return
    lock_events.append(event)


__all__ = [name for name in globals() if not name.startswith("__")]
