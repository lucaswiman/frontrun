"""Trace recording, filtering, and formatting for comprehensible race condition errors.

When frontrun finds a race condition, the raw counterexample is a list of thread
indices — one per bytecode instruction. This module transforms that into a
human-readable "story" of which source lines executed in which order.

The pipeline:
1. **Record** a TraceEvent at each opcode during the failing run.
2. **Filter** to events that touch shared state (LOAD_ATTR/STORE_ATTR, etc.)
3. **Deduplicate** consecutive events from the same thread on the same source line.
4. **Classify** the conflict pattern (lost update, order violation, etc.)
5. **Format** as an interleaved source-line trace.
"""

from __future__ import annotations

import dis
import linecache
import sys
from dataclasses import dataclass
from typing import Any

from frontrun._cooperative import real_lock

_PY_VERSION = sys.version_info[:2]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TraceEvent:
    """A single recorded event from the trace."""

    step_index: int
    thread_id: int
    filename: str
    lineno: int
    function_name: str
    opcode: str
    access_type: str | None = None  # "read", "write", or None
    attr_name: str | None = None  # e.g. "value", "balance"
    obj_type_name: str | None = None  # e.g. "Counter", "BankAccount"


@dataclass(slots=True)
class SourceLineEvent:
    """A deduplicated, source-level event for display."""

    thread_id: int
    filename: str
    lineno: int
    function_name: str
    source_line: str
    access_type: str | None = None  # "read", "write", or None (if mixed, "read+write")
    attr_name: str | None = None
    obj_type_name: str | None = None


# ---------------------------------------------------------------------------
# Trace recorder
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Accumulates TraceEvent objects during a single execution.

    Thread-safe: multiple threads call ``record()`` concurrently, each
    holding the scheduler lock (so ordering is deterministic).
    """

    __slots__ = ("events", "_step", "_lock", "enabled")

    def __init__(self, *, enabled: bool = True) -> None:
        self.events: list[TraceEvent] = []
        self._step = 0
        self._lock = real_lock()
        self.enabled = enabled

    def record(
        self,
        thread_id: int,
        frame: Any,
        opcode: str | None = None,
        access_type: str | None = None,
        attr_name: str | None = None,
        obj: Any = None,
    ) -> None:
        """Record one trace event from a frame object."""
        if not self.enabled:
            return
        code = frame.f_code
        obj_type_name: str | None = None
        if obj is not None:
            obj_type_name = type(obj).__name__

        with self._lock:
            step = self._step
            self._step += 1

        ev = TraceEvent(
            step_index=step,
            thread_id=thread_id,
            filename=code.co_filename,
            lineno=frame.f_lineno,
            function_name=code.co_name,
            opcode=opcode or "",
            access_type=access_type,
            attr_name=attr_name,
            obj_type_name=obj_type_name,
        )
        # Append under the recorder lock for ordering consistency
        with self._lock:
            self.events.append(ev)

    def record_from_opcode(
        self,
        thread_id: int,
        frame: Any,
    ) -> None:
        """Record an event using the frame's current instruction.

        Used by the bytecode explorer, which doesn't do shadow-stack
        analysis. We inspect the instruction to extract access info.
        """
        if not self.enabled:
            return
        code = frame.f_code
        offset = frame.f_lasti
        instr = _get_instruction(code, offset)
        if instr is None:
            return

        op = instr.opname
        access_type: str | None = None
        attr_name: str | None = None
        obj: Any = None

        if op == "LOAD_ATTR":
            access_type = "read"
            attr_name = instr.argval
        elif op == "STORE_ATTR":
            access_type = "write"
            attr_name = instr.argval
        elif op == "DELETE_ATTR":
            access_type = "write"
            attr_name = instr.argval
        elif op in ("BINARY_SUBSCR", "STORE_SUBSCR", "DELETE_SUBSCR"):
            access_type = "write" if op.startswith(("STORE", "DELETE")) else "read"
        elif op == "BINARY_OP":
            argrepr = instr.argrepr
            if argrepr and ("[" in argrepr or "NB_SUBSCR" in argrepr.upper()):
                access_type = "read"
        else:
            # Not an interesting opcode
            return

        self.record(
            thread_id=thread_id,
            frame=frame,
            opcode=op,
            access_type=access_type,
            attr_name=attr_name,
            obj=obj,
        )


# Lightweight instruction cache (separate from DPOR's to avoid cross-module coupling)
_instr_cache: dict[int, dict[int, dis.Instruction]] = {}


def _get_instruction(code: Any, offset: int) -> dis.Instruction | None:
    code_id = id(code)
    mapping = _instr_cache.get(code_id)
    if mapping is None:
        mapping = {}
        if _PY_VERSION >= (3, 11):
            instructions = dis.get_instructions(code, show_caches=False)
        else:
            instructions = dis.get_instructions(code)
        for inst in instructions:
            mapping[inst.offset] = inst
        _instr_cache[code_id] = mapping
    return mapping.get(offset)


# ---------------------------------------------------------------------------
# Filtering and deduplication
# ---------------------------------------------------------------------------


def _is_shared_access(ev: TraceEvent) -> bool:
    """Return True if this event represents an access to shared state."""
    return ev.access_type is not None


def filter_to_shared_accesses(events: list[TraceEvent]) -> list[TraceEvent]:
    """Keep only events that access shared mutable state."""
    return [ev for ev in events if _is_shared_access(ev)]


def deduplicate_to_source_lines(events: list[TraceEvent]) -> list[SourceLineEvent]:
    """Collapse consecutive events from the same thread+line into one SourceLineEvent.

    When multiple opcodes on the same source line produce events (e.g.,
    LOAD_ATTR then STORE_ATTR for ``self.value += 1``), merge them into
    a single entry with a combined access_type.
    """
    if not events:
        return []

    result: list[SourceLineEvent] = []
    prev_tid = -1
    prev_lineno = -1
    prev_filename = ""

    for ev in events:
        same_line = ev.thread_id == prev_tid and ev.lineno == prev_lineno and ev.filename == prev_filename

        if same_line and result:
            last = result[-1]
            # Merge access types
            if last.access_type != ev.access_type and ev.access_type is not None:
                if last.access_type is None:
                    last.access_type = ev.access_type
                elif last.access_type == "read" and ev.access_type == "write":
                    last.access_type = "read+write"
                elif last.access_type == "write" and ev.access_type == "read":
                    last.access_type = "read+write"
            # Prefer more specific attr info
            if ev.attr_name is not None and last.attr_name is None:
                last.attr_name = ev.attr_name
            if ev.obj_type_name is not None and last.obj_type_name is None:
                last.obj_type_name = ev.obj_type_name
        else:
            source_line = linecache.getline(ev.filename, ev.lineno).strip()
            result.append(
                SourceLineEvent(
                    thread_id=ev.thread_id,
                    filename=ev.filename,
                    lineno=ev.lineno,
                    function_name=ev.function_name,
                    source_line=source_line,
                    access_type=ev.access_type,
                    attr_name=ev.attr_name,
                    obj_type_name=ev.obj_type_name,
                )
            )
            prev_tid = ev.thread_id
            prev_lineno = ev.lineno
            prev_filename = ev.filename

    return result


# ---------------------------------------------------------------------------
# Conflict pattern classification
# ---------------------------------------------------------------------------


@dataclass
class ConflictInfo:
    """Description of the conflict pattern found in the trace."""

    pattern: str  # "lost_update", "stale_read", "write_write", "order_violation", "unknown"
    summary: str  # One-line human-readable explanation
    attr_name: str | None = None  # attribute involved, if identifiable


def classify_conflict(events: list[SourceLineEvent]) -> ConflictInfo:
    """Examine a filtered, deduplicated trace and classify the conflict type.

    Looks for classic patterns:
    - Lost update: R_a R_b W_a W_b (or R_a R_b W_b W_a)
    - Write-write: W_a W_b on same attribute without intervening sync
    """
    if not events:
        return ConflictInfo(pattern="unknown", summary="No shared-state accesses recorded.")

    # Track per-attribute access sequences across threads
    # Group by attr_name, look for cross-thread read-before-write patterns
    attr_accesses: dict[str, list[tuple[int, str]]] = {}  # attr -> [(thread_id, access_type), ...]
    for ev in events:
        key = ev.attr_name or "(unknown)"
        attr_accesses.setdefault(key, []).append((ev.thread_id, ev.access_type or "unknown"))

    for attr, accesses in attr_accesses.items():
        threads_involved = sorted({tid for tid, _ in accesses})
        if len(threads_involved) < 2:
            continue

        # Check for lost-update pattern: two threads both read before either writes
        # Pattern: R_a ... R_b ... W_a ... W_b (or W_b before W_a)
        first_read: dict[int, int] = {}  # thread -> index of first read
        first_write: dict[int, int] = {}  # thread -> index of first write
        for i, (tid, atype) in enumerate(accesses):
            if atype in ("read", "read+write") and tid not in first_read:
                first_read[tid] = i
            if atype in ("write", "read+write") and tid not in first_write:
                first_write[tid] = i

        # Look for pairs where both threads read before either writes
        for t_a in threads_involved:
            for t_b in threads_involved:
                if t_a >= t_b:
                    continue
                r_a = first_read.get(t_a)
                r_b = first_read.get(t_b)
                w_a = first_write.get(t_a)
                w_b = first_write.get(t_b)

                if r_a is not None and r_b is not None and w_a is not None and w_b is not None:
                    # Both read before both write?
                    writes_start = min(w_a, w_b)
                    if r_a < writes_start and r_b < writes_start:
                        obj_desc = attr
                        return ConflictInfo(
                            pattern="lost_update",
                            summary=(
                                f"Lost update: threads {t_a} and {t_b} both read "
                                f"{obj_desc} before either wrote it back."
                            ),
                            attr_name=attr,
                        )

        # Check for write-write without reads (simple overwrite)
        for t_a in threads_involved:
            for t_b in threads_involved:
                if t_a >= t_b:
                    continue
                w_a = first_write.get(t_a)
                w_b = first_write.get(t_b)
                if w_a is not None and w_b is not None:
                    return ConflictInfo(
                        pattern="write_write",
                        summary=f"Write-write conflict: threads {t_a} and {t_b} both wrote to {attr}.",
                        attr_name=attr,
                    )

    # Fallback: we recorded shared accesses but couldn't classify the pattern
    all_threads = sorted({ev.thread_id for ev in events})
    return ConflictInfo(
        pattern="unknown",
        summary=f"Race condition involving threads {', '.join(map(str, all_threads))}.",
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_trace(
    events: list[TraceEvent],
    *,
    num_threads: int,
    thread_names: list[str] | None = None,
    num_explored: int = 0,
    invariant_desc: str | None = None,
    show_opcodes: bool = False,
    reproduction_attempts: int = 0,
    reproduction_successes: int = 0,
) -> str:
    """Format a trace as a human-readable interleaved source-line display.

    Args:
        events: Raw trace events from a TraceRecorder.
        num_threads: Total number of threads.
        thread_names: Optional display names for threads.
        num_explored: Number of interleavings explored before finding the bug.
        invariant_desc: Description of the violated invariant.
        show_opcodes: If True, include opcode-level detail for each line.
        reproduction_attempts: How many times the schedule was replayed.
        reproduction_successes: How many replays reproduced the failure.

    Returns:
        Multi-line string suitable for printing or attaching to test output.
    """
    if thread_names is None:
        thread_names = [f"Thread {i}" for i in range(num_threads)]

    # Filter and deduplicate
    shared = filter_to_shared_accesses(events)
    if not shared:
        return _format_no_shared_accesses(events, num_explored=num_explored)

    lines = deduplicate_to_source_lines(shared)
    if not lines:
        return _format_no_shared_accesses(events, num_explored=num_explored)

    conflict = classify_conflict(lines)

    # Build output
    parts: list[str] = []

    # Header
    if num_explored > 0:
        parts.append(f"Race condition found after {num_explored} interleavings.\n")
    else:
        parts.append("Race condition found.\n")

    # Conflict summary
    parts.append(f"  {conflict.summary}\n")

    # Interleaved trace
    parts.append("")
    max_thread_label = max(len(name) for name in thread_names)

    # Compute short filename (just the basename)
    for line_ev in lines:
        label = thread_names[line_ev.thread_id].ljust(max_thread_label)
        short_file = _short_filename(line_ev.filename)
        loc = f"{short_file}:{line_ev.lineno}"
        access_tag = ""
        if line_ev.access_type:
            access_tag = f"  [{line_ev.access_type}]"
            if line_ev.attr_name:
                if line_ev.obj_type_name:
                    access_tag = f"  [{line_ev.access_type} {line_ev.obj_type_name}.{line_ev.attr_name}]"
                else:
                    access_tag = f"  [{line_ev.access_type} .{line_ev.attr_name}]"

        src = line_ev.source_line
        parts.append(f"  {label} | {loc:<25s} {src}{access_tag}")

        # Optional opcode detail
        if show_opcodes:
            # Show the raw opcodes that contributed to this source line
            matching = [
                ev
                for ev in shared
                if ev.thread_id == line_ev.thread_id and ev.lineno == line_ev.lineno and ev.filename == line_ev.filename
            ]
            for ev in matching:
                attr_detail = f" .{ev.attr_name}" if ev.attr_name else ""
                parts.append(f"  {'':>{max_thread_label}} |   {ev.opcode}{attr_detail}")

    # Invariant description
    if invariant_desc:
        parts.append("")
        parts.append(f"  Invariant violated: {invariant_desc}")

    # Reproduction stats
    if reproduction_attempts > 0:
        parts.append("")
        pct = reproduction_successes * 100 // reproduction_attempts
        parts.append(f"  Reproduced {reproduction_successes}/{reproduction_attempts} times ({pct}%)")

    parts.append("")
    return "\n".join(parts)


def _format_no_shared_accesses(events: list[TraceEvent], *, num_explored: int = 0) -> str:
    """Fallback when no shared-state accesses were detected."""
    if num_explored > 0:
        return f"Race condition found after {num_explored} interleavings (no shared-state accesses recorded).\n"
    return "Race condition found (no shared-state accesses recorded).\n"


def _short_filename(path: str) -> str:
    """Convert an absolute path to a short display name."""
    import os

    basename = os.path.basename(path)
    return basename
