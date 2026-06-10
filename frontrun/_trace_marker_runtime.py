from __future__ import annotations

import linecache
from collections.abc import Callable
from typing import Any

from frontrun._opcode_observer import install_thread_line_trace, uninstall_thread_line_trace


def _is_non_executable_line(filename: str, lineno: int) -> bool:
    """Whether *lineno* in *filename* is a comment-only or blank line.

    Such lines never produce their own ``line`` trace event, so a marker placed
    there is legitimately attached to the following executable line (the
    ``# frontrun: name`` above an ``await`` pattern).  An executable line, by
    contrast, would fire its own event when it runs — so a marker on it must
    only fire when it actually executed, not merely because the next physical
    line ran (finding 8).
    """
    source = linecache.getline(filename, lineno)
    if not source:
        # Unknown/unreadable line: be conservative and treat as executable so we
        # don't fire for a line we can't verify ran.
        return False
    stripped = source.strip()
    return stripped == "" or stripped.startswith("#")


def _release_execution_lock_safely(coordinator: Any) -> None:
    try:
        coordinator._execution_lock.release()
    except RuntimeError:
        pass


def _wait_for_marker(coordinator: Any, execution_name: str, marker_name: str) -> None:
    coordinator._execution_lock.release()
    coordinator.wait_for_turn(execution_name, marker_name, _reacquire_execution_lock=True)
    if coordinator.error:
        raise coordinator.error


def build_trace_function(
    coordinator: Any,
    marker_registry: Any,
    execution_name: str,
    *,
    include_previous_line: bool,
) -> Callable[[Any, str, Any], Any]:
    """Build a trace function that blocks execution when markers are reached."""
    _last_current_line_marker: list[tuple[str, int] | None] = [None]
    _last_prev_line_fired: list[tuple[str, int] | None] = [None]
    # Last executed (filename, lineno) per frame id, so we can tell whether the
    # physically-preceding line actually ran (vs. was skipped) — see finding 8.
    _last_executed: dict[int, tuple[str, int]] = {}

    def trace_function(frame: Any, event: str, arg: Any) -> Any:
        try:
            if event != "line":
                return trace_function

            marker_registry.scan_frame(frame)

            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            frame_id = id(frame)
            prev_executed = _last_executed.get(frame_id)
            _last_executed[frame_id] = (filename, lineno)

            marker_name = marker_registry.get_marker(filename, lineno)
            if marker_name:
                _last_current_line_marker[0] = (filename, lineno)
                _last_prev_line_fired[0] = None
                _wait_for_marker(coordinator, execution_name, marker_name)
                return trace_function

            if include_previous_line and lineno > 1 and _last_current_line_marker[0] != (filename, lineno - 1):
                prev_marker = marker_registry.get_marker(filename, lineno - 1)
                # Only fire the prev-line marker when line lineno-1 is a
                # comment/blank line (legitimately attached to this line) OR the
                # previous *executed* line in this frame was exactly lineno-1.
                # Otherwise lineno-1 is executable code that was skipped, and
                # firing its marker would report a step that never ran.
                prev_line_legit = _is_non_executable_line(filename, lineno - 1) or prev_executed == (
                    filename,
                    lineno - 1,
                )
                if prev_marker and prev_line_legit and _last_prev_line_fired[0] != (filename, lineno):
                    _last_prev_line_fired[0] = (filename, lineno)
                    _wait_for_marker(coordinator, execution_name, prev_marker)

            if _last_prev_line_fired[0] is not None and _last_prev_line_fired[0] != (filename, lineno):
                _last_prev_line_fired[0] = None

            return trace_function
        except Exception as error:
            _release_execution_lock_safely(coordinator)
            coordinator.report_error(error)
            return None

    return trace_function


def run_traced_callable(
    coordinator: Any,
    execution_name: str,
    body: Callable[[], None],
    error_sink: dict[str, Exception],
    trace_function: Callable[[Any, str, Any], Any] | None = None,
) -> None:
    """Run a callable with tracing enabled and guaranteed cleanup."""
    error: Exception | None = None
    try:
        coordinator._execution_lock.acquire()
        install_thread_line_trace(trace_function)
        body()
    except Exception as exc:
        error = exc
        error_sink[execution_name] = exc
    finally:
        uninstall_thread_line_trace()
        _release_execution_lock_safely(coordinator)
        if error is not None:
            coordinator.report_error(error)
