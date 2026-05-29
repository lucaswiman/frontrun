from __future__ import annotations

from collections.abc import Callable
from typing import Any

from frontrun._opcode_observer import install_thread_line_trace, uninstall_thread_line_trace


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

    def trace_function(frame: Any, event: str, arg: Any) -> Any:
        try:
            if event != "line":
                return trace_function

            marker_registry.scan_frame(frame)

            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

            marker_name = marker_registry.get_marker(filename, lineno)
            if marker_name:
                _last_current_line_marker[0] = (filename, lineno)
                _last_prev_line_fired[0] = None
                _wait_for_marker(coordinator, execution_name, marker_name)
                return trace_function

            if include_previous_line and lineno > 1 and _last_current_line_marker[0] != (filename, lineno - 1):
                prev_marker = marker_registry.get_marker(filename, lineno - 1)
                if prev_marker and _last_prev_line_fired[0] != (filename, lineno):
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
