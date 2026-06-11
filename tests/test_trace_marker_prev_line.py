"""Finding 8: include_previous_line must not fire for never-executed lines.

The prev-line mechanism exists so a marker comment placed on its own
(non-executable) line directly above an ``await`` fires when that await line
runs — comment-only lines produce no line event of their own, so the marker is
attached to the following executable line.

The bug: the check fired ``get_marker(filename, lineno-1)`` purely on physical
adjacency, so a marker on an *executable* line that was skipped (e.g. the body
of a not-taken ``if``) would be fired when the next physical line ran, even
though the marked code never executed.

Fix: fire the prev-line marker only when line ``lineno-1`` is a non-executable
(comment/blank) line, or when the previous *executed* line in the frame was
exactly ``lineno-1``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from frontrun._marker_coordination import MarkerRegistry
from frontrun._trace_marker_runtime import build_trace_function


class _RecordingCoordinator:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self._execution_lock = threading.Lock()
        # The real runtime acquires the execution lock before tracing begins
        # (run_traced_callable); _wait_for_marker releases then re-acquires it.
        self._execution_lock.acquire()
        self.fired: list[str] = []

    def wait_for_turn(self, execution_name: str, marker_name: str, *, _reacquire_execution_lock: bool = False) -> None:
        self.fired.append(marker_name)
        if _reacquire_execution_lock:
            self._execution_lock.acquire()

    def report_error(self, error: Exception) -> None:  # pragma: no cover - not expected
        self.error = error


class _FakeRegistry:
    """Marker registry with explicit (line -> name) markers and source lines."""

    def __init__(self, filename: str, markers: dict[int, str], source_lines: dict[int, str]) -> None:
        self.filename = filename
        self._markers = markers
        self._source = source_lines

    def scan_frame(self, frame: Any) -> None:
        pass

    def get_marker(self, filename: str, lineno: int) -> str | None:
        return self._markers.get(lineno)


def _frame(filename: str, lineno: int) -> Any:
    return SimpleNamespace(f_code=SimpleNamespace(co_filename=filename), f_lineno=lineno)


def test_prev_line_marker_not_fired_for_skipped_code_line():
    """A marker on a skipped executable line must NOT fire via the next line."""
    fname = "fake.py"
    # Line 11 has executable code with a marker; it is SKIPPED (if not taken).
    # Line 12 is the next executed line.  The marker must not fire.
    registry = MarkerRegistry()
    registry._markers[(fname, 11)] = "skipped_marker"  # type: ignore[attr-defined]
    registry._scanned_files.add(fname)  # type: ignore[attr-defined]
    # Make the source available so the runtime can tell line 11 is real code.
    import linecache

    linecache.cache[fname] = (  # type: ignore[assignment]
        0,
        None,
        [
            "def f():\n",
            "    if cond:\n",  # placeholder lines 1..10
            "        pass\n",
            "    a\n",
            "    b\n",
            "    c\n",
            "    d\n",
            "    e\n",
            "    g\n",
            "    h\n",
            "        x = 1  # frontrun: skipped_marker\n",  # line 11 (executable)
            "    y = 2\n",  # line 12
        ],
        fname,
    )

    coord = _RecordingCoordinator()
    trace = build_trace_function(coord, registry, "t1", include_previous_line=True)

    # Simulate: we land on line 12 (line 11 was skipped, never traced).
    trace(_frame(fname, 12), "line", None)

    assert "skipped_marker" not in coord.fired, (
        "prev-line marker fired for a skipped executable line (line 11 never ran)"
    )


def test_prev_line_marker_fired_for_comment_only_line():
    """The legitimate case: marker on a comment-only line above an await."""
    fname = "fake2.py"
    registry = MarkerRegistry()
    registry._markers[(fname, 2)] = "read_balance"  # type: ignore[attr-defined]
    registry._scanned_files.add(fname)  # type: ignore[attr-defined]
    import linecache

    linecache.cache[fname] = (  # type: ignore[assignment]
        0,
        None,
        [
            "async def transfer():\n",  # line 1
            "    # frontrun: read_balance\n",  # line 2 (comment only)
            "    current = await get()\n",  # line 3 (executable)
        ],
        fname,
    )

    coord = _RecordingCoordinator()
    trace = build_trace_function(coord, registry, "t1", include_previous_line=True)

    # Land on line 3; the comment marker on line 2 must fire.
    trace(_frame(fname, 3), "line", None)

    assert "read_balance" in coord.fired, "legitimate comment-line marker did not fire"
