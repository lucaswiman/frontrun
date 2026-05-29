"""Tests for the frontrun trace_markers module."""

import sys
import threading
import time

import pytest

from frontrun.common import Schedule, Step
from frontrun.trace_markers import MarkerRegistry, ThreadCoordinator, TraceExecutor, frontrun


class BankAccount:
    """A simple bank account class with a race condition vulnerability."""

    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        current = self.balance  # frontrun: read_balance
        new_balance = current + amount
        self.balance = new_balance  # frontrun: write_balance


def test_race_condition_buggy_schedule():
    """Both threads read before either writes, causing a lost update."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread2", "read_balance"),
            Step("thread1", "write_balance"),
            Step("thread2", "write_balance"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run({"thread1": lambda: account.transfer(50), "thread2": lambda: account.transfer(50)}, timeout=5.0)

    assert account.balance == 150


def test_race_condition_correct_schedule():
    """Each thread completes its transaction before the next starts."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread1", "write_balance"),
            Step("thread2", "read_balance"),
            Step("thread2", "write_balance"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run({"thread1": lambda: account.transfer(50), "thread2": lambda: account.transfer(50)}, timeout=5.0)

    assert account.balance == 200


def test_multiple_markers_same_thread():
    """A thread hitting multiple markers in sequence."""
    results = []

    def worker_with_markers():
        results.append("step1")  # frontrun: step1
        results.append("step2")  # frontrun: step2
        results.append("step3")  # frontrun: step3

    schedule = Schedule(
        [
            Step("main", "step1"),
            Step("main", "step2"),
            Step("main", "step3"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run({"main": worker_with_markers}, timeout=5.0)

    assert results == ["step1", "step2", "step3"]


def test_alternating_execution():
    """Alternating execution between two threads."""
    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    def worker1():
        x = 1  # frontrun: marker_a
        append_safe("t1_a")
        y = 2  # frontrun: marker_b
        append_safe("t1_b")

    def worker2():
        x = 1  # frontrun: marker_a
        append_safe("t2_a")
        y = 2  # frontrun: marker_b
        append_safe("t2_b")

    schedule = Schedule(
        [
            Step("thread1", "marker_a"),
            Step("thread2", "marker_a"),
            Step("thread1", "marker_b"),
            Step("thread2", "marker_b"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run({"thread1": worker1, "thread2": worker2}, timeout=5.0)

    assert results == ["t1_a", "t2_a", "t1_b", "t2_b"]


def test_convenience_function():
    """The frontrun() convenience function."""
    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    def worker1():
        x = 1  # frontrun: mark
        append_safe("t1")

    def worker2():
        x = 1  # frontrun: mark
        append_safe("t2")

    schedule = Schedule(
        [
            Step("t1", "mark"),
            Step("t2", "mark"),
        ]
    )

    frontrun(schedule=schedule, threads={"t1": worker1, "t2": worker2}, timeout=5.0)

    assert results == ["t1", "t2"]


def test_marker_registry():
    """MarkerRegistry scans frames and finds markers."""

    def test_function():
        x = 1  # frontrun: marker1
        y = 2  # frontrun: marker2
        return x + y

    registry = MarkerRegistry()
    found_markers = []

    def trace_func(frame, event, arg):
        if event == "line":
            registry.scan_frame(frame)
            marker = registry.get_marker(frame.f_code.co_filename, frame.f_lineno)
            if marker:
                found_markers.append(marker)
        return trace_func

    sys.settrace(trace_func)
    try:
        test_function()
    finally:
        sys.settrace(None)

    assert "marker1" in found_markers
    assert "marker2" in found_markers


def test_thread_coordinator():
    """ThreadCoordinator synchronizes threads in schedule order."""
    schedule = Schedule(
        [
            Step("t1", "m1"),
            Step("t2", "m1"),
            Step("t1", "m2"),
        ]
    )

    coordinator = ThreadCoordinator(schedule)
    results = []

    def thread1_work():
        results.append("t1_start")
        coordinator.wait_for_turn("t1", "m1")
        results.append("t1_m1")
        coordinator.wait_for_turn("t1", "m2")
        results.append("t1_m2")

    def thread2_work():
        results.append("t2_start")
        coordinator.wait_for_turn("t2", "m1")
        results.append("t2_m1")

    t1 = threading.Thread(target=thread1_work, daemon=True)
    t2 = threading.Thread(target=thread2_work, daemon=True)

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert "t1_m1" in results
    assert "t2_m1" in results
    assert "t1_m2" in results

    m1_index_t1 = results.index("t1_m1")
    m1_index_t2 = results.index("t2_m1")
    m2_index_t1 = results.index("t1_m2")

    assert m1_index_t1 < m1_index_t2, "t1 should hit m1 before t2"
    assert m1_index_t2 < m2_index_t1, "t2 should hit m1 before t1 hits m2"


def test_complex_race_scenario():
    """Three threads all read before any writes, causing maximum lost updates."""

    class SharedCounter:
        def __init__(self):
            self.value = 0

        def increment_racy(self):
            temp = self.value  # frontrun: read_counter
            temp = temp + 1
            self.value = temp  # frontrun: write_counter

    counter = SharedCounter()

    schedule = Schedule(
        [
            Step("t1", "read_counter"),
            Step("t2", "read_counter"),
            Step("t3", "read_counter"),
            Step("t1", "write_counter"),
            Step("t2", "write_counter"),
            Step("t3", "write_counter"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run(
        {"t1": counter.increment_racy, "t2": counter.increment_racy, "t3": counter.increment_racy}, timeout=5.0
    )

    # All three threads read 0, then all write 1
    assert counter.value == 1


def test_multiline_statements_with_markers():
    """Markers on closing parentheses of multiline calls work correctly."""
    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    code_template = """
def worker_{name}():
    append_safe(
        "thread{name}_step1"
    )  # frontrun: step1
    append_safe(
        "thread{name}_step2"
    )  # frontrun: step2
"""

    namespace1 = {"append_safe": append_safe}
    exec(code_template.format(name="1"), namespace1)
    worker1 = namespace1["worker_1"]

    namespace2 = {"append_safe": append_safe}
    exec(code_template.format(name="2"), namespace2)
    worker2 = namespace2["worker_2"]

    schedule = Schedule(
        [
            Step("thread1", "step1"),
            Step("thread2", "step1"),
            Step("thread1", "step2"),
            Step("thread2", "step2"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run({"thread1": worker1, "thread2": worker2}, timeout=5.0)

    assert "thread1_step1" in results
    assert "thread1_step2" in results
    assert "thread2_step1" in results
    assert "thread2_step2" in results


def test_multiline_with_nested_calls():
    """Markers on multiline statements with nested function calls."""
    results = []

    code = """
def worker():
    results.append(
        some_func(
            "arg1",
            "arg2",
        )
    )  # frontrun: nested_call

def some_func(a, b):
    return f"{a}-{b}"
"""

    namespace = {"results": results}
    exec(code, namespace)
    worker = namespace["worker"]

    schedule = Schedule(
        [
            Step("main", "nested_call"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run({"main": worker}, timeout=5.0)

    assert "arg1-arg2" in results


def test_markers_on_standalone_lines():
    """Markers on lines containing only the marker comment (not inline with code).

    Both styles work:
        # Inline with code:
        val = get()  # frontrun: read_value

        # Standalone line (marker gates the next statement):
        # frontrun: read_value
        val = get()
    """
    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    code = """
def worker1():
    # frontrun: read_value
    val = get_value()
    # frontrun: process_value
    append_safe("t1_processed")

def worker2():
    # frontrun: read_value
    val = get_value()
    # frontrun: process_value
    append_safe("t2_processed")

def get_value():
    return 42
"""

    namespace = {
        "append_safe": append_safe,
    }
    exec(code, namespace)
    worker1 = namespace["worker1"]
    worker2 = namespace["worker2"]

    schedule = Schedule(
        [
            Step("thread1", "read_value"),
            Step("thread2", "read_value"),
            Step("thread1", "process_value"),
            Step("thread2", "process_value"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run({"thread1": worker1, "thread2": worker2}, timeout=5.0)

    assert "t1_processed" in results
    assert "t2_processed" in results


@pytest.mark.intentionally_leaves_dangling_threads
def test_wait_timeout_is_total_not_per_thread():
    """wait(timeout=T) should wait at most ~T seconds total, not T per thread."""

    def slow_worker():
        time.sleep(10)

    schedule = Schedule([Step("t1", "never"), Step("t2", "never")])
    executor = TraceExecutor(schedule)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.run({"t1": slow_worker, "t2": slow_worker}, timeout=1.0)
    elapsed = time.monotonic() - start
    # With per-thread timeout, elapsed would be ~2s (1s per thread).
    # With a proper total deadline, it should be ~1s.
    assert elapsed < 1.8, f"wait() took {elapsed:.1f}s — timeout is per-thread, not total"


def test_trace_runtime_clears_trace_and_reports_errors(monkeypatch):
    """Shared trace-runtime cleanup should clear tracing and report errors."""

    from frontrun._trace_marker_runtime import run_traced_callable

    trace_calls: list[object | None] = []

    def fake_settrace(value):
        trace_calls.append(value)

    monkeypatch.setattr(sys, "settrace", fake_settrace)

    class FakeLock:
        def __init__(self):
            self.locked = False

        def acquire(self):
            assert not self.locked
            self.locked = True

        def release(self):
            assert self.locked
            self.locked = False

    class FakeCoordinator:
        def __init__(self):
            self._execution_lock = FakeLock()
            self.reported: list[Exception] = []

        def report_error(self, error):
            self.reported.append(error)

    coordinator = FakeCoordinator()
    errors: dict[str, Exception] = {}

    def body():
        raise ValueError("boom")

    run_traced_callable(
        coordinator=coordinator,
        execution_name="worker",
        body=body,
        error_sink=errors,
    )

    assert len(trace_calls) == 2
    assert trace_calls[1] is None
    assert coordinator._execution_lock.locked is False
    assert "worker" in errors
    assert isinstance(errors["worker"], ValueError)
    assert len(coordinator.reported) == 1


# ---------------------------------------------------------------------------
# New dict-form API tests
# ---------------------------------------------------------------------------


def test_dict_form_basic():
    """TraceExecutor.run() with a dict starts all threads and waits in one call."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread2", "read_balance"),
            Step("thread1", "write_balance"),
            Step("thread2", "write_balance"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run(
        {
            "thread1": lambda: account.transfer(50),
            "thread2": lambda: account.transfer(50),
        },
        timeout=5.0,
    )

    # Both threads read 100 before either writes → lost update → 150
    assert account.balance == 150


def test_dict_form_returns_none():
    """The dict form's run() returns None (like the async form)."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread1", "write_balance"),
            Step("thread2", "read_balance"),
            Step("thread2", "write_balance"),
        ]
    )

    executor = TraceExecutor(schedule)
    result = executor.run(
        {
            "thread1": lambda: account.transfer(50),
            "thread2": lambda: account.transfer(50),
        },
        timeout=5.0,
    )

    assert result is None
    assert account.balance == 200


@pytest.mark.intentionally_leaves_dangling_threads
def test_dict_form_timeout():
    """The dict form raises TimeoutError when threads don't finish in time."""

    def slow_worker():
        time.sleep(10)

    schedule = Schedule([Step("t1", "never")])
    executor = TraceExecutor(schedule)

    with pytest.raises(TimeoutError):
        executor.run({"t1": slow_worker}, timeout=0.5)


def test_dict_form_empty_dict():
    """Passing an empty dict raises ValueError with a helpful message."""
    schedule = Schedule([Step("t1", "m")])

    executor = TraceExecutor(schedule)
    with pytest.raises(ValueError, match="empty"):
        executor.run({})


def test_dict_form_non_callable_value():
    """Passing a non-callable value in the dict raises TypeError."""
    schedule = Schedule([Step("t1", "m")])

    executor = TraceExecutor(schedule)
    with pytest.raises(TypeError, match="callable"):
        executor.run({"t1": "not_a_function"})  # type: ignore[arg-type]


def test_old_form_emits_deprecation_warning():
    """Calling run(name, fn) emits DeprecationWarning."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread1", "write_balance"),
        ]
    )

    executor = TraceExecutor(schedule)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        executor.run("thread1", lambda: account.transfer(50))
    executor.wait(timeout=5.0)

    assert account.balance == 150


def test_old_form_still_works_after_warning():
    """The deprecated individual-call form still executes correctly."""
    account = BankAccount(balance=100)

    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread2", "read_balance"),
            Step("thread1", "write_balance"),
            Step("thread2", "write_balance"),
        ]
    )

    executor = TraceExecutor(schedule)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        executor.run("thread1", lambda: account.transfer(50))
        executor.run("thread2", lambda: account.transfer(50))
    executor.wait(timeout=5.0)

    assert account.balance == 150


class TestPreviousLineMarkerNoDoubleFire:
    """Previous-line markers should not fire repeatedly on the same line."""

    def test_previous_line_marker_updates_tracker(self):
        from frontrun._marker_coordination import MarkerRegistry
        from frontrun._trace_marker_runtime import build_trace_function

        class FakeCoordinator:
            def __init__(self):
                self._execution_lock = __import__("threading").Lock()
                self.error = None
                self.marker_fires = []

            def wait_for_turn(self, execution_name, marker_name, *, _reacquire_execution_lock=False):
                self.marker_fires.append(marker_name)
                if _reacquire_execution_lock:
                    self._execution_lock.acquire()

        registry = MarkerRegistry()
        registry._markers[("test.py", 5)] = "my_marker"

        coordinator = FakeCoordinator()

        trace_fn = build_trace_function(
            coordinator,
            registry,
            "thread1",
            include_previous_line=True,
        )

        class FakeCode:
            co_filename = "test.py"

        class FakeFrame:
            f_code = FakeCode()
            f_lineno = 6
            f_locals = {}

        coordinator._execution_lock.acquire()
        trace_fn(FakeFrame(), "line", None)
        trace_fn(FakeFrame(), "line", None)

        assert len(coordinator.marker_fires) == 1, (
            f"Previous-line marker should fire only once but fired "
            f"{len(coordinator.marker_fires)} times: {coordinator.marker_fires}"
        )
