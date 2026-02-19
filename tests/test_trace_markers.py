"""Tests for the frontrun trace_markers module."""

import sys
import sysconfig
import threading

import pytest

from frontrun.common import Schedule, Step
from frontrun.trace_markers import MarkerRegistry, ThreadCoordinator, TraceExecutor, frontrun

_free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


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
    executor.run("thread1", lambda: account.transfer(50))
    executor.run("thread2", lambda: account.transfer(50))
    executor.wait(timeout=5.0)

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
    executor.run("thread1", lambda: account.transfer(50))
    executor.run("thread2", lambda: account.transfer(50))
    executor.wait(timeout=5.0)

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
    executor.run("main", worker_with_markers)
    executor.wait(timeout=5.0)

    assert results == ["step1", "step2", "step3"]


@pytest.mark.skipif(_free_threaded, reason="Flaky under free-threaded Python; see GH-XXX")
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
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

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
    executor.run("t1", counter.increment_racy)
    executor.run("t2", counter.increment_racy)
    executor.run("t3", counter.increment_racy)
    executor.wait(timeout=5.0)

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
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

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
    executor.run("main", worker)
    executor.wait(timeout=5.0)

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
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    assert "t1_processed" in results
    assert "t2_processed" in results
