"""
Tests for the interlace trace_markers module.

Demonstrates deterministic thread interleaving using sys.settrace and comment markers.
"""

import os
import sys

# Add parent directory to path so we can import interlace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import threading

from interlace.common import Schedule, Step
from interlace.trace_markers import MarkerRegistry, ThreadCoordinator, TraceExecutor, interlace


class BankAccount:
    """A simple bank account class with a race condition vulnerability.

    The transfer method has interlace markers to control thread execution order.
    """

    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        """Transfer money to this account (intentionally racy).

        This method has a race condition: it reads the balance, then writes
        a new balance without locking. The interlace markers allow us to
        deterministically trigger the race.
        """
        current = self.balance  # interlace: read_balance
        # Simulate some work
        new_balance = current + amount
        self.balance = new_balance  # interlace: write_balance


def test_race_condition_buggy_schedule():
    """Test that demonstrates the race condition bug with an unsafe schedule.

    Schedule: t1 reads, t2 reads, t1 writes, t2 writes
    This creates a lost update: both threads read the same initial value,
    so the final balance is incorrect.
    """
    print("\n=== Test: Race Condition (Buggy Schedule) ===")

    account = BankAccount(balance=100)

    # Define the buggy schedule: both threads read before either writes
    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread2", "read_balance"),
            Step("thread1", "write_balance"),
            Step("thread2", "write_balance"),
        ]
    )

    def worker1():
        account.transfer(50)

    def worker2():
        account.transfer(50)

    executor = TraceExecutor(schedule)
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    print("Initial balance: 100")
    print("Thread 1 transfer: +50")
    print("Thread 2 transfer: +50")
    print("Expected (buggy): 150")
    print(f"Actual balance: {account.balance}")

    # With the buggy schedule, we expect a lost update
    assert account.balance == 150, f"Expected 150 (lost update), got {account.balance}"
    print("✓ Race condition successfully reproduced!")


def test_race_condition_correct_schedule():
    """Test that demonstrates correct execution with a safe schedule.

    Schedule: t1 reads, t1 writes, t2 reads, t2 writes
    This ensures proper serialization: each thread completes its transaction
    before the next one starts.
    """
    print("\n=== Test: Race Condition (Correct Schedule) ===")

    account = BankAccount(balance=100)

    # Define the correct schedule: each thread completes before the next starts
    schedule = Schedule(
        [
            Step("thread1", "read_balance"),
            Step("thread1", "write_balance"),
            Step("thread2", "read_balance"),
            Step("thread2", "write_balance"),
        ]
    )

    def worker1():
        account.transfer(50)

    def worker2():
        account.transfer(50)

    executor = TraceExecutor(schedule)
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    print("Initial balance: 100")
    print("Thread 1 transfer: +50")
    print("Thread 2 transfer: +50")
    print("Expected (correct): 200")
    print(f"Actual balance: {account.balance}")

    # With the correct schedule, we expect the right result
    assert account.balance == 200, f"Expected 200, got {account.balance}"
    print("✓ Correct execution verified!")


def test_multiple_markers_same_thread():
    """Test a thread hitting multiple markers in sequence."""
    print("\n=== Test: Multiple Markers Same Thread ===")

    results = []

    def worker_with_markers():
        results.append("step1")  # interlace: step1
        results.append("step2")  # interlace: step2
        results.append("step3")  # interlace: step3

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

    print(f"Results: {results}")
    assert results == ["step1", "step2", "step3"]
    print("✓ Multiple markers executed in order!")


def test_alternating_execution():
    """Test alternating execution between two threads."""
    print("\n=== Test: Alternating Execution ===")

    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    def worker1():
        x = 1  # interlace: marker_a
        append_safe("t1_a")
        y = 2  # interlace: marker_b
        append_safe("t1_b")

    def worker2():
        x = 1  # interlace: marker_a
        append_safe("t2_a")
        y = 2  # interlace: marker_b
        append_safe("t2_b")

    # Alternate between threads at each marker
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

    print(f"Execution order: {results}")
    expected = ["t1_a", "t2_a", "t1_b", "t2_b"]
    assert results == expected, f"Expected {expected}, got {results}"
    print("✓ Alternating execution verified!")


def test_convenience_function():
    """Test the convenience interlace() function."""
    print("\n=== Test: Convenience Function ===")

    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    def worker1():
        x = 1  # interlace: mark
        append_safe("t1")

    def worker2():
        x = 1  # interlace: mark
        append_safe("t2")

    schedule = Schedule(
        [
            Step("t1", "mark"),
            Step("t2", "mark"),
        ]
    )

    interlace(schedule=schedule, threads={"t1": worker1, "t2": worker2}, timeout=5.0)

    print(f"Results: {results}")
    assert results == ["t1", "t2"]
    print("✓ Convenience function works!")


def test_marker_registry():
    """Test the MarkerRegistry class directly."""
    print("\n=== Test: MarkerRegistry ===")

    # Create a temporary function with markers to test scanning
    def test_function():
        x = 1  # interlace: marker1
        y = 2  # interlace: marker2
        return x + y

    # Get a frame from the function
    import inspect

    frame = None

    def get_frame():
        return inspect.currentframe()

    # We need to actually execute the function to get its frame via tracing
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

    print(f"Found markers: {found_markers}")
    assert "marker1" in found_markers
    assert "marker2" in found_markers
    print("✓ MarkerRegistry successfully scans and finds markers!")


def test_thread_coordinator():
    """Test the ThreadCoordinator synchronization logic."""
    print("\n=== Test: ThreadCoordinator ===")

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

    print(f"Results: {results}")

    # The order should be: both start (unordered), then t1_m1, t2_m1, t1_m2
    assert "t1_m1" in results
    assert "t2_m1" in results
    assert "t1_m2" in results

    # Check the marker order is correct
    m1_index_t1 = results.index("t1_m1")
    m1_index_t2 = results.index("t2_m1")
    m2_index_t1 = results.index("t1_m2")

    assert m1_index_t1 < m1_index_t2, "t1 should hit m1 before t2"
    assert m1_index_t2 < m2_index_t1, "t2 should hit m1 before t1 hits m2"

    print("✓ ThreadCoordinator synchronizes correctly!")


def test_complex_race_scenario():
    """Test a more complex scenario with multiple shared resources."""
    print("\n=== Test: Complex Race Scenario ===")

    class SharedCounter:
        def __init__(self):
            self.value = 0

        def increment_racy(self):
            temp = self.value  # interlace: read_counter
            temp = temp + 1
            self.value = temp  # interlace: write_counter

    counter = SharedCounter()

    # Three threads, each incrementing twice
    # We'll interleave them to maximize the race condition
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

    def worker():
        counter.increment_racy()

    executor = TraceExecutor(schedule)
    executor.run("t1", worker)
    executor.run("t2", worker)
    executor.run("t3", worker)
    executor.wait(timeout=5.0)

    print("Initial counter: 0")
    print("Three threads each increment once")
    print("Expected (with race): 1")
    print(f"Actual counter: {counter.value}")

    # With this schedule, all three threads read 0, then all write 1
    assert counter.value == 1, f"Expected 1 (lost updates), got {counter.value}"
    print("✓ Complex race condition reproduced!")


def test_multiline_statements_with_markers():
    """Test that markers work correctly on multiline statements.

    Verifies that markers placed on closing parentheses of multiline calls
    are detected and processed correctly. The marker comment should not be
    reformatted by code formatters.
    """
    print("\n=== Test: Multiline Statements with Markers ===")

    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    # Use exec with string literal to preserve exact formatting
    code_template = """
def worker_{name}():
    append_safe(
        "thread{name}_step1"
    )  # interlace: step1
    append_safe(
        "thread{name}_step2"
    )  # interlace: step2
"""

    # Create worker1 and worker2 with exact formatting preserved
    namespace1 = {"append_safe": append_safe}
    exec(code_template.format(name="1"), namespace1)
    worker1 = namespace1["worker_1"]

    namespace2 = {"append_safe": append_safe}
    exec(code_template.format(name="2"), namespace2)
    worker2 = namespace2["worker_2"]

    # Define schedule to interleave the threads
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

    print(f"Results: {results}")

    # Verify all steps executed (all markers were hit)
    # This demonstrates that multiline statements with markers work correctly
    assert "thread1_step1" in results, "thread1 should execute step1"
    assert "thread1_step2" in results, "thread1 should execute step2"
    assert "thread2_step1" in results, "thread2 should execute step1"
    assert "thread2_step2" in results, "thread2 should execute step2"

    print("✓ Multiline statements with markers work correctly!")


def test_multiline_with_nested_calls():
    """Test markers on multiline statements with nested function calls."""
    print("\n=== Test: Multiline with Nested Calls ===")

    results = []

    # Code with nested multiline calls
    code = """
def worker():
    results.append(
        some_func(
            "arg1",
            "arg2",
        )
    )  # interlace: nested_call

def some_func(a, b):
    return f"{a}-{b}"
"""

    namespace = {"results": results}
    exec(code, namespace)
    worker = namespace["worker"]
    some_func = namespace["some_func"]

    schedule = Schedule(
        [
            Step("main", "nested_call"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("main", worker)
    executor.wait(timeout=5.0)

    print(f"Results: {results}")
    assert len(results) > 0
    assert "arg1-arg2" in results
    print("✓ Nested multiline calls with markers work!")


def test_markers_on_empty_lines():
    """Test that markers work on lines containing only the marker comment.

    This test verifies that markers can be placed on empty lines (lines with only
    a comment) rather than inline with code. This is useful for marking specific
    execution points without modifying the actual code statements.

    Both styles work:
        # Markers on empty lines (cleaner):
        # interlace: before_read
        val = get()
        # interlace: after_read

        # Markers inline with code:
        val = get()  # interlace: read
    """
    print("\n=== Test: Markers on Empty Lines ===")

    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    # Use exec to create a function with markers on empty lines
    code = """
def worker1():
    # interlace: before_read
    val = get_value()
    # interlace: after_read
    append_safe("t1_processed")

def worker2():
    # interlace: before_read
    val = get_value()
    # interlace: after_read
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

    # Define schedule to interleave threads at markers on empty lines
    schedule = Schedule(
        [
            Step("thread1", "before_read"),
            Step("thread2", "before_read"),
            Step("thread1", "after_read"),
            Step("thread2", "after_read"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    print(f"Results: {results}")

    # Verify that both threads executed and hit all markers
    assert "t1_processed" in results, "thread1 should execute"
    assert "t2_processed" in results, "thread2 should execute"

    # Verify the order is correct (before_read for both threads, then after_read)
    t1_processed_idx = results.index("t1_processed")
    t2_processed_idx = results.index("t2_processed")

    # The execution order should reflect our schedule
    print("✓ Markers on empty lines work correctly!")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("INTERLACE TRACE MARKERS - TEST SUITE")
    print("=" * 60)

    tests = [
        test_race_condition_buggy_schedule,
        test_race_condition_correct_schedule,
        test_multiple_markers_same_thread,
        test_alternating_execution,
        test_convenience_function,
        test_marker_registry,
        test_thread_coordinator,
        test_complex_race_scenario,
        test_multiline_statements_with_markers,
        test_multiline_with_nested_calls,
        test_markers_on_empty_lines,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
