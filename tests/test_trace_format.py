"""Tests for trace recording, filtering, and formatting.

Covers:
- TraceEvent / TraceRecorder
- Filtering and deduplication
- Conflict classification
- Full format_trace output
- Integration with explore_interleavings and explore_dpor
"""

from __future__ import annotations

from frontrun._trace_format import (
    SourceLineEvent,
    TraceEvent,
    TraceRecorder,
    classify_conflict,
    deduplicate_to_source_lines,
    filter_to_shared_accesses,
    format_trace,
)

# ---------------------------------------------------------------------------
# TraceEvent / TraceRecorder unit tests
# ---------------------------------------------------------------------------


class TestTraceRecorder:
    def test_record_basic(self) -> None:
        """Recording an event captures frame info."""

        class FakeCode:
            co_filename = "counter.py"
            co_name = "increment"

        class FakeFrame:
            f_code = FakeCode()
            f_lineno = 10
            f_lasti = 0

        recorder = TraceRecorder()
        recorder.record(thread_id=0, frame=FakeFrame(), opcode="LOAD_ATTR", access_type="read", attr_name="value")

        assert len(recorder.events) == 1
        ev = recorder.events[0]
        assert ev.thread_id == 0
        assert ev.filename == "counter.py"
        assert ev.lineno == 10
        assert ev.function_name == "increment"
        assert ev.opcode == "LOAD_ATTR"
        assert ev.access_type == "read"
        assert ev.attr_name == "value"

    def test_record_captures_obj_type(self) -> None:
        """Object type name is captured from the object."""

        class Counter:
            value = 0

        class FakeCode:
            co_filename = "counter.py"
            co_name = "increment"

        class FakeFrame:
            f_code = FakeCode()
            f_lineno = 10
            f_lasti = 0

        recorder = TraceRecorder()
        recorder.record(
            thread_id=0, frame=FakeFrame(), opcode="LOAD_ATTR", access_type="read", attr_name="value", obj=Counter()
        )

        assert recorder.events[0].obj_type_name == "Counter"

    def test_disabled_recorder_skips(self) -> None:
        """When disabled, record() is a no-op."""

        class FakeCode:
            co_filename = "test.py"
            co_name = "f"

        class FakeFrame:
            f_code = FakeCode()
            f_lineno = 1
            f_lasti = 0

        recorder = TraceRecorder(enabled=False)
        recorder.record(thread_id=0, frame=FakeFrame(), opcode="LOAD_ATTR", access_type="read")
        assert len(recorder.events) == 0

    def test_step_index_increments(self) -> None:
        """Each recorded event gets a sequential step index."""

        class FakeCode:
            co_filename = "test.py"
            co_name = "f"

        class FakeFrame:
            f_code = FakeCode()
            f_lineno = 1
            f_lasti = 0

        recorder = TraceRecorder()
        recorder.record(thread_id=0, frame=FakeFrame(), opcode="LOAD_ATTR", access_type="read")
        recorder.record(thread_id=1, frame=FakeFrame(), opcode="LOAD_ATTR", access_type="read")
        recorder.record(thread_id=0, frame=FakeFrame(), opcode="STORE_ATTR", access_type="write")

        assert [ev.step_index for ev in recorder.events] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_filter_keeps_shared_accesses(self) -> None:
        events = [
            TraceEvent(0, 0, "test.py", 1, "f", "LOAD_FAST"),
            TraceEvent(1, 0, "test.py", 2, "f", "LOAD_ATTR", access_type="read", attr_name="value"),
            TraceEvent(2, 0, "test.py", 3, "f", "POP_TOP"),
            TraceEvent(3, 0, "test.py", 4, "f", "STORE_ATTR", access_type="write", attr_name="value"),
        ]
        filtered = filter_to_shared_accesses(events)
        assert len(filtered) == 2
        assert filtered[0].opcode == "LOAD_ATTR"
        assert filtered[1].opcode == "STORE_ATTR"

    def test_filter_empty_list(self) -> None:
        assert filter_to_shared_accesses([]) == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_merge_same_line(self) -> None:
        """Multiple events on the same line from the same thread merge."""
        events = [
            TraceEvent(0, 0, "test.py", 10, "f", "LOAD_ATTR", access_type="read", attr_name="value"),
            TraceEvent(1, 0, "test.py", 10, "f", "STORE_ATTR", access_type="write", attr_name="value"),
        ]
        lines = deduplicate_to_source_lines(events)
        assert len(lines) == 1
        assert lines[0].access_type == "read+write"

    def test_different_threads_not_merged(self) -> None:
        """Same line from different threads produces separate entries."""
        events = [
            TraceEvent(0, 0, "test.py", 10, "f", "LOAD_ATTR", access_type="read", attr_name="value"),
            TraceEvent(1, 1, "test.py", 10, "f", "LOAD_ATTR", access_type="read", attr_name="value"),
        ]
        lines = deduplicate_to_source_lines(events)
        assert len(lines) == 2
        assert lines[0].thread_id == 0
        assert lines[1].thread_id == 1

    def test_different_lines_not_merged(self) -> None:
        """Different lines from the same thread produce separate entries."""
        events = [
            TraceEvent(0, 0, "test.py", 10, "f", "LOAD_ATTR", access_type="read", attr_name="value"),
            TraceEvent(1, 0, "test.py", 11, "f", "STORE_ATTR", access_type="write", attr_name="value"),
        ]
        lines = deduplicate_to_source_lines(events)
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Conflict classification
# ---------------------------------------------------------------------------


class TestConflictClassification:
    def test_lost_update_detected(self) -> None:
        """Classic R_a R_b W_a W_b pattern is classified as lost update."""
        lines = [
            SourceLineEvent(0, "test.py", 10, "inc", "temp = self.value", "read", "value", "Counter"),
            SourceLineEvent(1, "test.py", 10, "inc", "temp = self.value", "read", "value", "Counter"),
            SourceLineEvent(0, "test.py", 11, "inc", "self.value = temp + 1", "write", "value", "Counter"),
            SourceLineEvent(1, "test.py", 11, "inc", "self.value = temp + 1", "write", "value", "Counter"),
        ]
        conflict = classify_conflict(lines)
        assert conflict.pattern == "lost_update"
        assert "0" in conflict.summary and "1" in conflict.summary
        assert "value" in conflict.summary

    def test_write_write_detected(self) -> None:
        """Two threads writing without reads is a write-write conflict."""
        lines = [
            SourceLineEvent(0, "test.py", 10, "f", "self.x = 1", "write", "x", "State"),
            SourceLineEvent(1, "test.py", 10, "f", "self.x = 2", "write", "x", "State"),
        ]
        conflict = classify_conflict(lines)
        assert conflict.pattern == "write_write"

    def test_empty_events(self) -> None:
        conflict = classify_conflict([])
        assert conflict.pattern == "unknown"

    def test_single_thread_no_conflict(self) -> None:
        """Accesses from a single thread don't constitute a conflict."""
        lines = [
            SourceLineEvent(0, "test.py", 10, "f", "temp = self.value", "read", "value", "Counter"),
            SourceLineEvent(0, "test.py", 11, "f", "self.value = temp + 1", "write", "value", "Counter"),
        ]
        conflict = classify_conflict(lines)
        assert conflict.pattern == "unknown"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_basic_format(self) -> None:
        """format_trace produces readable output for a lost-update race."""
        events = [
            TraceEvent(0, 0, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(1, 1, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(2, 0, "counter.py", 11, "increment", "STORE_ATTR", "write", "value", "Counter"),
            TraceEvent(3, 1, "counter.py", 11, "increment", "STORE_ATTR", "write", "value", "Counter"),
        ]
        output = format_trace(events, num_threads=2, num_explored=3)
        assert "Race condition found after 3 interleavings" in output
        assert "Thread 0" in output
        assert "Thread 1" in output
        assert "Lost update" in output or "lost_update" in output.lower()
        assert "[read" in output
        assert "[write" in output

    def test_format_with_thread_names(self) -> None:
        events = [
            TraceEvent(0, 0, "bank.py", 5, "transfer", "LOAD_ATTR", "read", "balance", "Account"),
            TraceEvent(1, 1, "bank.py", 5, "transfer", "LOAD_ATTR", "read", "balance", "Account"),
        ]
        output = format_trace(events, num_threads=2, thread_names=["depositor", "withdrawer"])
        assert "depositor" in output
        assert "withdrawer" in output

    def test_format_no_shared_accesses(self) -> None:
        """When no shared accesses recorded, show fallback message."""
        events = [
            TraceEvent(0, 0, "test.py", 1, "f", "LOAD_FAST"),
        ]
        output = format_trace(events, num_threads=2, num_explored=5)
        assert "no shared-state accesses" in output

    def test_format_empty_events(self) -> None:
        output = format_trace([], num_threads=2)
        assert "no shared-state accesses" in output

    def test_show_opcodes_flag(self) -> None:
        """When show_opcodes=True, opcode details are included."""
        events = [
            TraceEvent(0, 0, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(1, 0, "counter.py", 10, "increment", "STORE_ATTR", "write", "value", "Counter"),
        ]
        output = format_trace(events, num_threads=1, show_opcodes=True)
        assert "LOAD_ATTR" in output
        assert "STORE_ATTR" in output

    def test_reproduction_stats_shown(self) -> None:
        """Reproduction stats are included in the output when provided."""
        events = [
            TraceEvent(0, 0, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(1, 1, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(2, 0, "counter.py", 11, "increment", "STORE_ATTR", "write", "value", "Counter"),
            TraceEvent(3, 1, "counter.py", 11, "increment", "STORE_ATTR", "write", "value", "Counter"),
        ]
        output = format_trace(events, num_threads=2, num_explored=3, reproduction_attempts=10, reproduction_successes=7)
        assert "7/10" in output
        assert "70%" in output

    def test_reproduction_stats_not_shown_when_zero(self) -> None:
        """When reproduction_attempts=0, no reproduction line is shown."""
        events = [
            TraceEvent(0, 0, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(1, 1, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
        ]
        output = format_trace(events, num_threads=2, reproduction_attempts=0, reproduction_successes=0)
        assert "Reproduced" not in output

    def test_reproduction_stats_zero_successes(self) -> None:
        """When the race never reproduces, show 0/N."""
        events = [
            TraceEvent(0, 0, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(1, 1, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
        ]
        output = format_trace(events, num_threads=2, reproduction_attempts=10, reproduction_successes=0)
        assert "0/10" in output
        assert "0%" in output


# ---------------------------------------------------------------------------
# Integration tests: explore_interleavings produces explanation
# ---------------------------------------------------------------------------


class TestBytecodeIntegration:
    def test_explore_counter_has_explanation(self) -> None:
        """explore_interleavings should produce an explanation for a counter race."""
        from frontrun.bytecode import explore_interleavings

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_interleavings(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_attempts=200,
            max_ops=200,
            seed=42,
        )

        assert not result.property_holds
        assert result.explanation is not None
        assert "Race condition found" in result.explanation
        # Should mention the relevant attribute
        assert "value" in result.explanation

    def test_explore_counter_reproduction_stats(self) -> None:
        """explore_interleavings should report reproduction stats."""
        from frontrun.bytecode import explore_interleavings

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_interleavings(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_attempts=200,
            max_ops=200,
            seed=42,
            reproduce_on_failure=5,
        )

        assert not result.property_holds
        assert result.reproduction_attempts == 5
        assert 0 <= result.reproduction_successes <= 5
        # The explanation should include the reproduction line
        assert result.explanation is not None
        assert "/5" in result.explanation

    def test_explore_counter_skip_reproduction(self) -> None:
        """reproduce_on_failure=0 skips reproduction testing."""
        from frontrun.bytecode import explore_interleavings

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_interleavings(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_attempts=200,
            max_ops=200,
            seed=42,
            reproduce_on_failure=0,
        )

        assert not result.property_holds
        assert result.reproduction_attempts == 0
        assert result.reproduction_successes == 0
        # The explanation should NOT include a reproduction line
        assert result.explanation is not None
        assert "Reproduced" not in result.explanation

    def test_flaky_race_reproduction_rate(self) -> None:
        """A flaky race (random.random()) should have reproduction rate well below 100%."""
        import random as stdlib_random

        from frontrun.bytecode import explore_interleavings

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        def flaky_invariant(c: Counter) -> bool:
            if c.value == 2:
                return True
            # Bug present, but randomly ignore it ~50% of the time
            return stdlib_random.random() < 0.5

        result = explore_interleavings(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=flaky_invariant,
            max_attempts=200,
            max_ops=200,
            seed=42,
            reproduce_on_failure=100,
        )

        assert not result.property_holds
        assert result.reproduction_attempts == 100
        # Reproduction rate should be ~50%, certainly not 100% or 0%
        assert 10 < result.reproduction_successes < 90, (
            f"Expected ~50% reproduction rate, got {result.reproduction_successes}/100"
        )
        assert result.explanation is not None
        assert "/100" in result.explanation

    def test_safe_counter_no_explanation(self) -> None:
        """When no race is found, explanation should be None."""
        import threading

        from frontrun.bytecode import explore_interleavings

        class SafeCounter:
            def __init__(self) -> None:
                self.value = 0
                self._lock = threading.Lock()

            def increment(self) -> None:
                with self._lock:
                    temp = self.value
                    self.value = temp + 1

        result = explore_interleavings(
            setup=SafeCounter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_attempts=50,
            max_ops=200,
            seed=42,
        )

        assert result.property_holds
        assert result.explanation is None
        assert result.reproduction_attempts == 0
        assert result.reproduction_successes == 0


# ---------------------------------------------------------------------------
# Integration tests: explore_dpor produces explanation
# ---------------------------------------------------------------------------


class TestDporIntegration:
    def test_dpor_counter_has_explanation(self) -> None:
        """explore_dpor should produce an explanation for a counter race."""
        from frontrun.dpor import explore_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds
        assert result.explanation is not None
        assert "Race condition found" in result.explanation
        assert "value" in result.explanation

    def test_dpor_counter_reproduction_stats(self) -> None:
        """explore_dpor should report reproduction stats."""
        from frontrun.dpor import explore_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=500,
            preemption_bound=2,
            reproduce_on_failure=5,
        )

        assert not result.property_holds
        assert result.reproduction_attempts == 5
        assert 0 <= result.reproduction_successes <= 5
        assert result.explanation is not None
        assert "/5" in result.explanation

    def test_dpor_safe_counter_no_explanation(self) -> None:
        """When DPOR finds no race, explanation should be None."""
        import threading

        from frontrun.dpor import explore_dpor

        class LockedCounter:
            def __init__(self) -> None:
                self.value = 0
                self.lock = threading.Lock()

            def increment(self) -> None:
                with self.lock:
                    temp = self.value
                    self.value = temp + 1

        result = explore_dpor(
            setup=LockedCounter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=50,
            preemption_bound=2,
        )

        assert result.property_holds
        assert result.explanation is None
        assert result.reproduction_attempts == 0
        assert result.reproduction_successes == 0
