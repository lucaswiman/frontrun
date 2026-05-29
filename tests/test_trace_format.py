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
    _collapse_runs,
    _merge_consecutive,
    build_call_chain,
    classify_conflict,
    deduplicate_to_source_lines,
    filter_to_shared_accesses,
    format_trace,
    qualified_name,
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

    def test_record_io_captures_detail_and_call_chain(self) -> None:
        """C-level I/O events can carry SQL detail and call chain context."""
        recorder = TraceRecorder()
        recorder.record_io(
            thread_id=0,
            resource_id="socket:unix:/var/run/postgresql/.s.PGSQL.5432",
            kind="read",
            detail="SQL: SELECT * FROM users WHERE id = 1",
            call_chain=["QuerySet.get", "AcquireUserView.post"],
        )

        ev = recorder.events[0]
        assert ev.detail == "SQL: SELECT * FROM users WHERE id = 1"
        assert ev.call_chain == ["QuerySet.get", "AcquireUserView.post"]


class TestTraceRecorderOrdering:
    def test_events_appended_in_step_order(self) -> None:
        import threading

        recorder = TraceRecorder()
        barrier = threading.Barrier(4)

        class FakeFrame:
            class f_code:  # noqa: N801
                co_filename = "test.py"
                co_name = "test_fn"

            f_lineno = 1

        def record_many(tid: int) -> None:
            barrier.wait()
            for _ in range(100):
                recorder.record(tid, FakeFrame(), "LOAD_ATTR", "read")  # type: ignore[arg-type]

        threads = [threading.Thread(target=record_many, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        steps = [e.step_index for e in recorder.events]
        assert steps == sorted(steps)


class TestCollapseRuns:
    def _make_events(self, count: int) -> list[SourceLineEvent]:
        return [
            SourceLineEvent(
                thread_id=i % 2,
                filename="test.py",
                lineno=i,
                function_name="f",
                source_line=f"line {i}",
            )
            for i in range(count)
        ]

    def test_even_max_lines_not_exceeded(self) -> None:
        result = _collapse_runs(self._make_events(100), max_lines=30)
        assert len(result) <= 30

    def test_small_max_lines_not_exceeded(self) -> None:
        assert len(_collapse_runs(self._make_events(20), max_lines=1)) <= 1
        assert len(_collapse_runs(self._make_events(20), max_lines=2)) <= 2

    def test_final_cap_counts_real_lines_not_list_entries(self) -> None:
        """When the final cap drops CollapsedRun entries, it must sum their
        counts rather than counting each as 1 line."""
        from frontrun._trace_format import CollapsedRun

        events = [
            SourceLineEvent(
                thread_id=tid,
                filename="test.py",
                lineno=tid * 100 + i,
                function_name="f",
                source_line=f"line {tid}_{i}",
            )
            for tid in range(5)
            for i in range(10)
        ]
        result = _collapse_runs(events, max_lines=10)
        total = sum(r.count if isinstance(r, CollapsedRun) else 1 for r in result)
        assert total == 50, f"CollapsedRun counts should sum to total events (50), got {total}"


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

    def test_merge_consecutive_does_not_mutate_input(self) -> None:
        """_merge_consecutive must not mutate the access_type of input SourceLineEvent objects."""
        events = [
            SourceLineEvent(
                0, "test.py", 10, "f", "x = self.value", access_type="read", attr_name="value", obj_type_name="C"
            ),
            SourceLineEvent(
                0, "test.py", 10, "f", "x = self.value", access_type="write", attr_name="value", obj_type_name="C"
            ),
        ]
        original_type = events[0].access_type
        _merge_consecutive(events)
        assert events[0].access_type == original_type, (
            f"_merge_consecutive mutated input: {original_type!r} -> {events[0].access_type!r}"
        )

    def test_merge_consecutive_does_not_mutate_nonfirst_input(self) -> None:
        """_merge_consecutive must clone non-first events before mutating access_type."""
        events = [
            SourceLineEvent(0, "test.py", 10, "f", "r1", access_type="read", attr_name="v", obj_type_name="C"),
            SourceLineEvent(0, "test.py", 20, "f", "r2", access_type="write", attr_name="v", obj_type_name="C"),
            SourceLineEvent(0, "test.py", 20, "f", "r3", access_type="read", attr_name="v", obj_type_name="C"),
        ]
        assert events[1].access_type == "write"
        _merge_consecutive(events)
        assert events[1].access_type == "write", (
            f"_merge_consecutive mutated non-first input event: 'write' -> {events[1].access_type!r}"
        )


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

    def test_format_trace_out_of_bounds_thread_id(self) -> None:
        """format_trace must not IndexError when thread_id >= len(thread_names)."""
        events = [
            TraceEvent(0, 0, "counter.py", 10, "f", "LOAD_ATTR", "read", "x", "C"),
            TraceEvent(1, 5, "counter.py", 10, "f", "STORE_ATTR", "write", "x", "C"),
        ]
        output = format_trace(events, num_threads=2, thread_names=["Alice"])
        assert "Thread 5" in output or "thread" in output.lower()

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

    def test_reproduction_percentage_rounds_not_truncates(self) -> None:
        """Reproduction percentage should be rounded, not floor-divided.

        2/3 = 66.67% → should display 67%, not 66%.
        """
        events = [
            TraceEvent(0, 0, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(1, 1, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
        ]
        output = format_trace(events, num_threads=2, reproduction_attempts=3, reproduction_successes=2)
        assert "2/3" in output
        assert "67%" in output


# ---------------------------------------------------------------------------
# Call chain helpers
# ---------------------------------------------------------------------------


class TestQualifiedName:
    def test_qualname_from_co_qualname(self) -> None:
        """Uses co_qualname when available (Python 3.11+)."""
        import sys

        class FakeCode:
            co_filename = "test.py"
            co_name = "dict"

        class FakeFrame:
            f_code = FakeCode()
            f_lineno = 1
            f_locals: dict[str, object] = {}

        if sys.version_info >= (3, 11):
            FakeCode.co_qualname = "DB.dict"  # type: ignore[attr-defined]
            assert qualified_name(FakeFrame()) == "DB.dict"
        else:
            # Falls back to co_name
            assert qualified_name(FakeFrame()) == "dict"

    def test_qualname_from_self(self) -> None:
        """Falls back to type(self).__name__ on 3.10."""
        import sys

        if sys.version_info >= (3, 11):
            return  # co_qualname takes precedence; tested above

        class DB:
            pass

        class FakeCode:
            co_filename = "test.py"
            co_name = "dict"

        class FakeFrame:
            f_code = FakeCode()
            f_lineno = 1
            f_locals = {"self": DB()}

        assert qualified_name(FakeFrame()) == "DB.dict"


class TestBuildCallChain:
    def test_chain_from_nested_frames(self) -> None:
        """Builds a chain of qualified names from nested frames."""

        class FakeCode:
            co_filename = "test.py"
            co_name = "inner"

        class FakeCodeOuter:
            co_filename = "test.py"
            co_name = "outer"

        class OuterFrame:
            f_code = FakeCodeOuter()
            f_lineno = 1
            f_locals: dict[str, object] = {}
            f_back = None

        class InnerFrame:
            f_code = FakeCode()
            f_lineno = 5
            f_locals: dict[str, object] = {}
            f_back = OuterFrame()

        chain = build_call_chain(InnerFrame(), filter_fn=lambda fn: fn == "test.py")
        assert chain == ["inner", "outer"]

    def test_chain_filters_non_user_frames(self) -> None:
        """Skips frames that don't pass the filter."""

        class UserCode:
            co_filename = "test.py"
            co_name = "user_func"

        class LibCode:
            co_filename = "/lib/python3.14/threading.py"
            co_name = "run"

        class TopFrame:
            f_code = UserCode()
            f_lineno = 1
            f_locals: dict[str, object] = {}
            f_back = None

        class MiddleFrame:
            f_code = LibCode()
            f_lineno = 10
            f_locals: dict[str, object] = {}
            f_back = TopFrame()

        class InnerFrame:
            f_code = UserCode()
            f_lineno = 5
            f_locals: dict[str, object] = {}
            f_back = MiddleFrame()

        chain = build_call_chain(InnerFrame(), filter_fn=lambda fn: fn == "test.py")
        assert chain == ["user_func", "user_func"]

    def test_empty_chain_returns_none(self) -> None:
        """Returns None when no frames pass the filter."""

        class LibCode:
            co_filename = "/lib/threading.py"
            co_name = "run"

        class Frame:
            f_code = LibCode()
            f_lineno = 1
            f_locals: dict[str, object] = {}
            f_back = None

        chain = build_call_chain(Frame(), filter_fn=lambda fn: fn == "test.py")
        assert chain is None

    def test_max_depth_limits_chain(self) -> None:
        """Chain is truncated at max_depth."""

        class Code:
            co_filename = "test.py"
            co_name = "f"

        class F3:
            f_code = Code()
            f_lineno = 1
            f_locals: dict[str, object] = {}
            f_back = None

        class F2:
            f_code = Code()
            f_lineno = 2
            f_locals: dict[str, object] = {}
            f_back = F3()

        class F1:
            f_code = Code()
            f_lineno = 3
            f_locals: dict[str, object] = {}
            f_back = F2()

        chain = build_call_chain(F1(), filter_fn=lambda fn: True, max_depth=2)
        assert chain is not None
        assert len(chain) == 2


class TestCallChainFormatting:
    def test_call_chain_shown_in_trace(self) -> None:
        """format_trace includes call chain info when present."""
        events = [
            TraceEvent(0, 0, "test.py", 10, "dict", "IO", "read", "file:/tmp/db.json", None, ["DB.dict", "do_incrs"]),
            TraceEvent(1, 1, "test.py", 10, "dict", "IO", "read", "file:/tmp/db.json", None, ["DB.dict", "do_incrs"]),
        ]
        output = format_trace(events, num_threads=2, num_explored=1)
        assert "Called from DB.dict <- do_incrs" in output

    def test_no_call_chain_when_none(self) -> None:
        """No chain tag when call_chain is None."""
        events = [
            TraceEvent(0, 0, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
            TraceEvent(1, 1, "counter.py", 10, "increment", "LOAD_ATTR", "read", "value", "Counter"),
        ]
        output = format_trace(events, num_threads=2, num_explored=1)
        assert " in " not in output or "interleavings" in output.split(" in ")[0]

    def test_call_chain_propagated_through_dedup(self) -> None:
        """call_chain survives deduplicate_to_source_lines."""
        events = [
            TraceEvent(0, 0, "test.py", 10, "dict", "IO", "read", "file:/tmp/x", None, ["DB.dict", "main"]),
        ]
        lines = deduplicate_to_source_lines(events)
        assert len(lines) == 1
        assert lines[0].call_chain == ["DB.dict", "main"]

    def test_io_detail_shown_in_trace(self) -> None:
        """format_trace includes detail lines for C-level I/O events."""
        events = [
            TraceEvent(
                0,
                0,
                "<C extension>",
                0,
                "",
                "IO",
                "read",
                "socket:unix:/var/run/postgresql/.s.PGSQL.5432",
                "IO",
                ["QuerySet.get", "AcquireUserView.post"],
                "SQL: SELECT * FROM auth_user WHERE id = 1",
            ),
            TraceEvent(
                1,
                1,
                "<C extension>",
                0,
                "",
                "IO",
                "write",
                "socket:unix:/var/run/postgresql/.s.PGSQL.5432",
                "IO",
                ["QuerySet.get", "AcquireUserView.post"],
                "SQL: SELECT * FROM auth_user WHERE id = 1",
            ),
        ]
        output = format_trace(events, num_threads=2, num_explored=1)
        assert "SQL: SELECT * FROM auth_user WHERE id = 1" in output
        assert "Called from QuerySet.get <- AcquireUserView.post" in output


# ---------------------------------------------------------------------------
# InterleavingResult repr
# ---------------------------------------------------------------------------


class TestInterleavingResultRepr:
    def test_short_counterexample_shown_fully(self) -> None:
        from frontrun.common import InterleavingResult

        r = InterleavingResult(property_holds=False, counterexample=[0, 1, 0, 1], num_explored=3)
        assert "[0, 1, 0, 1]" in repr(r)

    def test_long_counterexample_truncated(self) -> None:
        from frontrun.common import InterleavingResult

        r = InterleavingResult(property_holds=False, counterexample=[0] * 500, num_explored=3)
        s = repr(r)
        assert "500 steps" in s
        assert s.count("0") < 20  # not dumping all 500


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


class TestCollapsedRunThreadNames:
    """CollapsedRun should use custom thread_names, not hardcoded 'Thread N'."""

    def _make_many_events(self, thread_id: int, count: int) -> list[SourceLineEvent]:
        """Create a long run of events from one thread to trigger collapse."""
        return [
            SourceLineEvent(
                thread_id=thread_id,
                filename="counter.py",
                lineno=i + 1,
                function_name="increment",
                source_line=f"line {i}",
                access_type="read",
                obj_type_name="Counter",
                attr_name="value",
            )
            for i in range(count)
        ]

    def test_collapsed_run_uses_thread_names(self) -> None:
        """When events are collapsed, the summary should use thread_names, not 'Thread N'."""
        # Build a trace with enough events from thread 0 to force collapse,
        # plus a single event from thread 1 to create a shared access.
        events_t0 = self._make_many_events(0, 40)
        events_t1 = [
            SourceLineEvent(
                thread_id=1,
                filename="counter.py",
                lineno=1,
                function_name="increment",
                source_line="line 0",
                access_type="write",
                obj_type_name="Counter",
                attr_name="value",
            )
        ]
        all_events = events_t0 + events_t1

        output = format_trace(
            all_events,
            num_threads=2,
            thread_names=["Alice", "Bob"],
            max_lines=5,
        )
        # The collapsed run should say "Alice" not "Thread 0"
        assert "Thread 0" not in output, f"CollapsedRun used 'Thread 0' instead of 'Alice':\n{output}"
        assert "Alice" in output


class TestClassifyConflictObjTypeGrouping:
    """classify_conflict should group by (obj_type_name, attr_name), not attr_name alone."""

    def test_different_obj_types_same_attr_not_conflated(self):
        events = [
            SourceLineEvent(
                thread_id=0, filename="a.py", lineno=1, function_name="f",
                source_line="x = counter.value", access_type="read",
                attr_name="value", obj_type_name="Counter",
            ),
            SourceLineEvent(
                thread_id=1, filename="a.py", lineno=2, function_name="g",
                source_line="x = account.value", access_type="read",
                attr_name="value", obj_type_name="BankAccount",
            ),
            SourceLineEvent(
                thread_id=0, filename="a.py", lineno=3, function_name="f",
                source_line="counter.value = x + 1", access_type="write",
                attr_name="value", obj_type_name="Counter",
            ),
            SourceLineEvent(
                thread_id=1, filename="a.py", lineno=4, function_name="g",
                source_line="account.value = x + 1", access_type="write",
                attr_name="value", obj_type_name="BankAccount",
            ),
        ]
        result = classify_conflict(events)
        assert result.pattern != "lost_update", (
            "classify_conflict should NOT report a lost update when two different "
            "object types (Counter vs BankAccount) share the same attr_name 'value'. "
            f"Got pattern={result.pattern!r}, summary={result.summary!r}"
        )
