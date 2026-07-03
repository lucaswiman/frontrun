from __future__ import annotations

from frontrun._sql_anomaly import classify_sql_anomaly
from frontrun._trace_format import TraceEvent


def _sql_event(step: int, tid: int, table: str, kind: str) -> TraceEvent:
    return TraceEvent(
        step_index=step,
        thread_id=tid,
        filename="<C extension>",
        lineno=0,
        function_name="",
        opcode="IO",
        access_type=kind,
        attr_name=f"sql:{table}",
        obj_type_name="IO",
    )


def _sql_event_row(step: int, tid: int, table: str, row_key: str, kind: str) -> TraceEvent:
    return TraceEvent(
        step_index=step,
        thread_id=tid,
        filename="<C extension>",
        lineno=0,
        function_name="",
        opcode="IO",
        access_type=kind,
        attr_name=f"sql:{table}:{row_key}",
        obj_type_name="IO",
    )


def _non_sql_event(step: int, tid: int) -> TraceEvent:
    return TraceEvent(
        step_index=step,
        thread_id=tid,
        filename="test_file.py",
        lineno=10,
        function_name="some_func",
        opcode="LOAD_ATTR",
        access_type="read",
        attr_name="value",
        obj_type_name="Counter",
    )


class TestExtractSqlEvents:
    def test_filters_to_sql_events_only(self) -> None:
        # Mix of SQL and non-SQL events; SQL events form a write-write conflict
        events = [
            _non_sql_event(0, 0),
            _sql_event(1, 0, "accounts", "write"),
            _non_sql_event(2, 1),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None

    def test_extracts_table_from_resource_id(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "accounts", "read"),
            _sql_event(2, 0, "accounts", "write"),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert "accounts" in result.tables

    def test_extracts_table_from_row_level_id(self) -> None:
        events = [
            _sql_event_row(0, 0, "accounts", "(('id', '1'),)", "read"),
            _sql_event_row(1, 1, "accounts", "(('id', '1'),)", "read"),
            _sql_event_row(2, 0, "accounts", "(('id', '1'),)", "write"),
            _sql_event_row(3, 1, "accounts", "(('id', '1'),)", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert "accounts" in result.tables

    def test_empty_events_returns_empty(self) -> None:
        result = classify_sql_anomaly([])
        assert result is None


class TestClassifyLostUpdate:
    def test_classic_lost_update(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "accounts", "read"),
            _sql_event(2, 0, "accounts", "write"),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "lost_update"

    def test_lost_update_reverse_write_order(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "accounts", "read"),
            _sql_event(2, 1, "accounts", "write"),
            _sql_event(3, 0, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "lost_update"

    def test_lost_update_involves_correct_tables(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "accounts", "read"),
            _sql_event(2, 0, "accounts", "write"),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert "accounts" in result.tables

    def test_lost_update_involves_correct_threads(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "accounts", "read"),
            _sql_event(2, 0, "accounts", "write"),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert 0 in result.threads
        assert 1 in result.threads

    def test_serialized_t0_then_t1_is_not_lost_update(self) -> None:
        # T0 completes its read-write cycle before T1 starts: no lost update.
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 0, "accounts", "write"),
            _sql_event(2, 1, "accounts", "read"),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        # Serialized operations cannot produce a lost update anomaly.
        assert result is None or result.kind != "lost_update"

    def test_serialized_t1_then_t0_is_not_lost_update(self) -> None:
        # T1 completes its read-write cycle before T0 starts: no lost update.
        events = [
            _sql_event(0, 1, "accounts", "read"),
            _sql_event(1, 1, "accounts", "write"),
            _sql_event(2, 0, "accounts", "read"),
            _sql_event(3, 0, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is None or result.kind != "lost_update"


class TestClassifyWriteSkew:
    def test_write_skew_different_tables(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "balances", "read"),
            _sql_event(2, 0, "balances", "write"),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "write_skew"

    def test_write_skew_summary_mentions_tables(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "balances", "read"),
            _sql_event(2, 0, "balances", "write"),
            _sql_event(3, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert "accounts" in result.summary or "balances" in result.summary


class TestClassifyWriteSkewSingleTable:
    def test_write_skew_single_table_different_rows(self) -> None:
        events = [
            _sql_event_row(0, 0, "accounts", "id=1", "read"),
            _sql_event_row(1, 1, "accounts", "id=1", "write"),
            _sql_event_row(2, 1, "accounts", "id=2", "read"),
            _sql_event_row(3, 0, "accounts", "id=2", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "write_skew"


class TestFindCycleState:
    def test_edge_to_stale_entries_dont_corrupt_cycle(self) -> None:
        from frontrun._sql_anomaly import _find_cycle

        graph: dict[int, list[tuple[int, str, str]]] = {
            0: [(1, "WR", "tbl_a")],
            1: [(2, "RW", "tbl_a")],
            3: [(4, "WR", "tbl_b")],
            4: [(5, "RW", "tbl_b")],
            5: [(3, "WW", "tbl_b")],
        }

        result = _find_cycle(graph)
        assert result is not None
        for frm, to, _etype, _res in result:
            assert frm in {3, 4, 5} and to in {3, 4, 5}


class TestClassifyWriteWrite:
    def test_write_write_no_reads(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "write"),
            _sql_event(1, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "write_write"

    def test_write_write_correct_tables(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "write"),
            _sql_event(1, 1, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert "accounts" in result.tables


class TestClassifyDirtyRead:
    def test_dirty_read(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "write"),
            _sql_event(1, 1, "accounts", "read"),
            _sql_event(2, 0, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind in ("dirty_read", "non_repeatable_read", "write_write", "lost_update")

    def test_dirty_read_cycle_reports_all_tables(self) -> None:
        # A dirty read cycle spanning two tables: T0 writes A, T1 reads A (WR),
        # T1 writes B, T0 reads B (WR) — both tables must appear in result.tables.
        events = [
            _sql_event(0, 0, "table_a", "write"),
            _sql_event(1, 1, "table_a", "read"),
            _sql_event(2, 1, "table_b", "write"),
            _sql_event(3, 0, "table_b", "read"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "dirty_read"
        assert "table_a" in result.tables
        assert "table_b" in result.tables


class TestClassifyNonRepeatableRead:
    def test_non_repeatable_read(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "accounts", "write"),
            _sql_event(2, 0, "accounts", "read"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "non_repeatable_read"


class TestNoSqlEvents:
    def test_no_sql_events_returns_none(self) -> None:
        events = [
            _non_sql_event(0, 0),
            _non_sql_event(1, 1),
            _non_sql_event(2, 0),
        ]
        result = classify_sql_anomaly(events)
        assert result is None

    def test_empty_list_returns_none(self) -> None:
        result = classify_sql_anomaly([])
        assert result is None

    def test_single_thread_returns_none(self) -> None:
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 0, "accounts", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is None


class TestCrossGranularityNonRepeatableRead:
    """Bug: _check_non_repeatable_read matches by exact resource_id, missing
    cross-granularity conflicts where a row-level read is interleaved with
    a table-level write on the same table.

    Example: Thread 0 reads accounts:(('id','1'),) twice, with Thread 1
    doing an UPDATE (read+write) on accounts in between. The resource strings
    differ, so the non-repeatable read is missed.

    Note: write-only threads (INSERT/DELETE with no read) should still be
    classified as phantom reads, not non-repeatable reads.
    """

    def test_row_read_table_update_row_read(self) -> None:
        """Row-level reads with a table-level UPDATE in between should be NRR.

        Thread 1 does both read and write (UPDATE pattern), so this is a
        non-repeatable read, not a phantom read.
        """
        events = [
            _sql_event_row(0, 0, "accounts", "(('id','1'),)", "read"),
            _sql_event(1, 1, "accounts", "read"),  # Thread 1 reads (UPDATE reads WHERE clause)
            _sql_event(2, 1, "accounts", "write"),  # Thread 1 writes (UPDATE sets values)
            _sql_event_row(3, 0, "accounts", "(('id','1'),)", "read"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None, (
            "Cross-granularity non-repeatable read should be detected: "
            "Thread 0 reads a row twice while Thread 1 UPDATEs the same table"
        )
        assert result.kind == "non_repeatable_read"
        assert "accounts" in result.tables

    def test_table_read_row_update_table_read(self) -> None:
        """Table-level reads with a row-level UPDATE in between should be NRR.

        Thread 1 does both read and write at row level (UPDATE pattern).
        """
        events = [
            _sql_event(0, 0, "accounts", "read"),  # table-level read
            _sql_event_row(1, 1, "accounts", "(('id','1'),)", "read"),  # row-level read (UPDATE WHERE)
            _sql_event_row(2, 1, "accounts", "(('id','1'),)", "write"),  # row-level write (UPDATE SET)
            _sql_event(3, 0, "accounts", "read"),  # table-level read
        ]
        result = classify_sql_anomaly(events)
        assert result is not None, (
            "Cross-granularity non-repeatable read should be detected: "
            "Thread 0 reads a table twice while Thread 1 UPDATEs a row"
        )
        assert result.kind == "non_repeatable_read"
        assert "accounts" in result.tables

    def test_row_read_table_insert_row_read_is_phantom(self) -> None:
        """Row-level reads with a write-only INSERT in between is phantom, not NRR.

        Thread 1 only writes (INSERT pattern, no reads), so this should be
        classified as a phantom read.
        """
        events = [
            _sql_event_row(0, 0, "orders", "set_a", "read"),
            _sql_event(1, 1, "orders", "write"),  # Thread 1 INSERTs (write-only)
            _sql_event_row(2, 0, "orders", "set_a", "read"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "phantom_read", "Write-only thread (INSERT) between reads should be phantom_read, not NRR"


class TestClassifyPhantomRead:
    def test_phantom_read_insert_between_reads(self) -> None:
        """Thread 0 reads different tables; thread 1 does write-only INSERT on orders.

        NRR check fires first for same-resource read-write-read patterns, so
        phantom detection only triggers when NRR doesn't match (write by a
        thread that never reads the table).
        """
        # Thread 0 reads 'orders' at two different rows (row-level IDs differ)
        # Thread 1 inserts a new row (write-only, never reads 'orders')
        # NRR uses resource_id matching, so table-level read + table-level write + table-level read
        # will match NRR first.  To test phantom, we need NRR to NOT match.
        # Use row-level resource IDs where the two reads don't match the write resource:
        events = [
            _sql_event_row(0, 0, "orders", "set_a", "read"),  # Thread 0 reads set A
            _sql_event(1, 1, "orders", "write"),  # Thread 1 inserts (table-level write)
            _sql_event_row(2, 0, "orders", "set_a", "read"),  # Thread 0 re-reads set A
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        # NRR checks same resource_id: "orders:set_a" read-write-read.
        # The write is at "orders" (table-level), not "orders:set_a", so NRR doesn't match.
        # Phantom fires because thread 1 only writes, never reads 'orders'.
        assert result.kind == "phantom_read"
        assert "orders" in result.tables

    def test_phantom_vs_non_repeatable_read(self) -> None:
        """When writer also reads the table, it's non-repeatable read (not phantom)."""
        events = [
            _sql_event(0, 0, "accounts", "read"),
            _sql_event(1, 1, "accounts", "write"),  # UPDATE by thread 1 (also reads)
            _sql_event(2, 1, "accounts", "read"),  # thread 1 reads (shows it's an UPDATE)
            _sql_event(3, 0, "accounts", "read"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "non_repeatable_read"

    def test_no_phantom_single_read(self) -> None:
        """Thread reads table only once — no phantom possible."""
        events = [
            _sql_event(0, 0, "orders", "read"),
            _sql_event(1, 1, "orders", "write"),
        ]
        result = classify_sql_anomaly(events)
        # Only one read, so phantom can't fire. It'll be dirty_read or write-related.
        if result is not None:
            assert result.kind != "phantom_read"

    def test_phantom_read_summary_mentions_table(self) -> None:
        """Phantom read on table-level resources (NRR matches first for same resource)."""
        events = [
            _sql_event_row(0, 0, "products", "all_rows", "read"),
            _sql_event(1, 1, "products", "write"),  # INSERT (table-level, write-only)
            _sql_event_row(2, 0, "products", "all_rows", "read"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "phantom_read"
        assert "products" in result.summary

    def test_seq_membership_marker_insert_between_selects_is_phantom(self) -> None:
        """Realistic phantom trace using the :seq membership marker.

        A SELECT reports READ on the table and 'sql:<table>:seq'; a concurrent
        INSERT reports WRITE on a new *row* resource and 'sql:<table>:seq'.
        The only exact resource shared read-write-read is ':seq', which must
        NOT be classified as non-repeatable read — the :seq marker exists
        precisely to detect phantoms (INSERT/DELETE write it, SELECT/UPDATE
        read it).
        """
        events = [
            _sql_event(0, 0, "orders", "read"),  # SELECT #1 (table read)
            _sql_event_row(1, 0, "orders", "seq", "read"),
            _sql_event_row(2, 1, "orders", "(('id','5'),)", "write"),  # INSERT new row
            _sql_event_row(3, 1, "orders", "seq", "write"),
            _sql_event(4, 0, "orders", "read"),  # SELECT #2 (table read)
            _sql_event_row(5, 0, "orders", "seq", "read"),
        ]
        result = classify_sql_anomaly(events)
        assert result is not None
        assert result.kind == "phantom_read", result.kind


class TestRowLevelResourceIds:
    def test_different_rows_same_table(self) -> None:
        events = [
            _sql_event_row(0, 0, "accounts", "(('id', '1'),)", "read"),
            _sql_event_row(1, 1, "accounts", "(('id', '2'),)", "read"),
            _sql_event_row(2, 0, "accounts", "(('id', '1'),)", "write"),
            _sql_event_row(3, 1, "accounts", "(('id', '2'),)", "write"),
        ]
        result = classify_sql_anomaly(events)
        assert result is None
