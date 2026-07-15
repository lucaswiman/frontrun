"""SQL parse-failure fallback: unrecognized SQL must be a database-wide write.

Soundness invariant: over-merging is allowed, under-merging is forbidden.
A str statement that fails parsing (or parses to something the classifier
does not recognize) must get the same conservative treatment as opaque
bytes operations — a single database-wide WRITE — never zero accesses and
never a silently-narrow scope.
"""

from __future__ import annotations

from typing import Any

import pytest

from frontrun._io_detection import set_io_reporter
from frontrun._sql_cursor import _report_sql_access
from frontrun._sql_parsing import parse_sql_access

_DATABASE_WRITE = ("sql:__database__", "write")


def _capture(operation: Any) -> tuple[bool, list[tuple[str, str]]]:
    events: list[tuple[str, str]] = []
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        reported = _report_sql_access(operation, db_obj=None)
    finally:
        set_io_reporter(None)
    return reported, events


# ---------------------------------------------------------------------------
# Garbage / unparseable SQL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("GARBAGE !!!", id="garbage"),
        pytest.param("THIS IS NOT SQL AT ALL ;;;", id="not-sql"),
        pytest.param("", id="empty-string"),
        pytest.param("   \n\t  ", id="whitespace-only"),
    ],
)
def test_unparseable_sql_is_database_wide_write(operation: str) -> None:
    reported, events = _capture(operation)
    assert reported, f"{operation!r}: not reported at all (zero accesses — false certification risk)"
    assert _DATABASE_WRITE in events, f"{operation!r}: missing conservative database-wide write: {events}"


# ---------------------------------------------------------------------------
# Exotic-but-valid SQL the parser does not understand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("VACUUM ANALYZE users", id="vacuum"),
        pytest.param("CALL do_something(1)", id="call-procedure"),
        pytest.param("REINDEX TABLE users", id="reindex"),
        pytest.param("CLUSTER users USING users_pkey", id="cluster"),
        pytest.param("LISTEN mychannel", id="listen"),
        pytest.param("NOTIFY mychannel, 'payload'", id="notify"),
        pytest.param("CREATE EXTENSION IF NOT EXISTS pgcrypto", id="create-extension"),
        pytest.param("DO $$ BEGIN PERFORM refresh_rollups(); END $$", id="do-block"),
        pytest.param("EXPLAIN ANALYZE SELECT * FROM users", id="explain-analyze"),
    ],
)
def test_unrecognized_sql_is_database_wide_write(operation: str) -> None:
    reported, events = _capture(operation)
    assert reported
    assert _DATABASE_WRITE in events, f"{operation!r}: missing conservative database-wide write: {events}"


# ---------------------------------------------------------------------------
# Compound statements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("SET search_path TO reporting; CALL refresh_rollups()", id="set-then-call"),
        pytest.param("INSERT INTO audit VALUES (1); FROBNICATE ALL THE THINGS", id="dml-then-garbage"),
        pytest.param("SELECT * FROM users; GARBAGE !!!", id="select-then-garbage"),
    ],
)
def test_compound_statement_with_opaque_part_is_database_wide_write(operation: str) -> None:
    """If any part of a compound statement is opaque, the whole operation must
    keep the conservative database-wide write (partial narrow results would
    silently drop the opaque part's accesses)."""
    reported, events = _capture(operation)
    assert reported
    assert _DATABASE_WRITE in events, f"{operation!r}: missing conservative database-wide write: {events}"


# ---------------------------------------------------------------------------
# Opaque bytes operations (pre-existing behavior, pinned as regression)
# ---------------------------------------------------------------------------


def test_bytes_operation_is_database_wide_write() -> None:
    reported, events = _capture(b"UPDATE t SET x = 1")
    assert reported
    assert _DATABASE_WRITE in events


# ---------------------------------------------------------------------------
# Recognized-but-misclassified: SELECT INTO writes its target table
# ---------------------------------------------------------------------------


def test_select_into_target_table_is_a_write() -> None:
    """SELECT ... INTO creates and populates the target table; classifying it
    as a read under-merges (a concurrent reader of the target would not
    conflict)."""
    access = parse_sql_access("SELECT * INTO backup_users FROM users")
    assert "backup_users" in access.write_tables
    assert "users" in access.read_tables

    _, events = _capture("SELECT * INTO backup_users FROM users")
    assert ("sql:backup_users", "write") in events, f"SELECT INTO target not written: {events}"


def test_select_into_inside_union_target_table_is_a_write() -> None:
    access = parse_sql_access("SELECT a INTO t2 FROM t1 UNION SELECT b FROM t3")
    assert "t2" in access.write_tables
    assert {"t1", "t3"} <= access.read_tables


# ---------------------------------------------------------------------------
# Performance guard: well-modeled SQL must NOT become coarse
# ---------------------------------------------------------------------------


def test_well_modeled_select_is_not_database_wide_write() -> None:
    reported, events = _capture("SELECT * FROM users WHERE id = 1")
    assert reported
    assert _DATABASE_WRITE not in events, f"modeled SELECT escalated to database-wide write: {events}"
    assert ("sql:__database__", "read") in events


def test_well_modeled_update_is_not_database_wide_write() -> None:
    reported, events = _capture("UPDATE users SET name = 'x' WHERE id = 1")
    assert reported
    assert _DATABASE_WRITE not in events
