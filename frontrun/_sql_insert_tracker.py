"""Indexical INSERT ID tracking for deterministic DPOR resource IDs.

When threads INSERT into tables with autoincrement PKs, the assigned IDs
depend on execution order.  This module captures ``cursor.lastrowid`` after
each INSERT and maps concrete IDs to *logical aliases* like
``sql:users:t0_ins0`` ("thread 0's first INSERT into users").

Downstream operations referencing a captured concrete ID are translated to
the same logical alias, giving DPOR stable resource IDs across interleavings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._cooperative import get_context
from frontrun._sql_db_scope import _normalize_table_identity


@dataclass
class InsertRecord:
    """One captured INSERT event."""

    table: str
    db_scope: str | None
    thread_id: int | None  # None = main/setup thread (no scheduler context)
    seq: int  # per-(thread, table) counter
    concrete_id: Any | None  # lastrowid value (None if capture failed)
    logical_alias: str  # e.g. "sql:users:t0_ins0"
    captured: bool  # True if lastrowid was successfully captured


@dataclass
class _TrackerState:
    """Mutable state for the INSERT tracker, cleared between executions."""

    records: list[InsertRecord] = field(default_factory=list)
    concrete_to_alias: dict[tuple[str | None, str, Any], str] = field(default_factory=dict)
    thread_table_seq: dict[tuple[str | None, int | None, str], int] = field(default_factory=dict)
    uncaptured_tables: set[str] = field(default_factory=set)


_lock = _rt.lock()
_state = _TrackerState()


def _get_thread_id() -> int | None:
    """Return the scheduler thread ID, or None if outside scheduler context."""
    ctx = get_context()
    if ctx is not None:
        return ctx[1]
    return None


def _make_alias(table: str, thread_id: int | None, seq: int, db_scope: str | None = None) -> str:
    """Build a logical alias string for an INSERT."""
    tid = "setup" if thread_id is None else f"t{thread_id}"
    base = f"sql:{table}"
    if db_scope is not None:
        base = f"{base}:db={db_scope}"
    return f"{base}:{tid}_ins{seq}"


def record_insert(table: str, concrete_id: Any | None, *, db_scope: str | None = None) -> str:
    """Record an INSERT and return its logical alias.

    Called after INSERT execution with the captured ``lastrowid`` (or None
    if capture failed).  Thread ID is obtained from the scheduler context.
    """
    # Fold as ``_sql_resource_id`` does, so a later reference spelling the
    # table differently still resolves to this alias.
    table = _normalize_table_identity(table)
    thread_id = _get_thread_id()
    with _lock:
        key = (db_scope, thread_id, table)
        seq = _state.thread_table_seq.get(key, 0)
        _state.thread_table_seq[key] = seq + 1

        alias = _make_alias(table, thread_id, seq, db_scope=db_scope)
        captured = concrete_id is not None

        record = InsertRecord(
            table=table,
            db_scope=db_scope,
            thread_id=thread_id,
            seq=seq,
            concrete_id=concrete_id,
            logical_alias=alias,
            captured=captured,
        )
        _state.records.append(record)

        if captured:
            # Map (table, concrete_id) -> alias for downstream resolution.
            # Use str() for the concrete_id to handle int/str mismatches in
            # predicate values (SQL predicates are always strings).
            _state.concrete_to_alias[(db_scope, table, str(concrete_id))] = alias
        else:
            _state.uncaptured_tables.add(table)

        return alias


def resolve_alias(table: str, concrete_id: Any, *, db_scope: str | None = None) -> str | None:
    """Look up the logical alias for a concrete ID produced by an INSERT.

    Returns the alias string (e.g. ``"sql:users:t0_ins0"``) or ``None`` if
    the concrete ID was not produced by a tracked INSERT.
    """
    table = _normalize_table_identity(table)
    with _lock:
        return _state.concrete_to_alias.get((db_scope, table, str(concrete_id)))


def get_uncaptured_tables() -> set[str]:
    """Return tables where INSERT lastrowid capture failed."""
    with _lock:
        return _state.uncaptured_tables.copy()


def ensure_no_uncaptured_inserts() -> None:
    """Raise :class:`~frontrun.common.NondeterministicSQLError` if any INSERTs had uncaptured IDs."""
    tables = get_uncaptured_tables()
    if tables:
        from frontrun.common import NondeterministicSQLError

        tables_str = ", ".join(sorted(tables))
        raise NondeterministicSQLError(
            f"SQL INSERT statements on table(s) [{tables_str}] could not capture lastrowid.\n\n"
            "Without the assigned ID, frontrun cannot create stable indexical resource IDs\n"
            "for rows created by INSERTs, making DPOR results non-deterministic.\n\n"
            "For PostgreSQL, add a RETURNING clause to your INSERT statements, or\n"
            "pre-allocate rows with explicit IDs in your test setup.\n\n"
            "Pass warn_nondeterministic_sql=False to suppress this check."
        )


def clear_insert_tracker() -> None:
    """Reset all INSERT tracking state (call between DPOR executions)."""
    with _lock:
        _state.records.clear()
        _state.concrete_to_alias.clear()
        _state.thread_table_seq.clear()
        _state.uncaptured_tables.clear()


def get_records() -> list[InsertRecord]:
    """Return a copy of all INSERT records (for testing/debugging)."""
    with _lock:
        return list(_state.records)
