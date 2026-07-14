"""DBAPI cursor monkey-patching for SQL-level conflict detection.

Intercepts ``cursor.execute()`` and ``cursor.executemany()`` calls to
extract table-level read/write sets from SQL statements.  Reports each
table as a separate resource to the I/O reporter, suppressing the
coarser endpoint-level socket I/O reports.

Follows the same monkey-patching pattern as ``_io_detection.py``.

Implementation note: C-extension cursor types (sqlite3.Cursor, psycopg2
cursor) are immutable and cannot be patched directly via ``setattr``.
Instead, we patch the ``connect()`` function of each driver module to
inject a traced connection/cursor factory subclass.  For pure-Python
drivers like pymysql, direct class patching is used as a fallback.
"""

from __future__ import annotations

import importlib
import re
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from frontrun._deadlock import SchedulerAbort
from frontrun._io_detection import _io_tls, external_operation_scope, get_io_reporter, tx_store
from frontrun._io_detection import get_dpor_context as _get_dpor_context
from frontrun._patching import patch_method, restore_patches, wrap_method_metadata
from frontrun._schema import _detect_driver, get_schema
from frontrun._sql_db_scope import (
    _CONNECTION_DB_SCOPES,
    _DB_SCOPE_ATTR,
    _get_connection_db_scope,
    _get_primary_colset,
    _normalize_db_identity,
    _register_connection_db_scope,
    _stable_db_scope,
    _table_primary_colset,
)
from frontrun._sql_endpoint_suppression import (
    _set_active_sql_io_context,
    _suppress_endpoint_io,
    _suppress_lock,
    _suppress_tids,
    clear_permanent_suppressions,
    get_active_sql_io_context,
    is_sql_endpoint_suppressed,
    is_sql_write_suppressed,
    is_tid_suppressed,
    suppress_sql_endpoint,
    suppress_sql_write,
    suppress_tid_permanently,
)
from frontrun._sql_insert_tracker import record_insert, resolve_alias
from frontrun._sql_parsing import (
    LockIntent,
    TxOp,
    parse_sql_access,
)
from frontrun._sql_patch_registry import CONNECT_FACTORY_TARGETS, PYTHON_CURSOR_TARGETS
from frontrun._sql_row_locks import _acquire_pending_row_locks, _release_dpor_row_locks
from frontrun._sql_transactions import (
    _apply_tx_op_after_success,
    _detect_autobegin,
    _finalize_tx_end,
    _handle_tx_op,
    _prepare_tx_end,
    _report_or_buffer,
    handle_connection_commit,
    handle_connection_rollback,
    reset_connection_state,
)

# Try to import row-level predicate helpers.  These are always present in the
# same package, but guard with try/except for robustness.
try:
    from frontrun._sql_params import resolve_parameters
    from frontrun._sql_predicates import (
        EqualityPredicate,  # pyright: ignore[reportAssignmentType]
        extract_row_level_access,
    )
except ImportError:

    def resolve_parameters(sql: str, parameters: Any, paramstyle: str) -> str:  # type: ignore[misc]
        return sql

    def extract_row_level_access(sql: str, *, ast: Any | None = None) -> list[list[Any]] | None:  # type: ignore[misc]
        return None

    class EqualityPredicate:  # type: ignore[no-redef]
        def __init__(self, column: str, value: str):
            self.column = column
            self.value = value


# Re-exports for backward compatibility.  These helpers were originally
# defined in this module; they now live in _sql_endpoint_suppression and
# _sql_row_locks but are imported here so existing call sites and tests
# continue to work.
__all__ = [
    "_CONNECTION_DB_SCOPES",
    "_DB_SCOPE_ATTR",
    "_acquire_pending_row_locks",
    "_detect_autobegin",
    "_get_connection_db_scope",
    "_get_primary_colset",
    "_io_tls",
    "_normalize_db_identity",
    "_register_connection_db_scope",
    "_release_dpor_row_locks",
    "_report_or_buffer",
    "_set_active_sql_io_context",
    "_stable_db_scope",
    "_suppress_endpoint_io",
    "_suppress_lock",
    "_suppress_tids",
    "_table_primary_colset",
    "clear_permanent_suppressions",
    "get_active_sql_io_context",
    "is_sql_endpoint_suppressed",
    "is_sql_write_suppressed",
    "is_tid_suppressed",
    "reset_connection_state",
    "suppress_sql_endpoint",
    "suppress_sql_write",
    "suppress_tid_permanently",
]


# ---------------------------------------------------------------------------
# INSERT detection regex (used by _intercept_execute for post-INSERT capture).
# Reuses the same pattern as _sql_parsing._RE_INSERT to extract the table name
# in a single match, avoiding a redundant parse_sql_access call.
# ---------------------------------------------------------------------------

_RE_INSERT_TABLE = re.compile(
    r"^\s*INSERT\s+(?:OR\s+\w+\s+|IGNORE\s+)?INTO\s+(?:[`\"\[]?\w+[`\"\]]?\s*\.\s*)?[`\"\[]?(\w+)", re.I
)
_RE_UPDATE_TABLE = re.compile(r"^\s*UPDATE\s+(?:[`\"\[]?\w+[`\"\]]?\s*\.\s*)?[`\"\[]?(\w+)", re.I)


def _warm_sql_parsers() -> None:
    """Load optional SQL parsing dependencies before managed threads start.

    The first row-level SQL statement may lazily import ``sqlglot`` and a large
    set of helper modules. If that happens inside a DPOR-managed worker thread,
    the preload bridge observes a burst of unrelated file I/O and exploration
    can spend its small preemption budget on import noise instead of the user
    race. Warm the parser stack once on the main thread when patching SQL.

    Importantly, this also forces ``sqlglot.dialects.Dialect`` to be imported
    eagerly.  The ``sqlglot.dialects`` module uses a module-level
    ``_import_lock = threading.RLock()`` to guard lazy dialect loading.  If
    this lock is created after cooperative lock patching replaces
    ``threading.RLock``, it becomes a :class:`CooperativeRLock` which can
    deadlock when acquired outside a DPOR scheduler context (e.g. during
    counterexample reproduction).  Warming here — before ``patch_locks()`` —
    ensures the lock is a real ``RLock`` and all lazy imports are resolved.
    """
    try:
        extract_row_level_access("SELECT * FROM frontrun_warmup WHERE id = 1")
    except Exception:
        # Optional dependency missing or parser warmup failed. The actual SQL
        # interception path remains best-effort and will fall back naturally.
        pass
    # Force the lazy Dialect import so that sqlglot's _import_lock (which
    # guards __getattr__ for dialect loading) is exercised while threading
    # primitives are still real (not cooperative).
    try:
        from sqlglot.dialects import Dialect  # noqa: F401
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared sync DPOR scheduling + endpoint suppression
# ---------------------------------------------------------------------------


def _dpor_schedule_and_suppress_sync(
    reported: bool,
    operation: Any,
    parameters: Any,
    paramstyle: str,
    execute: Callable[[], Any],
    acquired_row_locks: list[str] | None = None,
    *,
    release_row_locks_on_error: bool = True,
) -> Any:
    """DPOR scheduling point + endpoint I/O suppression for sync SQL execution.

    Sync counterpart to ``_sql_cursor_async._dpor_schedule_and_suppress_async``.
    Forces a DPOR scheduling point if *reported* or *operation* is a string,
    acquires pending row locks, suppresses endpoint-level and cooperative-lock
    I/O during the actual driver call, and releases row locks on exception.

    Args:
        reported: Whether ``_report_sql_access`` recorded table accesses.
        operation: The raw SQL operation value (used to decide scheduling).
        parameters: The bound parameters (forwarded to ``_set_active_sql_io_context``).
        paramstyle: PEP 249 paramstyle string (forwarded to ``_set_active_sql_io_context``).
        execute: A zero-argument callable that performs the actual driver method call.
    """
    from frontrun._cooperative import suppress_sync_reporting, unsuppress_sync_reporting

    # Force a scheduling point so the scheduler can interleave between
    # SQL operations.  Without this, all code inside frontrun/ is skipped
    # by the tracer, so pending_io is never flushed between back-to-back
    # SQL calls.  This is needed both during DPOR exploration (to report
    # accesses) and during replay (to consume schedule entries that DPOR
    # generated for SQL statements).
    _dpor_ctx = _get_dpor_context()
    _held_sync_turn = False
    after_sync: Callable[[int], None] | None = None
    thread_id: int | None = None
    if _dpor_ctx is not None and (reported or isinstance(operation, str)):
        scheduler, thread_id = _dpor_ctx
        before_sync = getattr(scheduler, "before_sync_retry", None)
        after_sync = getattr(scheduler, "after_sync_retry", None)
        if callable(before_sync) and callable(after_sync):
            # Intentional: before_sync_retry grants this thread the exclusive
            # scheduler turn and holds it through the entire real DB execute()
            # below (after_sync_retry runs only in the outer finally).  This
            # deterministically serializes driver I/O so replay is exact.  The
            # trade-off is deliberate: any *modeled* contention is handed off
            # inside the turn via _acquire_pending_row_locks (row-lock
            # arbitration), while *unmodeled* DB-level blocking (advisory locks,
            # unique-index waits, etc.) will stall the whole scheduler until the
            # connection's lock_timeout aborts it.  Do not "fix" this by
            # releasing the turn around execute() without preserving that
            # determinism guarantee.
            if not before_sync(thread_id):
                raise SchedulerAbort("scheduler aborted before SQL execution")
            _held_sync_turn = True
        else:
            if not scheduler.report_and_wait(None, thread_id):
                raise SchedulerAbort("scheduler aborted before SQL execution")
    if reported:
        suppress_sql_write(operation, parameters, paramstyle)

    _set_active_sql_io_context(operation, parameters, paramstyle)
    # Suppress cooperative lock sync events during the actual DB call.
    # Internal psycopg2/driver locks are implementation details.
    suppress_sync_reporting()
    acquired: list[str] = []
    try:
        # Block if another DPOR thread holds a conflicting row lock. This runs
        # after the scheduling boundary and, for DporScheduler, while the SQL
        # transition still owns the turn. Otherwise a thread can become the
        # modeled row-lock holder while it is still waiting to be scheduled.
        acquired = _acquire_pending_row_locks()
        if acquired_row_locks is not None:
            acquired_row_locks.extend(acquired)
        if reported:
            with _suppress_endpoint_io():
                return execute()
        return execute()
    except Exception:
        # Release row locks on execution failure to prevent framework-induced
        # deadlocks.  Without this, if a SQL statement raises (e.g.,
        # OperationalError from SQLite lock contention), any row locks
        # acquired by _acquire_pending_row_locks remain held until thread
        # exit, blocking other DPOR threads indefinitely.
        if release_row_locks_on_error:
            _release_dpor_row_locks(acquired)
        raise
    finally:
        unsuppress_sync_reporting()
        if _held_sync_turn and after_sync is not None and thread_id is not None:
            after_sync(thread_id)


# ---------------------------------------------------------------------------
# Core interception logic
# ---------------------------------------------------------------------------


def _is_postgresql_db_object(db_obj: Any) -> bool:
    """Whether *db_obj* is a PostgreSQL cursor/connection with matched-row counts."""
    connection = getattr(db_obj, "connection", None)
    if connection is None:
        connection = db_obj
    try:
        return _detect_driver(connection) == "postgresql"
    except ValueError:
        return False


def _connection_for_db_object(db_obj: Any) -> Any:
    """Return the physical connection represented by a cursor/connection."""
    connection = getattr(db_obj, "connection", None)
    return db_obj if connection is None else connection


def _sql_resource_id(
    table: str,
    predicates: list[Any],
    temporal: str | None = None,
    *,
    db_scope: str | None = None,
) -> str:
    """Build a resource ID from table name and optional predicates."""
    resource = f"sql:{table}"
    if temporal:
        resource = f"{resource}:history:{temporal}"
    if db_scope is not None:
        resource = f"{resource}:db={db_scope}"
    if not predicates:
        return resource
    pred_key = tuple(sorted((p.column, p.value) for p in predicates))
    return f"{resource}:{pred_key}"


def _sql_sequence_resource_id(table: str, *, db_scope: str | None = None) -> str:
    """Build the shared sequence resource ID for INSERT ordering on a table."""
    return f"{_sql_resource_id(table, [], db_scope=db_scope)}:seq"


def _sql_database_resource_id(*, db_scope: str | None = None) -> str:
    return _sql_resource_id("__database__", [], db_scope=db_scope)


def _report_sql_access(
    operation: Any,
    parameters: Any = None,
    *,
    db_obj: Any = None,
    is_executemany: bool = False,
    paramstyle: str = "format",
    defer_tx_lock_release: bool = False,
    deferred_tx_end: list[Any] | None = None,
) -> bool:
    """Parse SQL and report table accesses to the per-thread reporter.

    Returns ``True`` if any SQL-level reporting was performed (which means
    endpoint-level I/O should be suppressed for the subsequent DB call).

    This helper is shared by both sync ``_intercept_execute`` and async
    ``_intercept_execute_async``.
    """
    reporter = get_io_reporter()
    reported = False

    if reporter is not None and isinstance(operation, str):
        access = parse_sql_access(operation)
        store = tx_store()
        connection = _connection_for_db_object(db_obj)
        owner = getattr(store, "_tx_connection", None)
        if getattr(store, "_in_transaction", False) and owner is not None and connection is not owner:
            if access.tx_op in (TxOp.COMMIT, TxOp.ROLLBACK) and not (access.read_tables or access.write_tables):
                # Ending an unrelated connection must not finalize the active
                # connection's modeled transaction.
                return True
            if access.tx_op is not None or access.read_tables or access.write_tables:
                raise RuntimeError(
                    "frontrun does not support overlapping SQL transactions on multiple connections in one worker; "
                    "use one connection per worker or end the active transaction first"
                )

        dpor_ctx = _get_dpor_context()
        semantic_fallback = bool(dpor_ctx is not None and getattr(dpor_ctx[0], "requires_semantic_io_fallback", False))
        if semantic_fallback and access.tx_op is None:
            has_parsed_access = bool(access.read_tables or access.write_tables)
            _report_or_buffer(
                reporter,
                _sql_database_resource_id(db_scope=_get_connection_db_scope(db_obj)),
                "read" if has_parsed_access else "write",
                track_row_lock=False,
            )
            reported = True
            if not has_parsed_access:
                return True

        # 1. Handle Transaction Control Operations
        if access.tx_op is not None:
            reported = True  # Suppress endpoint I/O for TX control too
            if defer_tx_lock_release:
                if access.tx_op in (TxOp.COMMIT, TxOp.ROLLBACK):
                    _prepare_tx_end(reporter, access.tx_op)
            else:
                _handle_tx_op(reporter, access.tx_op)
            if deferred_tx_end is not None and defer_tx_lock_release:
                deferred_tx_end.append(access.tx_op)

        # 2. Handle Data Access Operations
        if access.read_tables or access.write_tables:
            reported = True
            all_tables = access.read_tables | access.write_tables
            db_scope = _get_connection_db_scope(db_obj)

            # Row-level predicate extraction (WHERE equality, IN-lists, and INSERT VALUES)
            # Reuses the pre-parsed AST from parse_sql_access when available (avoids
            # a second sqlglot.parse_one call — Refactor 3).
            pred_rows: list[list[Any]] = [[]]  # default: one report, no predicates (table-level)
            has_row_level = False
            if len(all_tables) == 1 and not is_executemany:
                if parameters is not None:
                    resolved = resolve_parameters(operation, parameters, paramstyle)
                    rows = extract_row_level_access(resolved)
                else:
                    rows = extract_row_level_access(operation, ast=access.ast)
                if rows is not None:
                    pred_rows = rows
                    has_row_level = True

            lock_update = access.lock_intent in (LockIntent.UPDATE, LockIntent.UPDATE_SKIP_LOCKED)

            # Track which tables already reported their bridge resource in this op.
            reported_bridges: set[str] = set()

            def report_or_buffer(
                table: str, kind: str, rows: list[list[Any]], *, bridge_as_read: bool | None = None
            ) -> None:
                temporal = access.temporal_clauses.get(table) if access.temporal_clauses else None
                has_row_level_predicates = bool(rows and rows[0])

                # Whether this access participates in the primary-colset bridge
                # as a READ.  Normally this tracks ``kind == "read"`` (pure
                # INSERT writes stay fully row-granular and emit no bridge).
                # SELECT ... FOR UPDATE may elevate the row-level kind to
                # "write" for non-DPOR conflict creation, but the underlying
                # access is a read: it must still emit the primary READ bridge
                # so it conflicts with non-primary-colset accesses to the same
                # physical row, exactly like a plain UPDATE's read phase.
                if bridge_as_read is None:
                    bridge_as_read = kind == "read"

                # Conservative Column-Set Partitioning (Defect #1):
                # When using row-level predicates, we must ensure that accesses using
                # DIFFERENT sets of columns (e.g. SELECT by username vs UPDATE by id)
                # properly conflict. We do this by reporting a "bridge" resource (sql:<table>)
                # for every row-level access.
                #
                # To preserve row-level benefits for the most common column set (usually the PK),
                # we designate the first column set seen for each table as "primary".
                # - Primary colset accesses report a READ on the bridge resource.
                # - Non-primary colset accesses report a WRITE on the bridge resource.
                # - Table-level accesses report their actual kind (READ/WRITE) on the bridge.
                #
                # Result:
                # - Primary vs Primary: both READ bridge -> NO conflict on bridge. Row-level works!
                # - Primary vs Non-primary: READ vs WRITE bridge -> CONFLICT. Correct.
                # - Non-primary vs Non-primary: WRITE vs WRITE bridge -> CONFLICT (table-level).
                if dpor_ctx is not None and table not in reported_bridges and has_row_level_predicates:
                    colset = tuple(sorted(p.column for p in rows[0]))
                    primary = _get_primary_colset(table, colset, db_scope=db_scope)

                    if colset == primary:
                        # Primary row-level reads use a shared READ bridge so
                        # they still conflict conservatively with non-primary
                        # accesses, while primary row-level writes stay fully
                        # row-granular.
                        if bridge_as_read:
                            _report_or_buffer(
                                reporter,
                                _sql_resource_id(table, [], db_scope=db_scope),
                                "read",
                                force_immediate=lock_update,
                                track_row_lock=False,
                            )
                            reported_bridges.add(table)
                    else:
                        # Non-primary colset conflicts conservatively at table scope.
                        _report_or_buffer(
                            reporter,
                            _sql_resource_id(table, [], db_scope=db_scope),
                            "write",
                            force_immediate=lock_update,
                            track_row_lock=False,
                        )
                        reported_bridges.add(table)

                for row_preds in rows:
                    # Check if any predicate value matches a captured INSERT ID
                    alias = None
                    for pred in row_preds:
                        if isinstance(pred, EqualityPredicate):
                            alias = resolve_alias(table, pred.value, db_scope=db_scope)
                            if alias is not None:
                                break
                    res_id = (
                        alias if alias is not None else _sql_resource_id(table, row_preds, temporal, db_scope=db_scope)
                    )
                    _report_or_buffer(
                        reporter,
                        res_id,
                        kind,
                        force_immediate=lock_update,
                    )

            # Report explicit reads
            for table in access.read_tables:
                # SELECT FOR UPDATE reads row data and acquires an exclusive row
                # lock. Under DPOR, the row lock itself models lock ordering; the
                # row read is weak so it still conflicts with plain writers without
                # duplicating the dependency between two lock-protected writers.
                # SHARE locks are treated as reads (they don't block other shares).
                if lock_update and dpor_ctx is not None:
                    kind = "weak_read"
                elif lock_update:
                    kind = "write"
                else:
                    kind = "read"
                # FOR UPDATE remains a read for bridge purposes — emit the primary
                # READ bridge so it conflicts with non-primary-colset accesses to
                # the same physical row.
                report_or_buffer(table, kind, pred_rows, bridge_as_read=True)

            # Report implicit reads from Foreign Key dependencies
            schema = get_schema()
            for table in access.write_tables:
                fks = schema.get_fks(table)
                for fk in fks:
                    # Determine predicates for the referenced table
                    ref_pred_rows: list[list[Any]] = [[]]  # default table-level

                    # If we have row-level predicates for the write table
                    if has_row_level:
                        mapped_rows = []
                        for row in pred_rows:
                            # Check if row has the FK column
                            fk_val = None
                            for pred in row:
                                if isinstance(pred, EqualityPredicate) and pred.column == fk.column:
                                    fk_val = pred.value
                                    break

                            if fk_val is not None:
                                mapped_rows.append([EqualityPredicate(fk.ref_column, fk_val)])
                            else:
                                # If any row is missing the FK value, we must fall back to table-level
                                mapped_rows = [[]]
                                break
                        ref_pred_rows = mapped_rows

                    report_or_buffer(fk.ref_table, "read", ref_pred_rows)

            # Report writes
            for table in access.write_tables:
                report_or_buffer(table, "write", pred_rows)

            # Phantom read detection (sequence-number tracking):
            # SELECT depends on which rows exist in a table.  If a concurrent
            # INSERT adds a row (or DELETE removes one), the SELECT's result
            # changes.  Row-level conflict tracking misses this because the
            # new/removed row has a different resource ID than the SELECT's
            # table-level or row-level resource.
            #
            # Fix: use the table's :seq resource as a "membership" marker.
            # - Pure-read tables (SELECT) report READ on :seq.
            # - INSERT tables report WRITE on :seq (moved here from
            #   _capture_insert_id so the write is flushed at the INSERT's
            #   scheduling point, not left as orphaned pending_io).
            # - DELETE tables report WRITE on :seq.
            # - UPDATE tables report READ on :seq (defect #6 fix): UPDATE
            #   results depend on which rows exist (like SELECT), so
            #   concurrent INSERTs that add rows matching the UPDATE's WHERE
            #   clause are phantom reads. We use READ (not WRITE) to avoid
            #   false write-write conflicts between UPDATEs on different rows.
            #
            # This creates READ-WRITE conflicts between SELECT/UPDATE and
            # INSERT/DELETE, detecting phantom read races.
            pure_read_tables = access.read_tables - access.write_tables
            for table in pure_read_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "read",
                )
            # INSERT targets: in write_tables but NOT in read_tables (INSERT
            # doesn't read the target table, unlike UPDATE/DELETE).
            insert_tables = access.write_tables - access.read_tables
            for table in insert_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "write",
                )
            delete_tables = access.delete_tables or set()
            for table in delete_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "write",
                )
            # UPDATE targets: in both write_tables and read_tables (UPDATE
            # reads the WHERE clause and writes matched rows), excluding
            # DELETE tables. Report READ on :seq so DPOR creates conflict
            # arcs with concurrent INSERTs (which WRITE :seq). This lets
            # DPOR explore interleavings where both UPDATEs run before either
            # INSERT — the pattern that causes phantom races (defect #6).
            update_tables = (access.write_tables & access.read_tables) - delete_tables
            for table in update_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "read",
                )

    return reported


def _run_connection_tx_method(method: Callable[[], Any], operation: str, connection: Any = None) -> Any:
    with external_operation_scope():
        return _run_connection_tx_method_scoped(method, operation, connection)


def _run_connection_tx_method_scoped(method: Callable[[], Any], operation: str, connection: Any = None) -> Any:
    """Run a DB-API commit/rollback as one deterministic sync transition."""
    tx_op = TxOp.COMMIT if operation == "COMMIT" else TxOp.ROLLBACK
    handler = handle_connection_commit if operation == "COMMIT" else handle_connection_rollback
    store = tx_store()
    owner = getattr(store, "_tx_connection", None)
    tx_active = bool(getattr(store, "_in_transaction", False)) and (owner is None or owner is connection)
    if tx_active:
        handler(release_locks=False, finalize=False)

    # ROLLBACK is cleanup, not a user mutation.  Once the scheduler has
    # aborted, asking it for another SQL turn is guaranteed to be denied;
    # skipping the physical rollback would return an open transaction (and
    # its database locks) to a connection pool, poisoning later executions.
    # COMMIT remains scheduled/denied because it would make partial work
    # durable outside the counterexample schedule.
    dpor_ctx = _get_dpor_context()
    if operation == "ROLLBACK" and dpor_ctx is not None:
        scheduler = dpor_ctx[0]
        scheduler_aborted = (
            bool(getattr(scheduler, "_finished", False))
            or getattr(scheduler, "_error", None) is not None
            or bool(getattr(scheduler, "_aborted", False))
        )
        if scheduler_aborted:
            result = method()
            if tx_active:
                _finalize_tx_end(tx_op)
            elif owner is None:
                _release_dpor_row_locks()
            return result

    def execute() -> Any:
        result = method()
        # Keep modeled state and locks intact if physical I/O raises. On
        # success, finalize before handing the scheduler turn to another worker.
        if tx_active:
            _finalize_tx_end(tx_op)
        elif owner is None:
            _release_dpor_row_locks()
        return result

    return _dpor_schedule_and_suppress_sync(
        reported=True,
        operation=operation,
        parameters=None,
        paramstyle="format",
        execute=execute,
        release_row_locks_on_error=False,
    )


def _run_connection_close(method: Callable[[], Any], connection: Any) -> Any:
    """Clear modeled state after the owning physical connection closes."""
    result = method()
    if getattr(tx_store(), "_tx_connection", None) is connection:
        reset_connection_state()
    return result


def _wrap_connection_tx_methods(conn: Any) -> None:
    """Wrap a connection's ``commit`` / ``rollback`` to drive the tx state machine.

    DB-API call sites often end a transaction with ``conn.commit()`` /
    ``conn.rollback()`` rather than a textual ``COMMIT`` / ``ROLLBACK``.  Those
    method calls bypass ``cursor.execute()`` interception entirely, so the tx
    buffer is never flushed and row locks are held until thread exit (finding
    3).  Wrap the bound methods so they drive the same state machine.

    Best-effort: C-extension connection types (e.g. psycopg2) may reject
    instance attribute assignment; in that case we leave the connection
    unwrapped (autobegin/textual-COMMIT paths still apply).
    """
    for name in ("commit", "rollback", "close"):
        orig = getattr(conn, name, None)
        if orig is None or getattr(orig, "_frontrun_tx_wrapped", False):
            continue

        def _make(orig_method: Any = orig, _name: str = name) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                if _name == "close":
                    return _run_connection_close(lambda: orig_method(*args, **kwargs), conn)
                operation = _name.upper()
                suppress_sql_write(operation)
                return _run_connection_tx_method(lambda: orig_method(*args, **kwargs), operation, conn)

            _wrapped._frontrun_tx_wrapped = True  # type: ignore[attr-defined]
            return _wrapped

        try:
            setattr(conn, name, _make())
        except (AttributeError, TypeError):
            # C-level connection that forbids attribute assignment — skip.
            pass


def _capture_insert_id(cursor: Any, table: str) -> None:
    """Capture lastrowid after INSERT and report indexical alias.

    The shared sequence resource (sql:<table>:seq) WRITE is now reported
    in ``_report_sql_access`` instead of here.  This ensures the :seq write
    is flushed at the INSERT's scheduling point (via ``report_and_wait``),
    rather than being left as orphaned ``pending_io`` when the INSERT is
    the last operation before the thread exits.
    """
    reporter = get_io_reporter()
    if reporter is None:
        return

    lastrowid = getattr(cursor, "lastrowid", None)
    db_scope = _get_connection_db_scope(cursor)

    alias = record_insert(table, lastrowid, db_scope=db_scope)

    # Report the logical alias as a write (indexical tracking for determinism)
    _report_or_buffer(reporter, alias, "write")


def _record_uncaptured_insert(cursor: Any, table: str) -> None:
    """Record an executemany INSERT as having an uncaptured row ID (finding 10e).

    ``executemany`` inserts multiple rows but ``lastrowid`` exposes only the
    last one, so per-row indexical aliases cannot be built.  Recording the
    INSERT with ``concrete_id=None`` adds the table to the uncaptured set,
    keeping the determinism guard active for this path.
    """
    if get_io_reporter() is None:
        return
    db_scope = _get_connection_db_scope(cursor)
    record_insert(table, None, db_scope=db_scope)


def _execute_with_retry(execute: Callable[[], Any]) -> Any:
    """Execute a scheduled DB operation, retrying transient SQLite locks.

    Each retry must re-enter the scheduling envelope.  Retrying the raw driver
    call while holding an exclusive sync turn prevents the lock owner from
    running its COMMIT, creating a scheduler-induced SQLite deadlock.
    """
    for i in range(50):
        try:
            return execute()
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                raise
            time.sleep(0.01 * (i + 1))

    return execute()


def _intercept_execute(
    original_method: Any,
    self: Any,
    operation: Any,
    parameters: Any = None,
    *,
    is_executemany: bool = False,
    paramstyle: str = "format",
) -> Any:
    with external_operation_scope():
        return _intercept_execute_scoped(
            original_method,
            self,
            operation,
            parameters,
            is_executemany=is_executemany,
            paramstyle=paramstyle,
        )


def _intercept_execute_scoped(
    original_method: Any,
    self: Any,
    operation: Any,
    parameters: Any = None,
    *,
    is_executemany: bool = False,
    paramstyle: str = "format",
) -> Any:
    """Intercept a single execute/executemany call.

    Parses *operation*, reports table accesses to the per-thread reporter,
    activates suppression, then delegates to *original_method*.

    When *is_executemany* is False and the query touches exactly one table,
    resolves *parameters* and extracts row-level predicates (equality and
    IN-lists) so that
    the reported resource ID is finer-grained than plain ``sql:<table>``.

    Implements transaction grouping: when a transaction is active (BEGIN
    detected), I/O reports are buffered in TLS and only flushed when COMMIT is
    called.  ROLLBACK clears the buffer.  SAVEPOINTs and ROLLBACK TO SAVEPOINT
    are supported via buffer truncation.
    """
    insert_match = _RE_INSERT_TABLE.match(operation) if isinstance(operation, str) else None
    update_match = _RE_UPDATE_TABLE.match(operation) if isinstance(operation, str) else None

    # Detect autobegin: most DB-API drivers (psycopg2, pymysql) default to
    # autocommit=False, meaning the first statement implicitly starts a
    # transaction at the C/driver level without sending an explicit BEGIN
    # through cursor.execute().  We detect this and set _in_transaction so
    # that (a) accesses are buffered atomically, (b) the DPOR scheduler
    # treats the transaction as an atomic block, and (c) row locks are
    # tracked for deadlock detection.
    #
    # We skip this when connection.autocommit is True (each statement is
    # its own transaction, locks released immediately) or when we've
    # already seen an explicit BEGIN.
    _detect_autobegin(self)

    # Permanently suppress LD_PRELOAD *socket* events for this thread.
    # SQL-level reporting (table/row granularity) supersedes socket-level
    # I/O.  The listener only applies tid suppression to socket events,
    # so non-SQL file I/O from this thread passes through.
    # The patched connect() also registers the connection's socket endpoint
    # for endpoint-based suppression (which handles remote connections).
    suppress_tid_permanently()

    deferred_tx_end: list[Any] = []
    reported = _report_sql_access(
        operation,
        parameters,
        db_obj=self,
        is_executemany=is_executemany,
        paramstyle=paramstyle,
        defer_tx_lock_release=True,
        deferred_tx_end=deferred_tx_end,
    )
    deferred_tx_op = deferred_tx_end[0] if deferred_tx_end else None

    def execute() -> Any:
        if parameters is not None:
            result = original_method(self, operation, parameters)
        else:
            result = original_method(self, operation)
        if deferred_tx_op is not None:
            _apply_tx_op_after_success(deferred_tx_op, _connection_for_db_object(self))
        return result

    statement_row_locks: list[str] = []
    result = _execute_with_retry(
        lambda: _dpor_schedule_and_suppress_sync(
            reported,
            operation,
            parameters,
            paramstyle,
            execute,
            statement_row_locks,
            release_row_locks_on_error=deferred_tx_op is None,
        )
    )

    # Defect #6 fix: release this statement's speculative row locks for
    # PostgreSQL 0-row UPDATEs.
    # In PostgreSQL, an UPDATE that matches 0 rows acquires no row locks
    # (there are no rows to lock).  But frontrun's row-lock arbitration
    # acquired a scheduler-level lock based on the WHERE-clause resource ID
    # regardless of whether any rows matched.  This over-serialization
    # prevents DPOR from exploring interleavings where both 0-row UPDATEs
    # execute before either INSERT (the UPDATE-INSERT phantom race pattern).
    # PostgreSQL reports matched rows. MySQL reports changed rows by default,
    # so zero there does not prove that no row matched and took a lock.
    if update_match is not None and reported and statement_row_locks and _is_postgresql_db_object(self):
        rowcount = getattr(self, "rowcount", -1)
        if rowcount == 0:
            _release_dpor_row_locks(statement_row_locks)

    # Post-INSERT: capture lastrowid and record indexical alias
    if insert_match is not None and reported:
        if not is_executemany:
            _capture_insert_id(self, insert_match.group(1))
        else:
            # executemany INSERT assigns one ID per row, but ``lastrowid`` only
            # exposes the final row's ID — the per-row IDs cannot be captured.
            # Record the table as uncaptured so the nondeterminism guard still
            # fires for this path instead of silently passing (finding 10e).
            _record_uncaptured_insert(self, insert_match.group(1))

    return result


# ---------------------------------------------------------------------------
# Traced cursor/connection subclasses (created dynamically per driver)
# ---------------------------------------------------------------------------


# DB-API drivers spell the bind-parameter keyword differently:
# psycopg2 → ``vars``, pymysql → ``args``, sqlite3 / generic → ``parameters``.
_PARAM_KWARG_NAMES = ("parameters", "vars", "args", "params")


def _recover_param_kwarg(parameters: Any, kwargs: dict[str, Any]) -> Any:
    """Recover a bind-parameter value passed as a keyword (finding 10c).

    ``cur.execute(sql, vars=params)`` (psycopg2) / ``args=params`` (pymysql)
    lands the params in ``**kwargs``; without recovering them the placeholders
    are never substituted and the real driver call loses its parameters.
    """
    if parameters is not None or not kwargs:
        return parameters
    for name in _PARAM_KWARG_NAMES:
        if name in kwargs:
            return kwargs[name]
    return parameters


def _make_traced_cursor_class(base_cursor_cls: type, paramstyle: str = "format") -> type:
    """Return a subclass of *base_cursor_cls* that intercepts execute calls.

    The original methods are looked up from ``_ORIGINAL_METHODS`` at call time
    rather than captured at class creation.  This allows tests to swap out the
    stored original (e.g. to install a spy) and have the traced cursor pick up
    the new value transparently.

    *paramstyle* is the PEP 249 paramstyle for the driver (e.g. ``"qmark"``
    for sqlite3, ``"pyformat"`` for psycopg2, ``"format"`` for pymysql).
    It is stored as a class attribute so that ``_intercept_execute`` can
    resolve parameters before extracting row-level predicates.
    """

    _execute_key = (base_cursor_cls, "execute")
    _executemany_key = (base_cursor_cls, "executemany")
    _paramstyle = paramstyle

    class TracedCursor(base_cursor_cls):  # type: ignore[valid-type]
        _cursor_paramstyle: str = _paramstyle

        def execute(self, operation: Any, parameters: Any = None, /, **kwargs: Any) -> Any:  # type: ignore[override]
            original = _ORIGINAL_METHODS.get(_execute_key, base_cursor_cls.execute)
            parameters = _recover_param_kwarg(parameters, kwargs)
            return _intercept_execute(
                original, self, operation, parameters, is_executemany=False, paramstyle=self._cursor_paramstyle
            )

        def executemany(self, operation: Any, parameters: Any = None, /, **kwargs: Any) -> Any:  # type: ignore[override]
            original = _ORIGINAL_METHODS.get(_executemany_key, base_cursor_cls.executemany)
            parameters = _recover_param_kwarg(parameters, kwargs)
            return _intercept_execute(
                original, self, operation, parameters, is_executemany=True, paramstyle=self._cursor_paramstyle
            )

    TracedCursor.__name__ = f"Traced{base_cursor_cls.__name__}"
    TracedCursor.__qualname__ = f"Traced{base_cursor_cls.__qualname__}"
    return TracedCursor


# Shared cache of traced cursor subclasses, keyed by (base factory, paramstyle).
# Used both by the patched connect() (connection-level cursor_factory) and by
# the wrapped conn.cursor() (per-cursor cursor_factory) so an explicit factory
# is wrapped exactly once and the user's row format is preserved (finding 5).
_TRACED_CURSOR_CLASSES: dict[tuple[type, str], type] = {}


def _get_traced_cursor_class(base_cursor_cls: type, paramstyle: str) -> type:
    """Return (and cache) a traced subclass of *base_cursor_cls*.

    Already-traced classes (subclasses of one we built) are returned as-is to
    avoid double-wrapping.
    """
    if getattr(base_cursor_cls, "_frontrun_traced_cursor", False):
        return base_cursor_cls
    key = (base_cursor_cls, paramstyle)
    cached = _TRACED_CURSOR_CLASSES.get(key)
    if cached is None:
        cached = _make_traced_cursor_class(base_cursor_cls, paramstyle=paramstyle)
        cached._frontrun_traced_cursor = True  # type: ignore[attr-defined]
        _TRACED_CURSOR_CLASSES[key] = cached
    return cached


_TRACED_CONNECTION_CLASSES: dict[tuple[type, str], type] = {}


def _get_traced_connection_class(base_connection_cls: type, paramstyle: str) -> type:
    """Return a connection subclass that drives tx state for C-extension drivers."""
    if getattr(base_connection_cls, "_frontrun_traced_connection", False):
        return base_connection_cls
    key = (base_connection_cls, paramstyle)
    cached = _TRACED_CONNECTION_CLASSES.get(key)
    if cached is not None:
        return cached

    class TracedConnection(base_connection_cls):  # type: ignore[valid-type]
        def cursor(self, *args: Any, **kwargs: Any) -> Any:
            factory = kwargs.get("cursor_factory")
            if factory is not None:
                kwargs["cursor_factory"] = _get_traced_cursor_class(factory, paramstyle)
            return super().cursor(*args, **kwargs)

        def commit(self) -> None:
            _run_connection_tx_method(super().commit, "COMMIT", self)

        def rollback(self) -> None:
            _run_connection_tx_method(super().rollback, "ROLLBACK", self)

        def close(self) -> None:
            _run_connection_close(super().close, self)

    TracedConnection.__name__ = f"Traced{base_connection_cls.__name__}"
    TracedConnection.__qualname__ = f"Traced{base_connection_cls.__qualname__}"
    TracedConnection._frontrun_traced_connection = True  # type: ignore[attr-defined]
    TracedConnection.commit._frontrun_tx_wrapped = True  # type: ignore[attr-defined]
    TracedConnection.rollback._frontrun_tx_wrapped = True  # type: ignore[attr-defined]
    TracedConnection.close._frontrun_tx_wrapped = True  # type: ignore[attr-defined]
    _TRACED_CONNECTION_CLASSES[key] = TracedConnection
    return TracedConnection


def _wrap_connection_cursor(conn: Any, paramstyle: str) -> None:
    """Wrap ``conn.cursor`` so an explicit ``cursor_factory`` is traced too.

    The patched ``connect()`` injects a traced class via the connection-level
    ``cursor_factory``, but ``conn.cursor(cursor_factory=RealDictCursor)``
    overrides it with an *untraced* class, making those queries invisible at
    every level (finding 5).  We wrap ``conn.cursor`` so an explicit factory is
    dynamically subclassed with the traced mixin, preserving the user's row
    format.  Best-effort: skipped when the connection forbids attribute
    assignment (the connection-level traced factory still covers default
    cursors).
    """
    orig_cursor = getattr(conn, "cursor", None)
    if orig_cursor is None or getattr(orig_cursor, "_frontrun_cursor_wrapped", False):
        return

    def _wrapped_cursor(*args: Any, **kwargs: Any) -> Any:
        factory = kwargs.get("cursor_factory")
        if factory is not None:
            kwargs["cursor_factory"] = _get_traced_cursor_class(factory, paramstyle)
        return orig_cursor(*args, **kwargs)

    _wrapped_cursor._frontrun_cursor_wrapped = True  # type: ignore[attr-defined]
    try:
        conn.cursor = _wrapped_cursor
    except (AttributeError, TypeError):
        pass


def _make_traced_sqlite3_connection_class(base_cls: type = sqlite3.Connection) -> type:
    """Return a *base_cls* subclass whose cursor() uses TracedCursor.

    On Python 3.14+, ``Connection.execute()`` creates cursors in C without
    calling ``self.cursor()``, so we must also override ``execute`` and
    ``executemany`` to route through the traced cursor.
    """
    _traced_cursor_cls = _make_traced_cursor_class(sqlite3.Cursor, paramstyle="qmark")

    class TracedConnection(base_cls):  # type: ignore[valid-type]
        def cursor(self, factory: type = _traced_cursor_cls) -> sqlite3.Cursor:  # type: ignore[override]
            return super().cursor(factory)

        def execute(self, sql: Any, parameters: Any = (), /) -> sqlite3.Cursor:  # type: ignore[override]
            cur = self.cursor()
            cur.execute(sql, parameters)
            return cur

        def executemany(self, sql: Any, parameters: Any = (), /) -> sqlite3.Cursor:  # type: ignore[override]
            cur = self.cursor()
            cur.executemany(sql, parameters)
            return cur

        def commit(self) -> None:  # type: ignore[override]
            # Drive the tx state machine before the driver call so the buffered
            # accesses are flushed even when COMMIT is issued via the connection
            # method rather than as SQL text (finding 3).
            _run_connection_tx_method(super().commit, "COMMIT", self)

        def rollback(self) -> None:  # type: ignore[override]
            _run_connection_tx_method(super().rollback, "ROLLBACK", self)

        def close(self) -> None:  # type: ignore[override]
            _run_connection_close(super().close, self)

    TracedConnection.__name__ = f"Traced{base_cls.__name__}"
    TracedConnection.__qualname__ = f"Traced{base_cls.__qualname__}"
    return TracedConnection


_TRACED_SQLITE3_CONN_CLASSES: dict[type, type] = {}


def _get_traced_sqlite3_connection_class(base_cls: type) -> type:
    """Return (and cache) a traced subclass of *base_cls* for sqlite3.

    Already-traced classes are returned as-is to avoid double-wrapping.
    """
    if getattr(base_cls, "_frontrun_traced_sqlite3_conn", False):
        return base_cls
    cached = _TRACED_SQLITE3_CONN_CLASSES.get(base_cls)
    if cached is None:
        cached = _make_traced_sqlite3_connection_class(base_cls)
        cached._frontrun_traced_sqlite3_conn = True  # type: ignore[attr-defined]
        _TRACED_SQLITE3_CONN_CLASSES[base_cls] = cached
    return cached


# ---------------------------------------------------------------------------
# Global patching state
# ---------------------------------------------------------------------------

_sql_patched = False

# Stores (module, attribute_name, original_value) for each patched site
_PATCHES: list[tuple[Any, str, Any]] = []

# Expose a dict-like view keyed by (class, method_name) for test introspection.
# For the factory-based approach we store the original connect function here.
_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}

# Global lock_timeout (milliseconds) to inject on new PostgreSQL connections.
# Set by frontrun.explore(lock_timeout=...) and cleared after exploration.
_lock_timeout_ms: int | None = None


def set_lock_timeout(ms: int | None) -> None:
    """Set the global lock_timeout that will be injected on new PG connections."""
    global _lock_timeout_ms  # noqa: PLW0603
    _lock_timeout_ms = ms


def get_lock_timeout() -> int | None:
    """Return the current global lock_timeout (milliseconds), or None."""
    return _lock_timeout_ms


# ---------------------------------------------------------------------------
# sqlite3 patching
# ---------------------------------------------------------------------------


def _patch_sqlite3() -> None:
    """Patch sqlite3.connect to inject TracedConnection factory."""
    orig_connect = sqlite3.connect
    traced_conn_cls = _make_traced_sqlite3_connection_class()

    def patched_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        user_factory = kwargs.get("factory")
        if user_factory is None:
            kwargs["factory"] = traced_conn_cls
        else:
            kwargs["factory"] = _get_traced_sqlite3_connection_class(user_factory)
        conn = orig_connect(*args, **kwargs)
        identity = _normalize_db_identity("sqlite", *args, **kwargs)
        if identity is None:
            identity = f"sqlite-memory:{id(conn)}"
        _register_connection_db_scope(conn, identity)
        return conn

    sqlite3.connect = wrap_method_metadata(patched_connect, orig_connect, name="connect")  # type: ignore[assignment]
    _PATCHES.append((sqlite3, "connect", orig_connect))
    # Expose for tests via _ORIGINAL_METHODS — key by (cursor_class, method_name)
    _ORIGINAL_METHODS[(sqlite3.Cursor, "execute")] = sqlite3.Cursor.execute
    _ORIGINAL_METHODS[(sqlite3.Cursor, "executemany")] = sqlite3.Cursor.executemany


# ---------------------------------------------------------------------------
# Generic Python-class patching (for pure-Python drivers)
# ---------------------------------------------------------------------------


def _patch_class_methods(cls: type, paramstyle: str) -> None:
    """Directly patch execute/executemany on a Python cursor class."""
    for method_name in ("execute", "executemany"):
        _is_executemany = method_name == "executemany"

        def _make_patched(
            orig: Any, mname: str = method_name, ps: str = paramstyle, _iem: bool = _is_executemany
        ) -> Any:
            def _patched(self: Any, operation: Any, parameters: Any = None, *args: Any, **kwargs: Any) -> Any:
                # Recover params passed as a keyword (pymysql ``args=``) — see
                # finding 10c.  ``*args`` already captures a positional extra.
                if parameters is None and args:
                    parameters = args[0]
                parameters = _recover_param_kwarg(parameters, kwargs)
                return _intercept_execute(orig, self, operation, parameters, is_executemany=_iem, paramstyle=ps)

            return wrap_method_metadata(_patched, orig, name=mname)

        patch_method(
            cls,
            method_name,
            originals=_ORIGINAL_METHODS,
            patches=_PATCHES,
            make_wrapper=_make_patched,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Drivers to attempt patching via direct class method replacement (pure Python)
_PYTHON_CURSOR_TARGETS: list[tuple[str, str, str]] = [
    ("pymysql.cursors", "Cursor", "pymysql"),
]


def patch_sql() -> None:
    """Monkey-patch DBAPI cursor.execute() for known drivers."""
    global _sql_patched  # noqa: PLW0603
    if _sql_patched:
        return

    _warm_sql_parsers()

    # sqlite3 requires factory-injection approach
    _patch_sqlite3()

    # Pure-Python drivers can be patched directly
    for target in PYTHON_CURSOR_TARGETS:
        try:
            mod = importlib.import_module(target.module_path)
            cls = getattr(mod, target.class_name)
            driver_mod = importlib.import_module(target.paramstyle_module)
            paramstyle = getattr(driver_mod, "paramstyle", "format")
            _patch_class_methods(cls, paramstyle)
        except (ImportError, AttributeError):
            pass  # driver not installed — skip silently

    def _make_patched_connect(
        orig: Any,
        default_cursor_cls: type,
        paramstyle: str,
        driver: str,
        default_connection_cls: type | None = None,
    ) -> Any:
        def patched_connect(*args: Any, **kwargs: Any) -> Any:
            # Wrap whatever cursor_factory the caller already set (e.g. Django's Cursor),
            # rather than using setdefault, which is a no-op when the caller set it first.
            user_factory = kwargs.get("cursor_factory", default_cursor_cls)
            kwargs["cursor_factory"] = _get_traced_cursor_class(user_factory, paramstyle)
            if driver == "psycopg2" and default_connection_cls is not None:
                user_connection_factory = kwargs.get("connection_factory", default_connection_cls)
                kwargs["connection_factory"] = _get_traced_connection_class(user_connection_factory, paramstyle)
            from frontrun._cooperative import suppress_sync_reporting as _ssr
            from frontrun._cooperative import unsuppress_sync_reporting as _usr

            # Suppress LD_PRELOAD events BEFORE the actual connect call.
            # The background pipe reader may process events from connect()
            # before we return; suppressing the tid first ensures those
            # events are dropped in the listener() callback.  After the
            # connection is established we register the *endpoint* for
            # permanent suppression and remove the thread-level suppression.
            suppress_tid_permanently()
            _ssr()
            try:
                conn = orig(*args, **kwargs)
            finally:
                _usr()
            # Now that the connection is established, register its socket
            # endpoint for permanent suppression.  The thread-level tid
            # suppression remains as a belt-and-suspenders fallback for
            # any socket events that raced through the pipe before the
            # endpoint was registered.  The listener only uses tid
            # suppression for *socket* events, so file I/O passes through.
            suppress_sql_endpoint(conn)
            _wrap_connection_tx_methods(conn)
            _wrap_connection_cursor(conn, paramstyle)
            identity = _normalize_db_identity("connection", conn)
            if identity is None and args and isinstance(args[0], str):
                identity = f"{driver}-dsn:{args[0]}"
            if identity is None:
                relevant = {
                    "host": kwargs.get("host"),
                    "port": kwargs.get("port"),
                    "dbname": kwargs.get("dbname") or kwargs.get("database") or kwargs.get("db"),
                }
                identity = _normalize_db_identity("mapping", driver, relevant)
            if identity is not None:
                _register_connection_db_scope(conn, identity)
            # Inject SET lock_timeout if configured (defect #6 workaround).
            # Use the *original* cursor class to avoid triggering DPOR
            # scheduling points during connection setup.
            if _lock_timeout_ms is not None and driver in ("psycopg2", "psycopg"):
                _ssr()
                try:
                    _was_autocommit = conn.autocommit
                    conn.autocommit = True
                    _lt_cur = conn.cursor(cursor_factory=default_cursor_cls)
                    try:
                        _lock_timeout_sql = f"SET lock_timeout = '{int(_lock_timeout_ms)}ms'"
                        suppress_sql_write("BEGIN")
                        suppress_sql_write(_lock_timeout_sql)
                        _lt_cur.execute(_lock_timeout_sql)
                    finally:
                        _lt_cur.close()
                    conn.autocommit = _was_autocommit
                finally:
                    _usr()
            return conn

        return patched_connect

    # psycopg2: patch via cursor_factory injection into connect()
    for target in CONNECT_FACTORY_TARGETS:
        try:
            driver_mod = importlib.import_module(target.module_name)
            cursor_mod = importlib.import_module(target.cursor_module_name)
            orig_cursor_cls = getattr(cursor_mod, target.cursor_attr_name)
            orig_connection_cls = getattr(cursor_mod, "connection", None)
            orig_connect = driver_mod.connect
            setattr(
                driver_mod,
                "connect",
                _make_patched_connect(
                    orig_connect,
                    orig_cursor_cls,
                    paramstyle=target.paramstyle,
                    driver=target.driver,
                    default_connection_cls=orig_connection_cls if isinstance(orig_connection_cls, type) else None,
                ),
            )
            _PATCHES.append((driver_mod, "connect", orig_connect))
            _ORIGINAL_METHODS[(orig_cursor_cls, "execute")] = orig_cursor_cls.execute
            _ORIGINAL_METHODS[(orig_cursor_cls, "executemany")] = orig_cursor_cls.executemany
        except (ImportError, AttributeError):
            pass

    _sql_patched = True


def clear_sql_metadata() -> None:
    """Reset all global SQL resource tracking metadata.

    Call this between DPOR exploration sessions to ensure test isolation.
    """
    _table_primary_colset.clear()
    _CONNECTION_DB_SCOPES.clear()


def unpatch_sql() -> None:
    """Restore original DBAPI cursor methods and connect functions."""
    global _sql_patched  # noqa: PLW0603
    if not _sql_patched:
        return
    restore_patches(_PATCHES)
    _PATCHES.clear()
    _ORIGINAL_METHODS.clear()
    _TRACED_CURSOR_CLASSES.clear()
    _TRACED_SQLITE3_CONN_CLASSES.clear()
    _sql_patched = False
