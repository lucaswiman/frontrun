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

from frontrun._io_detection import _io_tls, get_io_reporter
from frontrun._io_detection import get_dpor_context as _get_dpor_context
from frontrun._patching import patch_method, restore_patches, wrap_method_metadata
from frontrun._schema import get_schema
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
    is_tid_suppressed,
    suppress_sql_endpoint,
    suppress_tid_permanently,
)
from frontrun._sql_insert_tracker import record_insert, resolve_alias
from frontrun._sql_parsing import (
    LockIntent,
    parse_sql_access,
)
from frontrun._sql_patch_registry import CONNECT_FACTORY_TARGETS, PYTHON_CURSOR_TARGETS
from frontrun._sql_row_locks import _acquire_pending_row_locks, _release_dpor_row_locks
from frontrun._sql_transactions import (
    _detect_autobegin,
    _handle_tx_op,
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
    "is_tid_suppressed",
    "reset_connection_state",
    "suppress_sql_endpoint",
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
) -> Any:
    """DPOR scheduling point + endpoint I/O suppression for sync SQL execution.

    Sync counterpart to ``_sql_cursor_async._dpor_schedule_and_suppress_async``.
    Acquires pending row locks, forces a DPOR scheduling point if *reported* or
    *operation* is a string, suppresses endpoint-level and cooperative-lock I/O
    during the actual driver call, and releases row locks on exception.

    Args:
        reported: Whether ``_report_sql_access`` recorded table accesses.
        operation: The raw SQL operation value (used to decide scheduling).
        parameters: The bound parameters (forwarded to ``_set_active_sql_io_context``).
        paramstyle: PEP 249 paramstyle string (forwarded to ``_set_active_sql_io_context``).
        execute: A zero-argument callable that performs the actual driver method call.
    """
    from frontrun._cooperative import suppress_sync_reporting, unsuppress_sync_reporting

    # Block if another DPOR thread holds a conflicting row lock
    _acquire_pending_row_locks()

    # Force a scheduling point so the scheduler can interleave between
    # SQL operations.  Without this, all code inside frontrun/ is skipped
    # by the tracer, so pending_io is never flushed between back-to-back
    # SQL calls.  This is needed both during DPOR exploration (to report
    # accesses) and during replay (to consume schedule entries that DPOR
    # generated for SQL statements).
    _dpor_ctx = _get_dpor_context()
    if _dpor_ctx is not None and (reported or isinstance(operation, str)):
        _dpor_ctx[0].report_and_wait(None, _dpor_ctx[1])

    _set_active_sql_io_context(operation, parameters, paramstyle)
    # Suppress cooperative lock sync events during the actual DB call.
    # Internal psycopg2/driver locks are implementation details.
    suppress_sync_reporting()
    try:
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
        _release_dpor_row_locks()
        raise
    finally:
        unsuppress_sync_reporting()


# ---------------------------------------------------------------------------
# Core interception logic
# ---------------------------------------------------------------------------


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


def _report_sql_access(
    operation: Any,
    parameters: Any = None,
    *,
    db_obj: Any = None,
    is_executemany: bool = False,
    paramstyle: str = "format",
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

        # 1. Handle Transaction Control Operations
        if access.tx_op is not None:
            reported = True  # Suppress endpoint I/O for TX control too
            _handle_tx_op(reporter, access.tx_op)

        # 2. Handle Data Access Operations
        if access.read_tables or access.write_tables:
            reported = True
            all_tables = access.read_tables | access.write_tables
            dpor_ctx = _get_dpor_context()
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

            def report_or_buffer(table: str, kind: str, rows: list[list[Any]]) -> None:
                temporal = access.temporal_clauses.get(table) if access.temporal_clauses else None
                has_row_level_predicates = bool(rows and rows[0])

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
                        if kind == "read":
                            _report_or_buffer(
                                reporter,
                                _sql_resource_id(table, [], db_scope=db_scope),
                                "read",
                                force_immediate=lock_update,
                            )
                            reported_bridges.add(table)
                    else:
                        # Non-primary colset conflicts conservatively at table scope.
                        _report_or_buffer(
                            reporter,
                            _sql_resource_id(table, [], db_scope=db_scope),
                            "write",
                            force_immediate=lock_update,
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
                    _report_or_buffer(reporter, res_id, kind, force_immediate=lock_update)

            # Report explicit reads
            for table in access.read_tables:
                # SELECT FOR UPDATE is both read and write to create conflicts.
                # SHARE locks are treated as reads (they don't block other shares).
                kind = "write" if lock_update else "read"
                report_or_buffer(table, kind, pred_rows)

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
    for name, handler in (("commit", handle_connection_commit), ("rollback", handle_connection_rollback)):
        orig = getattr(conn, name, None)
        if orig is None or getattr(orig, "_frontrun_tx_wrapped", False):
            continue

        def _make(orig_method: Any = orig, _handler: Any = handler) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                _handler()
                return orig_method(*args, **kwargs)

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


def _execute_with_retry(original_method: Any, cursor: Any, operation: Any, parameters: Any = None) -> Any:
    """Execute a DB-API method, retrying transient SQLite lock errors."""
    for i in range(50):
        try:
            if parameters is not None:
                return original_method(cursor, operation, parameters)
            return original_method(cursor, operation)
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                raise
            time.sleep(0.01 * (i + 1))

    if parameters is not None:
        return original_method(cursor, operation, parameters)
    return original_method(cursor, operation)


def _intercept_execute(
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

    reported = _report_sql_access(
        operation,
        parameters,
        db_obj=self,
        is_executemany=is_executemany,
        paramstyle=paramstyle,
    )

    result = _dpor_schedule_and_suppress_sync(
        reported,
        operation,
        parameters,
        paramstyle,
        lambda: _execute_with_retry(original_method, self, operation, parameters),
    )

    # Defect #6 fix: release row locks for 0-row UPDATEs.
    # In PostgreSQL, an UPDATE that matches 0 rows acquires no row locks
    # (there are no rows to lock).  But frontrun's row-lock arbitration
    # acquired a scheduler-level lock based on the WHERE-clause resource ID
    # regardless of whether any rows matched.  This over-serialization
    # prevents DPOR from exploring interleavings where both 0-row UPDATEs
    # execute before either INSERT (the UPDATE-INSERT phantom race pattern).
    if update_match is not None and reported:
        rowcount = getattr(self, "rowcount", -1)
        if rowcount == 0:
            _release_dpor_row_locks()

    # Post-INSERT: capture lastrowid and record indexical alias
    if insert_match is not None and not is_executemany and reported:
        _capture_insert_id(self, insert_match.group(1))

    return result


# ---------------------------------------------------------------------------
# Traced cursor/connection subclasses (created dynamically per driver)
# ---------------------------------------------------------------------------


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
            return _intercept_execute(
                original, self, operation, parameters, is_executemany=False, paramstyle=self._cursor_paramstyle
            )

        def executemany(self, operation: Any, parameters: Any = None, /, **kwargs: Any) -> Any:  # type: ignore[override]
            original = _ORIGINAL_METHODS.get(_executemany_key, base_cursor_cls.executemany)
            return _intercept_execute(
                original, self, operation, parameters, is_executemany=True, paramstyle=self._cursor_paramstyle
            )

    TracedCursor.__name__ = f"Traced{base_cursor_cls.__name__}"
    TracedCursor.__qualname__ = f"Traced{base_cursor_cls.__qualname__}"
    return TracedCursor


def _make_traced_sqlite3_connection_class() -> type:
    """Return a sqlite3.Connection subclass whose cursor() uses TracedCursor.

    On Python 3.14+, ``Connection.execute()`` creates cursors in C without
    calling ``self.cursor()``, so we must also override ``execute`` and
    ``executemany`` to route through the traced cursor.
    """
    _traced_cursor_cls = _make_traced_cursor_class(sqlite3.Cursor, paramstyle="qmark")

    class TracedConnection(sqlite3.Connection):
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
            handle_connection_commit()
            super().commit()

        def rollback(self) -> None:  # type: ignore[override]
            handle_connection_rollback()
            super().rollback()

    TracedConnection.__name__ = "TracedConnection"
    TracedConnection.__qualname__ = "TracedConnection"
    return TracedConnection


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
# Set by explore_dpor(lock_timeout=...) and cleared after exploration.
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
        if "factory" not in kwargs:
            kwargs["factory"] = traced_conn_cls
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

    def _make_patched_connect(orig: Any, default_cursor_cls: type, paramstyle: str, driver: str) -> Any:
        # Cache traced subclasses by (caller-provided cursor_factory class) to avoid
        # creating a new class on every connect() call.
        _cache: dict[type, type] = {}

        def patched_connect(*args: Any, **kwargs: Any) -> Any:
            # Wrap whatever cursor_factory the caller already set (e.g. Django's Cursor),
            # rather than using setdefault, which is a no-op when the caller set it first.
            user_factory = kwargs.get("cursor_factory", default_cursor_cls)
            if user_factory not in _cache:
                _cache[user_factory] = _make_traced_cursor_class(user_factory, paramstyle=paramstyle)
            kwargs["cursor_factory"] = _cache[user_factory]
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
                        _lt_cur.execute(f"SET lock_timeout = '{int(_lock_timeout_ms)}ms'")
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
            orig_connect = driver_mod.connect
            setattr(
                driver_mod,
                "connect",
                _make_patched_connect(
                    orig_connect, orig_cursor_cls, paramstyle=target.paramstyle, driver=target.driver
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
    _sql_patched = False
