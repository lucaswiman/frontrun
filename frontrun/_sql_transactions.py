"""Transaction grouping for SQL interception.

Tracks transaction state on the per-thread :data:`_io_tls` so that:

* I/O reports inside a transaction can be **buffered** and flushed atomically
  on COMMIT (or discarded on ROLLBACK), preserving transactional atomicity
  for the DPOR scheduler.
* **Autobegin** transactions (psycopg2's default ``autocommit=False`` mode,
  where the first statement implicitly opens a transaction) are detected
  even though no explicit ``BEGIN`` flows through ``cursor.execute()``.
* **Savepoints** are honored via buffer-index bookmarks: ``ROLLBACK TO``
  truncates the buffer back to the savepoint's index.
* DPOR row-locks are released on COMMIT/ROLLBACK.

State lives on :data:`_io_tls` (shared with ``_sql_cursor``):

* ``_in_transaction`` — bool
* ``_is_autobegin``   — bool (autobegin reports immediately, like
  READ COMMITTED, instead of buffering)
* ``_tx_buffer``      — list of pending ``(res_id, kind)`` tuples
* ``_tx_savepoints``  — dict mapping savepoint name to buffer index
* ``_pending_row_locks`` — list of resource IDs needing DPOR row-lock
  arbitration (drained by ``_sql_row_locks._acquire_pending_row_locks``)
* ``_held_row_locks`` — set of row-lock resources already acquired by this
  transaction; later accesses to those rows are serialized by the lock
"""

from __future__ import annotations

from typing import Any

from frontrun._io_detection import tx_store
from frontrun._sql_parsing import TxOp
from frontrun._sql_row_locks import _release_dpor_row_locks

__all__ = [
    "_detect_autobegin",
    "_handle_tx_op",
    "_report_or_buffer",
    "handle_connection_commit",
    "handle_connection_rollback",
    "reset_connection_state",
]


def _report_or_buffer(
    reporter: Any,
    res_id: str,
    kind: str,
    *,
    force_immediate: bool = False,
    track_row_lock: bool = True,
    report_access: bool = True,
) -> None:
    """Report a SQL access immediately, or buffer it if inside a transaction.

    When ``force_immediate=True`` the access is reported right away even
    inside a transaction (used for SELECT FOR UPDATE to let the DPOR engine
    learn about write-intent conflicts before C-level blocking can occur).
    Transaction atomicity is preserved because the DPOR scheduler still
    skips yielding inside transactions.

    When ``report_access=False`` inside a transaction, only row-lock
    arbitration is tracked. This is used for SELECT FOR UPDATE after the
    scheduler models the row lock directly; reporting the row itself as a
    second write creates redundant DPOR branches.

    Autobegin transactions (``_is_autobegin=True``) are NOT buffered: with
    READ COMMITTED isolation (PostgreSQL default), individual statements are
    visible to other transactions, so DPOR must see each access point to
    explore interleavings.  Row-lock tracking still works because
    ``_in_transaction`` is True.
    """
    store = tx_store()
    in_tx = getattr(store, "_in_transaction", False)
    is_autobegin = getattr(store, "_is_autobegin", False)
    held_row_locks = getattr(store, "_held_row_locks", set())
    if track_row_lock and res_id in held_row_locks:
        return

    should_report_access = report_access or not (track_row_lock and in_tx)
    if should_report_access:
        if in_tx and not force_immediate and not is_autobegin:
            if not hasattr(store, "_tx_buffer"):
                store._tx_buffer = []
            store._tx_buffer.append((res_id, kind))
        else:
            reporter(res_id, kind)

    # Track resources that need row-lock arbitration.
    # SELECT FOR UPDATE (force_immediate) always needs arbitration.
    # Any write inside a transaction (INSERT, UPDATE, DELETE) also needs
    # arbitration because PG row locks (e.g. from UNIQUE constraints or
    # row-level locks) can cause the cooperative scheduler to deadlock
    # when one thread blocks in the kernel waiting for another's lock
    # (defect #6).
    if track_row_lock and in_tx and (force_immediate or kind == "write"):
        pending = getattr(store, "_pending_row_locks", None)
        if pending is None:
            pending = []
            store._pending_row_locks = pending
        pending.append(res_id)


def _connection_autocommit(conn: Any) -> bool | None:
    """Return the connection's autocommit flag, or None if undetectable.

    Different drivers expose this differently:

    * psycopg2 / psycopg / sqlite3 → ``conn.autocommit`` is a bool attribute.
    * pymysql → ``conn.autocommit`` is a *method* (always truthy); the real
      flag is ``conn.get_autocommit()`` / ``conn.autocommit_mode`` (finding 4).

    Reading the bound method as a truthy value would always look like
    autocommit-on, so callable attributes are resolved via the accessor.
    """
    autocommit = getattr(conn, "autocommit", None)
    if callable(autocommit):
        getter = getattr(conn, "get_autocommit", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                return None
        mode = getattr(conn, "autocommit_mode", None)
        if mode is not None:
            return bool(mode)
        return None  # callable but no usable accessor — can't detect
    if autocommit is None:
        return None
    return bool(autocommit)


def _detect_autobegin(cursor: Any) -> None:
    """Set ``_in_transaction`` if the connection is in autobegin mode.

    DB-API drivers like psycopg2 default to ``autocommit=False``, which
    means the first statement implicitly starts a transaction at the
    C/driver level — no explicit ``BEGIN`` flows through
    ``cursor.execute()``.  We detect this by checking the cursor's
    connection: if ``autocommit`` is not ``True`` and we haven't already
    seen a ``BEGIN``, we treat the connection as having an implicit
    transaction.

    This is best-effort: if the connection doesn't expose ``autocommit``
    (e.g. sqlite3), we leave ``_in_transaction`` unchanged and fall back
    to statement-level tracking.
    """
    store = tx_store()
    if getattr(store, "_in_transaction", False):
        return  # already in a transaction
    conn = getattr(cursor, "connection", None)
    if conn is None:
        return
    autocommit = _connection_autocommit(conn)
    if autocommit is None:
        return  # driver doesn't expose autocommit — can't detect
    if not autocommit:
        # autocommit=False → autobegin: implicit transaction is active.
        # Set _is_autobegin so _report_or_buffer reports accesses
        # immediately (READ COMMITTED doesn't buffer) while still
        # tracking row locks via the _in_transaction flag.
        store._in_transaction = True
        store._is_autobegin = True
        store._tx_buffer = []
        store._tx_savepoints = {}


def _handle_tx_op(reporter: Any, tx: Any) -> None:
    """Apply a transaction-control operation (BEGIN/COMMIT/ROLLBACK/SAVEPOINT).

    Updates the per-thread transaction state, flushes the buffered access
    list on COMMIT, discards it on ROLLBACK, and releases any DPOR row
    locks held by the current thread on COMMIT/ROLLBACK.  Savepoints are
    implemented as buffer-index bookmarks.
    """
    store = tx_store()
    if tx is TxOp.BEGIN:
        store._in_transaction = True
        store._is_autobegin = False
        store._tx_buffer = []
        store._tx_savepoints = {}
    elif tx is TxOp.COMMIT:
        store._in_transaction = False
        store._is_autobegin = False
        buffer = getattr(store, "_tx_buffer", [])
        if reporter is not None:
            for res_id, kind in buffer:
                reporter(res_id, kind)
        store._tx_buffer = []
        store._tx_savepoints = {}
        _release_dpor_row_locks()
    elif tx is TxOp.ROLLBACK:
        store._in_transaction = False
        store._is_autobegin = False
        store._tx_buffer = []
        store._tx_savepoints = {}
        _release_dpor_row_locks()
    else:  # SavepointOp
        savepoints = getattr(store, "_tx_savepoints", {})
        if tx.op == "savepoint":
            buffer = getattr(store, "_tx_buffer", [])
            savepoints[tx.name] = len(buffer)
            store._tx_savepoints = savepoints
        elif tx.op == "rollback_to":
            if tx.name in savepoints:
                idx = savepoints[tx.name]
                buffer = getattr(store, "_tx_buffer", [])
                store._tx_buffer = buffer[:idx]
                stale = [name for name, sp_idx in savepoints.items() if sp_idx > idx]
                for name in stale:
                    del savepoints[name]
        else:  # "release"
            savepoints.pop(tx.name, None)


def handle_connection_commit() -> None:
    """Drive the COMMIT state machine for a Python-level ``conn.commit()``.

    DB-API drivers expose ``commit()`` / ``rollback()`` as connection methods;
    many call sites use those instead of issuing a textual ``COMMIT`` through
    ``cursor.execute()``.  Without intercepting them, the transaction buffer is
    never flushed, ``_in_transaction`` stays True forever, and DPOR row locks
    are held until thread exit (finding 3).  This flushes the buffer and clears
    the transaction state exactly like a textual COMMIT.
    """
    if not getattr(tx_store(), "_in_transaction", False):
        return
    from frontrun._io_detection import get_io_reporter
    from frontrun._sql_endpoint_suppression import suppress_sql_write

    suppress_sql_write("COMMIT")
    _handle_tx_op(get_io_reporter(), TxOp.COMMIT)


def handle_connection_rollback() -> None:
    """Drive the ROLLBACK state machine for a Python-level ``conn.rollback()``."""
    if not getattr(tx_store(), "_in_transaction", False):
        return
    from frontrun._io_detection import get_io_reporter
    from frontrun._sql_endpoint_suppression import suppress_sql_write

    suppress_sql_write("ROLLBACK")
    _handle_tx_op(get_io_reporter(), TxOp.ROLLBACK)


def reset_connection_state() -> None:
    """Clear per-thread SQL transaction state.

    Call this when a connection is returned to a connection pool to prevent
    stale ``_in_transaction`` / ``_tx_buffer`` / ``_tx_savepoints`` state
    from leaking across logical sessions.  Safe to call even when no
    transaction is active (it's a no-op in that case).
    """
    store = tx_store()
    for attr in (
        "_in_transaction",
        "_is_autobegin",
        "_tx_buffer",
        "_tx_savepoints",
        "_pending_row_locks",
        "_held_row_locks",
    ):
        if hasattr(store, attr):
            delattr(store, attr)
