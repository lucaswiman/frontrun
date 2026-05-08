"""SQL statement parsing for read/write table extraction.

Provides ``parse_sql_access(sql)`` which returns a :class:`SqlAccessResult`
for conflict detection. Uses a regex fast-path for simple statements and falls
back to sqlglot for complex SQL (CTEs, subqueries, UNION, etc.).
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

# ---------------------------------------------------------------------------
# Typed enums for lock intent and transaction operations
# ---------------------------------------------------------------------------


class LockIntent(enum.Enum):
    """Lock mode extracted from SQL statements (FOR UPDATE, FOR SHARE, LOCK TABLE)."""

    UPDATE = "UPDATE"
    SHARE = "SHARE"
    UPDATE_SKIP_LOCKED = "UPDATE_SKIP_LOCKED"


class TxOp(enum.Enum):
    """Simple transaction control operations."""

    BEGIN = "BEGIN"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True)
class SavepointOp:
    """Tagged SAVEPOINT / ROLLBACK TO / RELEASE operation."""

    op: Literal["savepoint", "rollback_to", "release"]
    name: str


TxControl = TxOp | SavepointOp


class SqlAccessResult(NamedTuple):
    """Result of parsing a SQL statement for read/write table extraction."""

    read_tables: set[str]
    write_tables: set[str]
    lock_intent: LockIntent | None
    tx_op: TxControl | None
    temporal_clauses: dict[str, str] | None
    ast: Any | None = None  # Pre-parsed sqlglot AST (when available from _sqlglot_parse)
    delete_tables: set[str] | None = None  # Tables targeted by DELETE (for phantom read detection)


_EMPTY = SqlAccessResult(set(), set(), None, None, None, None, None)


def _strip_quotes(name: str) -> str:
    """Remove surrounding quotes/backticks and extract table from schema.table.

    Handles quoted schema-qualified names like "public"."users" and
    `myschema`.`users` by splitting on "." or `.` boundaries first,
    then stripping quotes from the last component.
    """
    # Split on dot boundaries between quoted identifiers: "schema"."table"
    # or `schema`.`table`.  Also handles unquoted schema.table.
    if '"."' in name:
        parts = name.split('"."')
        last = parts[-1]
    elif "`.`" in name:
        parts = name.split("`.`")
        last = parts[-1]
    elif "].[" in name:
        parts = name.split("].[")
        last = parts[-1]
    elif '".`' in name or '`."' in name:
        # Mixed quoting — unlikely but handle gracefully
        last = name.rsplit(".", 1)[-1]
    else:
        # No "." or `.` boundary detected.  Split on plain dot to
        # separate schema from table, then strip quotes from the
        # resulting table component.  The previous code assumed a
        # leading quote meant the *entire* string was quoted and used
        # [1:-1] — but for mixed cases like "public".users the last
        # character is part of the table name, not a closing quote.
        last = name.rsplit(".", 1)[-1]

    # Strip remaining quotes from the last component
    if last.startswith(('"', "`", "[")):
        last = last[1:]
    if last.endswith(('"', "`", "]")):
        last = last[:-1]
    return last


def _merge_lock_intent(a: LockIntent | None, b: LockIntent | None) -> LockIntent | None:
    """Merge two lock intents, preferring UPDATE > UPDATE_SKIP_LOCKED > SHARE."""
    if a is LockIntent.UPDATE or b is LockIntent.UPDATE:
        return LockIntent.UPDATE
    if a is LockIntent.UPDATE_SKIP_LOCKED or b is LockIntent.UPDATE_SKIP_LOCKED:
        return LockIntent.UPDATE_SKIP_LOCKED
    if a is LockIntent.SHARE or b is LockIntent.SHARE:
        return LockIntent.SHARE
    return None


def _sqlglot_parse(sql: str) -> SqlAccessResult | None:
    """Parse a SQL statement and return table access information.

    Handles all statement types:
    - DML (SELECT/INSERT/UPDATE/DELETE), CTEs, UNION, MERGE via sqlglot AST
    - COPY, ROLLBACK TO SAVEPOINT, SET AUTOCOMMIT via sqlglot AST
    - Constructs sqlglot cannot parse (LOCK TABLE, SAVEPOINT, RELEASE,
      DEALLOCATE, PREPARE, EXECUTE, START TRANSACTION, END) via string checks

    Returns a single merged SqlAccessResult, or None if parsing fails entirely
    (endpoint-level I/O detection remains as fallback).
    The ``ast`` field is populated for single-statement SQL from the sqlglot parse.
    """
    try:
        import sqlglot  # type: ignore[import-untyped]
        from sqlglot import errors as sqlglot_errors  # type: ignore[import-untyped]
        from sqlglot import exp  # type: ignore[import-untyped]
    except ImportError:
        return None

    # ---------------------------------------------------------------------------
    # Pre-checks: constructs sqlglot cannot parse, handled via string operations.
    # Only applies to single-statement SQL (no interior semicolons).
    # ---------------------------------------------------------------------------
    stripped = sql.strip().rstrip(";").strip()
    if ";" not in stripped:
        upper = stripped.upper()

        # START TRANSACTION — sqlglot misparses as Alias
        if upper == "START TRANSACTION" or upper.startswith("START TRANSACTION "):
            return SqlAccessResult(set(), set(), None, TxOp.BEGIN, None)

        # END / END TRANSACTION / END WORK — PostgreSQL aliases for COMMIT.
        # sqlglot parses bare ``END`` as a Column identifier and
        # ``END TRANSACTION`` / ``END WORK`` as an Alias, so none of them
        # reach the normal ``exp.Commit`` path.
        if upper == "END" or upper in ("END TRANSACTION", "END WORK"):
            return SqlAccessResult(set(), set(), None, TxOp.COMMIT, None)

        # ABORT / ABORT TRANSACTION / ABORT WORK — PostgreSQL aliases for
        # ROLLBACK.  sqlglot parses bare ``ABORT`` as a Column identifier
        # and the multi-word variants as Alias.
        if upper == "ABORT" or upper in ("ABORT TRANSACTION", "ABORT WORK"):
            return SqlAccessResult(set(), set(), None, TxOp.ROLLBACK, None)

        # SAVEPOINT <name> — sqlglot misparses as Alias
        if upper.startswith("SAVEPOINT "):
            parts = stripped[10:].strip().split()
            if parts:
                return SqlAccessResult(set(), set(), None, SavepointOp("savepoint", _strip_quotes(parts[0])), None)

        # RELEASE [SAVEPOINT] <name> — sqlglot ERROR
        if upper.startswith("RELEASE "):
            rest = stripped[8:].strip()
            if rest.upper().startswith("SAVEPOINT "):
                rest = rest[10:].strip()
            parts = rest.split()
            if parts:
                return SqlAccessResult(set(), set(), None, SavepointOp("release", _strip_quotes(parts[0])), None)

        # LOCK TABLE <table>[, <table>...] [IN <mode> MODE] — sqlglot ERROR for all dialects
        if upper.startswith("LOCK TABLE "):
            rest = stripped[11:].strip()
            in_idx = rest.upper().find(" IN ")
            tbl_raw = rest[:in_idx].strip() if in_idx > 0 else rest.strip()
            tables = {_strip_quotes(t.strip()) for t in tbl_raw.split(",")}
            table_lock_intent: LockIntent = LockIntent.UPDATE
            if in_idx > 0:
                mode_part = rest[in_idx + 4 :].upper()
                mode_end = mode_part.find(" MODE")
                mode = mode_part[:mode_end] if mode_end > 0 else mode_part
                if "SHARE" in mode and "EXCLUSIVE" not in mode:
                    table_lock_intent = LockIntent.SHARE
            return SqlAccessResult(set(), tables, table_lock_intent, None, None)

        # DEALLOCATE [PREPARE] <name> | DEALLOCATE ALL — sqlglot misparses
        if upper.startswith("DEALLOCATE "):
            return SqlAccessResult(set(), set(), None, None, None)

        # PREPARE <name> AS <sql> — sqlglot treats as opaque Command
        if upper.startswith("PREPARE "):
            as_idx = upper.find(" AS ")
            if as_idx > 0:
                inner_sql = stripped[as_idx + 4 :].strip()
                if inner_sql:
                    inner = _sqlglot_parse(inner_sql)
                    if inner is not None:
                        return inner
            return SqlAccessResult(set(), set(), None, None, None)

        # EXECUTE <name> [(params)] — opaque without a prepared stmt registry
        if upper.startswith("EXECUTE "):
            return SqlAccessResult(set(), set(), None, None, None)

    # Pre-process pyformat parameter placeholders (%s, %(name)s) which
    # sqlglot default dialect chokes on (misinterprets % as modulo).
    if "%" in sql:
        sql = sql.replace("%%", "\x00")
        sql = re.sub(r"%(?:\(\w+\))?s", "?", sql)
        sql = sql.replace("\x00", "%")

    try:
        expressions = sqlglot.parse(sql)
    except sqlglot_errors.ParseError:
        expressions = None

    # Fallback dialects: mysql handles backtick identifiers and MySQL-specific syntax
    # (ON DUPLICATE KEY UPDATE, etc.); tsql handles FOR SYSTEM_TIME.
    # Also re-parse with mysql when backticks are present even if default succeeded,
    # since the default dialect may misparse backtick-quoted identifiers.
    sql_upper = sql.upper()
    if not expressions or "`" in sql:
        try:
            mysql_exprs = sqlglot.parse(sql, read="mysql")
            if mysql_exprs:
                expressions = mysql_exprs
        except sqlglot_errors.ParseError:
            pass
    if not expressions and "FOR SYSTEM_TIME" in sql_upper:
        try:
            expressions = sqlglot.parse(sql, read="tsql")
        except sqlglot_errors.ParseError:
            pass
    if not expressions:
        return None  # unparseable → fall back to endpoint-level

    all_write: set[str] = set()
    all_read: set[str] = set()
    all_delete: set[str] = set()
    all_lock_intent: LockIntent | None = None
    all_tx_op: TxControl | None = None
    all_temporal: dict[str, str] | None = None
    first_ast: Any | None = None

    for ast in expressions:
        if ast is None:
            continue

        if first_ast is None:
            first_ast = ast

        write: set[str] = set()
        read: set[str] = set()
        lock_intent: LockIntent | None = None
        tx_op: TxControl | None = None

        # Transaction control
        if isinstance(ast, exp.Transaction):
            tx_op = TxOp.BEGIN
        elif isinstance(ast, exp.Commit):
            tx_op = TxOp.COMMIT
        elif isinstance(ast, exp.Rollback):
            sp = ast.args.get("savepoint")
            if sp:
                tx_op = SavepointOp("rollback_to", sp.name)
            else:
                tx_op = TxOp.ROLLBACK
        elif isinstance(ast, exp.Set):
            # SET AUTOCOMMIT = 0 → BEGIN, SET AUTOCOMMIT = 1 → COMMIT
            for item in ast.find_all(exp.SetItem):
                eq = item.this
                if eq and isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column):
                    if eq.this.name.upper() == "AUTOCOMMIT" and isinstance(eq.expression, exp.Literal):
                        tx_op = TxOp.BEGIN if eq.expression.this == "0" else TxOp.COMMIT
        elif isinstance(ast, exp.Copy):
            # COPY table FROM (write) / TO (read); COPY (subquery) TO → no table name
            tbl_node = ast.this
            if isinstance(tbl_node, exp.Schema):
                tbl_node = tbl_node.this  # COPY table(cols) → Schema wraps Table
            if isinstance(tbl_node, exp.Table):
                tbl_name = tbl_node.name
                if ast.args.get("kind"):  # kind=True means FROM (write into table)
                    write.add(tbl_name)
                else:
                    read.add(tbl_name)
        else:
            # Extract lock intent from SELECT (including inside CTEs)
            def _extract_lock_intent_from_select(select_node: exp.Expression) -> LockIntent | None:
                """Extract lock intent from a SELECT, checking FOR UPDATE SKIP LOCKED."""
                lock_node = select_node.find(exp.Lock)
                if not lock_node:
                    return None
                if lock_node.args.get("update"):
                    # wait=False means SKIP LOCKED in sqlglot
                    if lock_node.args.get("wait") is False:
                        return LockIntent.UPDATE_SKIP_LOCKED
                    return LockIntent.UPDATE
                # Not update → share lock
                intent = LockIntent.SHARE
                kind_val = lock_node.args.get("kind")
                if kind_val:
                    kind_upper = str(kind_val).upper()
                    if "UPDATE" in kind_upper:
                        intent = LockIntent.UPDATE
                    elif "SHARE" in kind_upper:
                        intent = LockIntent.SHARE
                return intent

            if isinstance(ast, exp.Select):
                lock_intent = _extract_lock_intent_from_select(ast)

            # Also extract lock intent from CTEs (e.g. WITH cte AS (SELECT ... FOR UPDATE SKIP LOCKED))
            for cte_node in ast.find_all(exp.CTE):
                cte_intent = _extract_lock_intent_from_select(cte_node.this)
                if cte_intent is not None:
                    lock_intent = _merge_lock_intent(lock_intent, cte_intent)

            # Advisory locks (PostgreSQL, MySQL)
            for call in ast.find_all(exp.Anonymous):
                name = call.this.lower()
                if name in (
                    "pg_advisory_lock",
                    "pg_advisory_xact_lock",
                    "pg_advisory_lock_shared",
                    "pg_advisory_xact_lock_shared",
                    "get_lock",
                ):
                    # Extract lock ID/name if it's a literal
                    if call.expressions:
                        lock_ids: list[str] = []
                        # For get_lock, the second argument is a timeout, not part of the ID
                        args_to_use = call.expressions[:1] if name == "get_lock" else call.expressions
                        for arg in args_to_use:
                            if isinstance(arg, exp.Literal):
                                lock_ids.append(str(arg.this))
                            else:
                                lock_ids.append("?")
                        lock_id_str = ":".join(lock_ids)
                        write.add(f"advisory_lock:{lock_id_str}")
                        if "shared" in name:
                            lock_intent = _merge_lock_intent(lock_intent, LockIntent.SHARE)
                        else:
                            lock_intent = _merge_lock_intent(lock_intent, LockIntent.UPDATE)

            # Shared table visitor logic
            for t in ast.find_all(exp.Table):
                # Check for system versioning (FOR SYSTEM_TIME)
                version = t.find(exp.Version)
                if version:
                    clause = str(version)
                    # Standardize: sqlglot often translates to "FOR TIMESTAMP" in its internal representation
                    clause = clause.replace("FOR TIMESTAMP ", "FOR SYSTEM_TIME ")
                    # Extract only the predicate part
                    clause = clause.replace("FOR SYSTEM_TIME ", "").strip()
                    if all_temporal is None:
                        all_temporal = {}
                    all_temporal[t.name] = clause

            if isinstance(ast, exp.Insert):
                tbl = ast.find(exp.Table)
                if tbl:
                    write.add(tbl.name)
                # Source tables (everything after the target)
                if ast.expression:  # the SELECT source
                    for t in ast.expression.find_all(exp.Table):
                        read.add(t.name)
            elif isinstance(ast, (exp.Update, exp.Delete)):
                tbl = ast.this
                if isinstance(tbl, exp.Table):
                    write.add(tbl.name)
                    read.add(tbl.name)
                    if isinstance(ast, exp.Delete):
                        all_delete.add(tbl.name)
                for t in ast.find_all(exp.Table):
                    if t.name not in write:
                        read.add(t.name)
            elif isinstance(ast, exp.Select):
                for t in ast.find_all(exp.Table):
                    read.add(t.name)
            elif isinstance(ast, (exp.Union, exp.Intersect, exp.Except)):
                for t in ast.find_all(exp.Table):
                    read.add(t.name)
            elif isinstance(ast, exp.Merge):
                target = ast.this
                if isinstance(target, exp.Table):
                    write.add(target.name)
                    read.add(target.name)
                # All non-target tables are read sources
                for t in ast.find_all(exp.Table):
                    if t.name not in write:
                        read.add(t.name)
            else:
                # DDL, GRANT, etc. — conservatively treat as write
                for t in ast.find_all(exp.Table):
                    write.add(t.name)

        all_read.update(read)
        all_write.update(write)
        all_lock_intent = _merge_lock_intent(all_lock_intent, lock_intent)
        if tx_op:
            all_tx_op = tx_op  # Take the last tx_op

    # For single-statement SQL, attach AST so callers can avoid re-parsing
    result_ast = first_ast if len([e for e in expressions if e is not None]) == 1 else None
    return SqlAccessResult(
        all_read,
        all_write,
        all_lock_intent,
        all_tx_op,
        all_temporal,
        result_ast,
        all_delete if all_delete else None,
    )


def parse_sql_access(sql: str) -> SqlAccessResult:
    """Extract table access info from a SQL statement.

    Returns empty sets if parsing fails entirely
    (endpoint-level I/O detection remains as fallback).
    """
    result = _sqlglot_parse(sql)
    if result is not None:
        return result

    # Parse failure: return empty sets → endpoint-level fallback
    return _EMPTY
