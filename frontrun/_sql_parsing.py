"""SQL statement parsing for read/write table extraction.

Provides ``parse_sql_access(sql)`` which returns a :class:`SqlAccessResult`
for conflict detection. All statements go through sqlglot AST analysis;
string-based handling covers only constructs sqlglot cannot parse
(LOCK TABLE, SAVEPOINT, PREPARE, etc.).
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
    insert_tables: set[str] | None = None  # Explicit INSERT targets, including targets also read


_EMPTY = SqlAccessResult(set(), set(), None, None, None, None, None)


def _strip_quotes(name: str) -> str:
    """Remove surrounding quotes/backticks and extract table from schema.table.

    Dots inside a quoted identifier belong to the identifier, so only an
    unquoted dot separates schema/catalog components.
    """
    component_start = 0
    close_quote: str | None = None
    i = 0
    while i < len(name):
        char = name[i]
        if close_quote is None:
            if char in ('"', "`"):
                close_quote = char
            elif char == "[":
                close_quote = "]"
            elif char == ".":
                component_start = i + 1
        elif char == close_quote:
            # SQL identifiers escape their closing delimiter by doubling it.
            if i + 1 < len(name) and name[i + 1] == close_quote:
                i += 1
            else:
                close_quote = None
        i += 1

    last = name[component_start:]
    for open_quote, closing_quote in (('"', '"'), ("`", "`"), ("[", "]")):
        if len(last) >= 2 and last[0] == open_quote and last[-1] == closing_quote:
            return last[1:-1].replace(closing_quote * 2, closing_quote)
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


def _update_write_tables(node: Any) -> set[str]:
    """Tables written by an UPDATE, including MySQL multi-table updates.

    ``node.this`` is only the first table; a ``UPDATE t1 JOIN t2 ... SET
    t1.a=1, t2.b=2`` writes every table named on the left-hand side of a SET
    assignment.  We union those qualifying tables with ``node.this`` so multi-
    table writes are not demoted to reads (finding 8).
    """
    from sqlglot import exp  # type: ignore[import-untyped]

    tables: set[str] = set()
    if isinstance(node.this, exp.Table):
        tables.add(node.this.name)
    # Resolve SET-target column tables (aliases) back to physical table names.
    alias_to_table: dict[str, str] = {}
    for t in node.find_all(exp.Table):
        if t.alias:
            alias_to_table[t.alias] = t.name
        alias_to_table.setdefault(t.name, t.name)
    for assignment in node.args.get("expressions", []) or []:
        col = assignment.this if isinstance(assignment, exp.EQ) else None
        if isinstance(col, exp.Column) and col.table:
            tables.add(alias_to_table.get(col.table, col.table))
    return tables


def _delete_write_tables(node: Any) -> set[str]:
    """Tables deleted by a DELETE, including MySQL multi-table deletes.

    ``node.this`` is only the first target table; a ``DELETE t1, t2 FROM t1
    JOIN t2 ...`` deletes rows from every table listed before FROM, which
    sqlglot places in ``node.args['tables']``.  We union those with
    ``node.this`` so multi-table deletes are not demoted to reads.
    """
    from sqlglot import exp  # type: ignore[import-untyped]

    tables: set[str] = set()
    if isinstance(node.this, exp.Table):
        tables.add(node.this.name)
    for tbl in node.args.get("tables", []) or []:
        if isinstance(tbl, exp.Table):
            tables.add(tbl.name)
    return tables


def _is_descendant(node: Any, ancestor: Any) -> bool:
    """Whether *node* is contained in *ancestor* in a sqlglot AST."""
    current = node
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False


def _has_visible_cte_binding(table: Any, name: str, with_nodes: list[Any]) -> bool:
    """Whether an unqualified table occurrence is definitely a CTE reference.

    CTE visibility is lexical and belongs to one ``WITH`` node, not to the
    statement as a whole.  The main query sees every alias in its own ``WITH``;
    a CTE body sees earlier siblings and, under ``WITH RECURSIVE``, itself.
    Forward sibling visibility differs between supported SQL dialects, so it is
    deliberately not treated as proof of a query-local reference.
    """
    for with_node in with_nodes:
        query = with_node.parent
        if query is None or not _is_descendant(table, query):
            continue
        ctes = list(with_node.expressions)
        aliases = [cte.alias for cte in ctes]
        if name not in aliases:
            continue

        containing_index = next((i for i, cte in enumerate(ctes) if _is_descendant(table, cte)), None)
        if containing_index is None:
            return True
        if name in aliases[:containing_index]:
            return True
        if with_node.args.get("recursive") and name == aliases[containing_index]:
            return True
    return False


def _query_local_cte_aliases(ast: Any, cte_aliases: set[str], write_tables: set[str]) -> set[str]:
    """Return aliases proven to have no physical table occurrence.

    Access sets store unqualified names, so subtracting a CTE alias is safe only
    when *every* occurrence of that name is proven query-local.  Qualified
    references, DML targets, non-recursive self references, and dialect-
    ambiguous forward references all retain the name.  This may over-merge two
    scopes that reuse an alias, but can never erase a real dependency.
    """
    from sqlglot import exp  # type: ignore[import-untyped]

    physical_names = cte_aliases & write_tables
    with_nodes = list(ast.find_all(exp.With))
    for table in ast.find_all(exp.Table):
        name = table.name
        if name not in cte_aliases or name in physical_names:
            continue
        if table.catalog or table.db or not _has_visible_cte_binding(table, name, with_nodes):
            physical_names.add(name)
    return cte_aliases - physical_names


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

        # MySQL: LOCK TABLES <tbl> [AS alias] <READ|WRITE> [, ...] — sqlglot ERROR.
        # Distinct grammar from the PostgreSQL "LOCK TABLE ... IN ... MODE" form
        # handled below: here each table carries its own trailing lock type.
        if upper.startswith("LOCK TABLES "):
            rest = stripped[12:].strip()
            read_tables: set[str] = set()
            write_tables: set[str] = set()
            mysql_lock_intent: LockIntent = LockIntent.SHARE
            for entry in rest.split(","):
                tokens = entry.strip().split()
                if not tokens:
                    continue
                # Drop a trailing READ / WRITE / "LOW_PRIORITY WRITE" / "READ LOCAL".
                lock_words = {"READ", "WRITE", "LOCAL", "LOW_PRIORITY"}
                name_tokens = [t for t in tokens if t.upper() not in lock_words]
                # The table name is the first token; any "AS alias" tokens follow it.
                tbl_name = _strip_quotes(name_tokens[0]) if name_tokens else None
                if tbl_name is None:
                    continue
                if any(t.upper() == "WRITE" for t in tokens):
                    write_tables.add(tbl_name)
                    mysql_lock_intent = LockIntent.UPDATE
                else:
                    read_tables.add(tbl_name)
            if read_tables or write_tables:
                return SqlAccessResult(read_tables, write_tables, mysql_lock_intent, None, None)
            return SqlAccessResult(set(), set(), mysql_lock_intent, None, None)

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
    # sqlglot default dialect chokes on (misinterprets % as modulo).  Skip
    # single-quoted string literals so a literal like ``'a%sb'`` is left
    # untouched and yields the same resource ID via the parameterized and
    # literal paths (finding 7).
    if "%" in sql:
        from frontrun._sql_params import _split_literal_segments

        def _rewrite(segment: str) -> str:
            segment = segment.replace("%%", "\x00")
            segment = re.sub(r"%(?:\(\w+\))?s", "?", segment)
            return segment.replace("\x00", "%")

        sql = "".join(text if is_lit else _rewrite(text) for text, is_lit in _split_literal_segments(sql))

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
    all_insert: set[str] = set()
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

            # CTE nodes are needed three times below (lock intent, alias
            # filtering, data-modifying classification); walk the tree once.
            ctes = list(ast.find_all(exp.CTE))

            # Also extract lock intent from CTEs (e.g. WITH cte AS (SELECT ... FOR UPDATE SKIP LOCKED))
            for cte_node in ctes:
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

            # CTE alias occurrences that are proven query-local must not be
            # reported as tables (finding 1).  The same unqualified name may
            # still denote a physical table in another lexical scope.
            cte_aliases = {c.alias for c in ctes if c.alias}

            def _classify_node(node: Any, *, top_level: bool) -> None:
                """Add reads/writes for a single DML node into the shared sets."""
                if isinstance(node, exp.Insert):
                    tbl = node.this
                    if isinstance(tbl, exp.Schema):
                        tbl = tbl.this  # INSERT INTO t(cols) → Schema wraps Table
                    target_node = tbl if isinstance(tbl, exp.Table) else None
                    if target_node is not None:
                        write.add(target_node.name)
                        all_insert.add(target_node.name)
                    # Source tables: read every table except the INSERT target
                    # node itself.  Identity comparison (not name) preserves
                    # self-table INSERT...SELECT (read AND write of the same
                    # name), while still picking up sources referenced via a CTE
                    # (WITH src AS ... INSERT ... SELECT * FROM src).
                    for t in node.find_all(exp.Table):
                        if t is not target_node:
                            read.add(t.name)
                elif isinstance(node, (exp.Update, exp.Delete)):
                    if isinstance(node, exp.Update):
                        targets = _update_write_tables(node)
                    else:
                        targets = _delete_write_tables(node)
                    for name in targets:
                        write.add(name)
                        read.add(name)
                    if isinstance(node, exp.Delete):
                        all_delete.update(targets)
                    for t in node.find_all(exp.Table):
                        if t.name not in write:
                            read.add(t.name)
                elif isinstance(node, exp.Merge):
                    target = node.this
                    if isinstance(target, exp.Table):
                        write.add(target.name)
                        read.add(target.name)
                        for when in node.find_all(exp.When):
                            action = when.args.get("then")
                            if isinstance(action, exp.Insert):
                                all_insert.add(target.name)
                            elif isinstance(action, exp.Var) and action.name.upper() == "DELETE":
                                all_delete.add(target.name)
                    for t in node.find_all(exp.Table):
                        if t.name not in write:
                            read.add(t.name)
                elif isinstance(node, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
                    # SELECT ... INTO <table> creates and populates the target
                    # table — a write, not a read (classifying it as a read
                    # under-merges).  The INTO clause may sit on the top-level
                    # SELECT or on a SELECT inside a set operation.
                    into_target_ids: set[int] = set()
                    selects = [node] if isinstance(node, exp.Select) else node.find_all(exp.Select)
                    for select_node in selects:
                        into = select_node.args.get("into")
                        into_tbl = into.this if into is not None else None
                        if isinstance(into_tbl, exp.Table):
                            write.add(into_tbl.name)
                            into_target_ids.add(id(into_tbl))
                    for t in node.find_all(exp.Table):
                        if id(t) not in into_target_ids:
                            read.add(t.name)
                elif top_level:
                    # DDL, GRANT, etc. — conservatively treat as write.
                    for t in node.find_all(exp.Table):
                        write.add(t.name)

            # Classify data-modifying CTEs first so their write targets are in
            # the write set before the top-level node's source-read pass runs.
            for cte_node in ctes:
                inner = cte_node.this
                if isinstance(inner, (exp.Update, exp.Delete, exp.Insert, exp.Merge)):
                    _classify_node(inner, top_level=False)

            _classify_node(ast, top_level=True)

            # Drop only aliases whose every occurrence is proven query-local.
            # Writes are always physical targets (CTEs cannot be mutated by
            # INSERT/UPDATE/DELETE/MERGE), so a colliding write name survives.
            query_local = _query_local_cte_aliases(ast, cte_aliases, write)
            read -= query_local

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
        all_insert if all_insert else None,
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
