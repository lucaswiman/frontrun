"""Tests for SQL statement parsing (read/write table extraction).

Covers:
- _sqlglot_parse: SELECT, INSERT, UPDATE, DELETE, JOINs, CTEs, UNION, MERGE,
  COPY, LOCK TABLE, SAVEPOINT, RELEASE, PREPARE/EXECUTE/DEALLOCATE,
  SET AUTOCOMMIT, ROLLBACK TO SAVEPOINT
- parse_sql_access: end-to-end routing and fallback behaviour
- Edge cases: quoted identifiers, schema qualification, whitespace, semicolons
"""

from __future__ import annotations

import pytest

from frontrun._sql_parsing import (
    LockIntent,
    TxOp,
    _sqlglot_parse,
    _strip_quotes,
    parse_sql_access,
)

# ---------------------------------------------------------------------------
# _strip_quotes
# ---------------------------------------------------------------------------


class TestStripQuotes:
    def test_unquoted_simple(self):
        assert _strip_quotes("users") == "users"

    def test_double_quoted(self):
        assert _strip_quotes('"My Table"') == "My Table"

    def test_backtick_quoted(self):
        assert _strip_quotes("`orders`") == "orders"

    def test_schema_qualified_unquoted(self):
        assert _strip_quotes("public.users") == "users"

    def test_schema_qualified_double_quoted(self):
        # "public"."users" → extract table name 'users' from schema-qualified quoted identifier
        assert _strip_quotes('"public"."users"') == "users"

    def test_deep_dotted(self):
        assert _strip_quotes("db.schema.table") == "table"

    def test_no_quotes_no_schema(self):
        assert _strip_quotes("accounts") == "accounts"

    def test_quoted_schema_unquoted_table(self):
        """Bug: _strip_quotes('"public".users') should return 'users' not 'user'.

        When the schema is quoted but the table is not, the [1:-1] slice
        intended to strip surrounding quotes incorrectly removes the last
        character of the table name instead of a closing quote.
        """
        assert _strip_quotes('"public".users') == "users"

    def test_backtick_schema_unquoted_table(self):
        """Same bug with backtick-quoted schema and unquoted table."""
        assert _strip_quotes("`myschema`.orders") == "orders"

    def test_unquoted_schema_quoted_table(self):
        """When schema is unquoted but table is quoted, extract table correctly."""
        assert _strip_quotes('public."Special Table"') == "Special Table"

    def test_backtick_schema_backtick_table(self):
        """Backtick-quoted schema and backtick-quoted table."""
        assert _strip_quotes("`myschema`.`users`") == "users"

    def test_bracket_quoted_simple(self):
        assert _strip_quotes("[users]") == "users"

    def test_bracket_quoted_schema_qualified(self):
        assert _strip_quotes("[dbo].[users]") == "users"

    def test_bracket_quoted_table_only(self):
        assert _strip_quotes("dbo.[users]") == "users"

    def test_double_quoted_name_with_interior_dot(self):
        """A fully-quoted identifier may contain a dot as part of the name.

        Splitting ``"my.table"`` on the interior dot treats it as a schema
        separator and returns ``'table'`` — a different resource id from the
        ``my.table`` that sqlglot extracts for the same name in DML, so a
        ``LOCK TABLE`` never conflicts with the rows it guards (an under-merge).
        """
        assert _strip_quotes('"my.table"') == "my.table"

    def test_backtick_quoted_name_with_interior_dot(self):
        assert _strip_quotes("`my.table`") == "my.table"

    def test_bracket_quoted_name_with_interior_dot(self):
        assert _strip_quotes("[my.table]") == "my.table"


# ---------------------------------------------------------------------------
# _sqlglot_parse — non-DML constructs now handled inside _sqlglot_parse
# ---------------------------------------------------------------------------


class TestSqlglotParseNonDml:
    """Constructs previously handled by the regex path, now absorbed into _sqlglot_parse."""

    @pytest.fixture(autouse=True)
    def _require_sqlglot(self):
        pytest.importorskip("sqlglot")

    # -----------------------------------------------------------------------
    # Transaction control — string-checked before sqlglot call
    # -----------------------------------------------------------------------

    def test_start_transaction(self):
        r = _sqlglot_parse("START TRANSACTION")
        assert r is not None and r.tx_op is TxOp.BEGIN

    def test_end_as_commit(self):
        r = _sqlglot_parse("END")
        assert r is not None and r.tx_op is TxOp.COMMIT

    def test_end_transaction_as_commit(self):
        """``END TRANSACTION`` is a PostgreSQL alias for COMMIT.

        sqlglot misparses it as ``Alias`` (``END AS TRANSACTION``), so the
        parser must detect it via the string pre-checks alongside the bare
        ``END``.
        """
        r = _sqlglot_parse("END TRANSACTION")
        assert r is not None and r.tx_op is TxOp.COMMIT

    def test_end_work_as_commit(self):
        """``END WORK`` is a PostgreSQL alias for COMMIT."""
        r = _sqlglot_parse("END WORK")
        assert r is not None and r.tx_op is TxOp.COMMIT

    def test_abort_as_rollback(self):
        """``ABORT`` is a PostgreSQL alias for ROLLBACK.

        sqlglot parses bare ``ABORT`` as a ``Column`` identifier, so the
        parser must recognise it via the string pre-checks.
        """
        r = _sqlglot_parse("ABORT")
        assert r is not None and r.tx_op is TxOp.ROLLBACK

    def test_abort_work_as_rollback(self):
        r = _sqlglot_parse("ABORT WORK")
        assert r is not None and r.tx_op is TxOp.ROLLBACK

    def test_abort_transaction_as_rollback(self):
        r = _sqlglot_parse("ABORT TRANSACTION")
        assert r is not None and r.tx_op is TxOp.ROLLBACK

    def test_savepoint(self):
        r = _sqlglot_parse("SAVEPOINT sp1")
        from frontrun._sql_parsing import SavepointOp

        assert r is not None and r.tx_op == SavepointOp("savepoint", "sp1")

    def test_release_savepoint(self):
        r = _sqlglot_parse("RELEASE SAVEPOINT sp1")
        from frontrun._sql_parsing import SavepointOp

        assert r is not None and r.tx_op == SavepointOp("release", "sp1")

    def test_release_no_keyword(self):
        r = _sqlglot_parse("RELEASE sp1")
        from frontrun._sql_parsing import SavepointOp

        assert r is not None and r.tx_op == SavepointOp("release", "sp1")

    def test_rollback_to_savepoint(self):
        r = _sqlglot_parse("ROLLBACK TO SAVEPOINT sp1")
        from frontrun._sql_parsing import SavepointOp

        assert r is not None and r.tx_op == SavepointOp("rollback_to", "sp1")

    def test_savepoint_quoted_name_strips_quotes(self):
        """SAVEPOINT with quoted name should strip quotes so lookups match ROLLBACK TO."""
        from frontrun._sql_parsing import SavepointOp

        r = _sqlglot_parse('SAVEPOINT "sp1"')
        assert r is not None and r.tx_op == SavepointOp("savepoint", "sp1"), (
            f"Quoted savepoint name should be stripped to 'sp1', got {r!r}"
        )

    def test_release_quoted_name_strips_quotes(self):
        """RELEASE with quoted name should strip quotes for consistency."""
        from frontrun._sql_parsing import SavepointOp

        r = _sqlglot_parse('RELEASE SAVEPOINT "sp1"')
        assert r is not None and r.tx_op == SavepointOp("release", "sp1"), (
            f"Quoted release savepoint name should be stripped to 'sp1', got {r!r}"
        )

    def test_set_autocommit_0(self):
        r = _sqlglot_parse("SET AUTOCOMMIT = 0")
        assert r is not None and r.tx_op is TxOp.BEGIN

    def test_set_autocommit_1(self):
        r = _sqlglot_parse("SET AUTOCOMMIT = 1")
        assert r is not None and r.tx_op is TxOp.COMMIT

    # -----------------------------------------------------------------------
    # LOCK TABLE
    # -----------------------------------------------------------------------

    def test_lock_table_exclusive(self):
        r = _sqlglot_parse("LOCK TABLE users IN EXCLUSIVE MODE")
        assert r is not None
        assert "users" in r.write_tables
        assert r.lock_intent is LockIntent.UPDATE

    def test_lock_table_share(self):
        r = _sqlglot_parse("LOCK TABLE users IN SHARE MODE")
        assert r is not None
        assert "users" in r.write_tables
        assert r.lock_intent is LockIntent.SHARE

    def test_lock_table_no_mode(self):
        r = _sqlglot_parse("LOCK TABLE orders")
        assert r is not None
        assert "orders" in r.write_tables
        assert r.lock_intent is LockIntent.UPDATE

    def test_lock_table_quoted_name_with_interior_dot(self):
        """LOCK TABLE on a quoted name containing a dot must lock that exact
        table, matching the resource id DML uses for the same name — otherwise
        the lock guards a phantom table and never conflicts (under-merge)."""
        r = parse_sql_access('LOCK TABLE "my.table" IN EXCLUSIVE MODE')
        assert r.write_tables == {"my.table"}

    # -----------------------------------------------------------------------
    # COPY
    # -----------------------------------------------------------------------

    def test_copy_from_stdin(self):
        r = _sqlglot_parse("COPY users FROM STDIN")
        assert r is not None
        assert "users" in r.write_tables and r.read_tables == set()

    def test_copy_to_stdout(self):
        r = _sqlglot_parse("COPY orders TO STDOUT")
        assert r is not None
        assert "orders" in r.read_tables and r.write_tables == set()

    def test_copy_with_columns(self):
        r = _sqlglot_parse("COPY users (name, email) FROM STDIN")
        assert r is not None
        assert "users" in r.write_tables

    # -----------------------------------------------------------------------
    # DEALLOCATE / PREPARE / EXECUTE
    # -----------------------------------------------------------------------

    def test_deallocate(self):
        r = _sqlglot_parse("DEALLOCATE my_stmt")
        assert r is not None and r.read_tables == set() and r.write_tables == set()

    def test_prepare_select(self):
        r = _sqlglot_parse("PREPARE my_q AS SELECT * FROM users WHERE id = $1")
        assert r is not None and "users" in r.read_tables

    def test_prepare_insert(self):
        r = _sqlglot_parse("PREPARE ins AS INSERT INTO orders (total) VALUES ($1)")
        assert r is not None and "orders" in r.write_tables

    def test_execute(self):
        r = _sqlglot_parse("EXECUTE my_q")
        assert r is not None and r.read_tables == set() and r.write_tables == set()

    # -----------------------------------------------------------------------
    # Miscellaneous
    # -----------------------------------------------------------------------

    def test_empty_string(self):
        r = _sqlglot_parse("")
        # Either None or empty result is acceptable; must not crash
        assert r is None or (r.read_tables == set() and r.write_tables == set())

    def test_whitespace_only(self):
        r = _sqlglot_parse("   \n\t  ")
        assert r is None or (r.read_tables == set() and r.write_tables == set())


# ---------------------------------------------------------------------------
# _sqlglot_parse — full parser (requires sqlglot)
# ---------------------------------------------------------------------------


class TestSqlglotParse:
    @pytest.fixture(autouse=True)
    def _require_sqlglot(self):
        pytest.importorskip("sqlglot")

    def test_simple_select(self):
        r, w, *_ = _sqlglot_parse("SELECT id FROM users WHERE id = 1")
        assert r == {"users"} and w == set()

    def test_simple_insert(self):
        r, w, *_ = _sqlglot_parse("INSERT INTO orders (amount) VALUES (99)")
        assert r == set() and w == {"orders"}

    def test_simple_update(self):
        r, w, *_ = _sqlglot_parse("UPDATE accounts SET balance = 0 WHERE id = 1")
        assert "accounts" in w and "accounts" in r

    def test_simple_delete(self):
        r, w, *_ = _sqlglot_parse("DELETE FROM sessions WHERE id = 1")
        assert "sessions" in w and "sessions" in r

    def test_cte_select(self):
        sql = "WITH recent AS (SELECT * FROM orders WHERE created > '2024-01-01') SELECT * FROM recent"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in r and w == set()

    def test_cte_multiple(self):
        sql = "WITH a AS (SELECT * FROM t1), b AS (SELECT * FROM t2) SELECT * FROM a JOIN b ON a.id = b.id"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "t1" in r and "t2" in r and w == set()

    def test_subquery_in_where(self):
        sql = "SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE active = true)"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in r and "users" in r and w == set()

    def test_union_select(self):
        # UNION produces a Union AST node. These are now handled explicitly as reads.
        sql = "SELECT id FROM customers UNION SELECT id FROM vendors"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "customers" in r and "vendors" in r and w == set()

    def test_insert_select_simple(self):
        sql = "INSERT INTO archive SELECT * FROM orders WHERE status = 'closed'"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert w == {"archive"}
        assert "orders" in r

    def test_insert_select_with_join(self):
        sql = (
            "INSERT INTO summary (user_id, total) "
            "SELECT u.id, SUM(o.amount) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id"
        )
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert w == {"summary"}
        assert "users" in r and "orders" in r

    def test_insert_select_same_table(self):
        """INSERT INTO t SELECT * FROM t must report t as both read AND written."""
        sql = "INSERT INTO orders SELECT * FROM orders WHERE status = 'old'"
        r, w, *_ = _sqlglot_parse(sql)
        assert w == {"orders"}
        assert r is not None
        assert "orders" in r, "self-table INSERT...SELECT must include the table in reads"

    def test_update_from_join(self):
        sql = "UPDATE orders SET status = 'shipped' FROM shipments WHERE orders.id = shipments.order_id"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in w
        assert "shipments" in r

    def test_merge_statement(self):
        sql = (
            "MERGE INTO target USING source ON (target.id = source.id) "
            "WHEN MATCHED THEN UPDATE SET target.v = source.v "
            "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (source.id, source.v)"
        )
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "target" in w
        assert "source" in r

    def test_nested_subqueries(self):
        sql = (
            "SELECT * FROM users WHERE id IN "
            "(SELECT user_id FROM orders WHERE product_id IN "
            "(SELECT id FROM products WHERE category = 'tech'))"
        )
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "users" in r and "orders" in r and "products" in r and w == set()

    def test_select_with_subquery_in_from(self):
        sql = "SELECT sub.total FROM (SELECT SUM(amount) AS total FROM payments) sub"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "payments" in r and w == set()

    def test_unparseable_sql_returns_none(self):
        result = _sqlglot_parse("THIS IS NOT SQL AT ALL !!!")
        assert result is None

    def test_cte_with_recursive(self):
        sql = (
            "WITH RECURSIVE tree AS ("
            "  SELECT id, parent_id FROM categories WHERE parent_id IS NULL "
            "  UNION ALL "
            "  SELECT c.id, c.parent_id FROM categories c JOIN tree t ON c.parent_id = t.id"
            ") SELECT * FROM tree"
        )
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "categories" in r and w == set()

    def test_insert_returning(self):
        sql = "INSERT INTO events (type) VALUES ('login') RETURNING id"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "events" in w

    def test_delete_with_subquery(self):
        sql = "DELETE FROM orders WHERE user_id IN (SELECT id FROM users WHERE banned = true)"
        r, w, *_ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in w
        assert "users" in r


# ---------------------------------------------------------------------------
# parse_sql_access — end-to-end routing and fallback
# ---------------------------------------------------------------------------


class TestParseSqlAccess:
    def test_simple_select(self):
        r, w, *_ = parse_sql_access("SELECT * FROM users")
        assert r == {"users"} and w == set()

    def test_simple_insert(self):
        r, w, *_ = parse_sql_access("INSERT INTO logs (msg) VALUES ('hi')")
        assert r == set() and w == {"logs"}

    def test_simple_update(self):
        r, w, *_ = parse_sql_access("UPDATE sessions SET active = false WHERE id = 7")
        assert "sessions" in w and "sessions" in r

    def test_simple_delete(self):
        r, w, *_ = parse_sql_access("DELETE FROM temp_tokens WHERE expires < NOW()")
        assert "temp_tokens" in w and "temp_tokens" in r

    def test_join_select(self):
        r, w, *_ = parse_sql_access("SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id")
        assert "users" in r and "orders" in r and w == set()

    def test_insert_select(self):
        r, w, *_ = parse_sql_access("INSERT INTO archive SELECT * FROM orders")
        assert "orders" in r and "archive" in w

    def test_cte_routed_to_sqlglot(self):
        pytest.importorskip("sqlglot")
        sql = "WITH t AS (SELECT * FROM products) SELECT * FROM t"
        r, w, *_ = parse_sql_access(sql)
        assert "products" in r and w == set()

    def test_union_routed_to_sqlglot(self):
        pytest.importorskip("sqlglot")
        # UNION falls through regex fast-path (returns None) and is handled by sqlglot.
        # These are now handled explicitly as reads.
        sql = "SELECT id FROM users UNION SELECT id FROM admins"
        r, w, *_ = parse_sql_access(sql)
        assert "users" in r and "admins" in r and w == set()

    def test_merge_routed_to_sqlglot(self):
        pytest.importorskip("sqlglot")
        sql = (
            "MERGE INTO inventory USING incoming ON inventory.sku = incoming.sku "
            "WHEN MATCHED THEN UPDATE SET inventory.qty = inventory.qty + incoming.qty "
            "WHEN NOT MATCHED THEN INSERT (sku, qty) VALUES (incoming.sku, incoming.qty)"
        )
        r, w, *_ = parse_sql_access(sql)
        assert "inventory" in w and "incoming" in r

    def test_unparseable_returns_empty_sets(self):
        r, w, *_ = parse_sql_access(";;; GARBAGE SQL ;;;")
        assert isinstance(r, set) and isinstance(w, set)

    def test_empty_sql_returns_empty_sets(self):
        r, w, *_ = parse_sql_access("")
        assert r == set() and w == set()

    def test_whitespace_only_returns_empty_sets(self):
        r, w, *_ = parse_sql_access("   \n   ")
        assert r == set() and w == set()

    def test_case_insensitive_select(self):
        r, w, *_ = parse_sql_access("select * from USERS")
        assert "USERS" in r and w == set()

    def test_case_insensitive_insert(self):
        r, w, *_ = parse_sql_access("insert into Orders (amount) values (50)")
        assert "Orders" in w

    def test_case_insensitive_update(self):
        r, w, *_ = parse_sql_access("UPDATE Accounts SET balance = 10 WHERE id = 1")
        assert "Accounts" in w

    def test_case_insensitive_delete(self):
        r, w, *_ = parse_sql_access("DELETE FROM Sessions WHERE id = 2")
        assert "Sessions" in w

    def test_trailing_semicolon_stripped(self):
        r, w, *_ = parse_sql_access("SELECT * FROM orders;")
        assert "orders" in r and w == set()

    def test_multiple_joins(self):
        sql = (
            "SELECT * FROM users u "
            "JOIN orders o ON u.id = o.user_id "
            "JOIN shipments s ON o.id = s.order_id "
            "JOIN products p ON o.product_id = p.id"
        )
        r, w, *_ = parse_sql_access(sql)
        assert {"users", "orders", "shipments", "products"} <= r and w == set()

    def test_schema_qualified_table(self):
        r, w, *_ = parse_sql_access("SELECT * FROM public.accounts WHERE id = 1")
        assert "accounts" in r and w == set()

    def test_quoted_table_name(self):
        r, w, *_ = parse_sql_access('SELECT * FROM "My Schema"')
        assert "My Schema" in r and w == set()

    def test_returns_sets_not_lists(self):
        r, w, *_ = parse_sql_access("SELECT * FROM users")
        assert isinstance(r, set) and isinstance(w, set)

    def test_complex_subquery_without_sqlglot(self):
        # Even if sqlglot not available, parse_sql_access should return sets
        sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
        r, w, *_ = parse_sql_access(sql)
        assert isinstance(r, set) and isinstance(w, set)


# ---------------------------------------------------------------------------
# Multi-statement SQL
# ---------------------------------------------------------------------------


class TestMultiStatementSql:
    @pytest.fixture(autouse=True)
    def _require_sqlglot(self):
        pytest.importorskip("sqlglot")

    def test_multi_update(self):
        sql = "UPDATE accounts SET balance = balance - 100 WHERE id = 1; UPDATE accounts SET balance = balance + 100 WHERE id = 2"
        r, w, lock, tx, *_ = parse_sql_access(sql)
        assert w == {"accounts"}
        assert "accounts" in r

    def test_multi_different_tables(self):
        sql = "UPDATE table_a SET x=1; UPDATE table_b SET x=2"
        r, w, lock, tx, *_ = parse_sql_access(sql)
        assert w == {"table_a", "table_b"}
        assert "table_a" in r and "table_b" in r

    def test_mixed_read_write(self):
        sql = "SELECT * FROM users; UPDATE accounts SET active=true"
        r, w, lock, tx, *_ = parse_sql_access(sql)
        assert "users" in r
        assert "accounts" in r
        assert "accounts" in w

    def test_lock_intent_merge(self):
        sql = "SELECT * FROM users FOR UPDATE; SELECT * FROM accounts FOR SHARE"
        r, w, lock, tx, *_ = parse_sql_access(sql)
        assert lock is LockIntent.UPDATE
        assert r == {"users", "accounts"}

    def test_lock_intent_merge_reverse(self):
        sql = "SELECT * FROM accounts FOR SHARE; SELECT * FROM users"
        r, w, lock, tx, *_ = parse_sql_access(sql)
        assert lock is LockIntent.SHARE
        assert r == {"users", "accounts"}


# ---------------------------------------------------------------------------
# Data-modifying CTEs (finding 1)
# ---------------------------------------------------------------------------


class TestDataModifyingCte:
    @pytest.fixture(autouse=True)
    def _require_sqlglot(self):
        pytest.importorskip("sqlglot")

    def test_update_inside_cte_is_a_write(self):
        sql = "WITH u AS (UPDATE jobs SET state='running' WHERE id=1 RETURNING id) SELECT * FROM u"
        r, w, *_ = parse_sql_access(sql)
        assert "jobs" in w, "data-modifying CTE UPDATE must be in the write set"
        assert "jobs" in r, "UPDATE reads its own table (WHERE clause)"
        # The phantom CTE alias must not appear as a table.
        assert "u" not in r and "u" not in w

    def test_delete_inside_cte_with_insert(self):
        sql = "WITH moved AS (DELETE FROM queue WHERE id=1 RETURNING *) INSERT INTO archive SELECT * FROM moved"
        r, w, *_ = parse_sql_access(sql)
        assert "queue" in w, "data-modifying CTE DELETE must be in the write set"
        assert "archive" in w
        assert "moved" not in r and "moved" not in w

    def test_insert_reads_cte_source(self):
        sql = "WITH src AS (SELECT * FROM b) INSERT INTO a SELECT * FROM src"
        r, w, *_ = parse_sql_access(sql)
        assert "a" in w
        assert "b" in r, "source table referenced via CTE must be read"
        assert "src" not in r and "src" not in w

    def test_plain_cte_alias_excluded_from_reads(self):
        sql = "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent"
        r, w, *_ = parse_sql_access(sql)
        assert "orders" in r
        assert "recent" not in r


# ---------------------------------------------------------------------------
# Multi-table UPDATE (finding 8)
# ---------------------------------------------------------------------------


class TestMultiTableUpdate:
    @pytest.fixture(autouse=True)
    def _require_sqlglot(self):
        pytest.importorskip("sqlglot")

    def test_mysql_multi_table_update_writes_all_set_tables(self):
        sql = "UPDATE t1 JOIN t2 ON t1.id = t2.id SET t1.a = 1, t2.b = 2"
        r, w, *_ = parse_sql_access(sql)
        assert "t1" in w and "t2" in w, "both SET targets must be writes"


# ---------------------------------------------------------------------------
# Multi-table DELETE (MySQL: DELETE t1, t2 FROM ...)
# ---------------------------------------------------------------------------


class TestMultiTableDelete:
    @pytest.fixture(autouse=True)
    def _require_sqlglot(self):
        pytest.importorskip("sqlglot")

    def test_mysql_multi_table_delete_writes_all_targets(self):
        # MySQL multi-table DELETE deletes rows from every table listed
        # before FROM, so both t1 and t2 are write/delete targets.
        result = parse_sql_access("DELETE t1, t2 FROM t1 JOIN t2 ON t1.id=t2.id")
        assert "t2" in result.write_tables, result.write_tables
        assert result.delete_tables is not None and "t2" in result.delete_tables, result.delete_tables


# ---------------------------------------------------------------------------
# LOCK TABLES (MySQL spelling, finding 10a)
# ---------------------------------------------------------------------------


class TestLockTablesMysql:
    def test_lock_tables_write(self):
        r, w, lock, *_ = parse_sql_access("LOCK TABLES t WRITE")
        assert "t" in w
        assert lock is LockIntent.UPDATE

    def test_lock_tables_read(self):
        r, w, lock, *_ = parse_sql_access("LOCK TABLES t READ")
        assert "t" in r
        assert "t" not in w
        assert lock is LockIntent.SHARE

    def test_tx_ops(self):
        sql = "BEGIN; UPDATE accounts SET x=1"
        r, w, lock, tx, *_ = parse_sql_access(sql)
        assert tx is TxOp.BEGIN
        assert "accounts" in w

    def test_multi_tx_ops(self):
        sql = "BEGIN; UPDATE accounts SET x=1; COMMIT"
        r, w, lock, tx, *_ = parse_sql_access(sql)
        assert tx is TxOp.COMMIT  # returns last tx_op
        assert "accounts" in w


class TestLockTablesMysqlMixed:
    def test_lock_tables_mixed_read_write(self):
        """LOCK TABLES t1 READ, t2 WRITE: t1 in read_tables, t2 in write_tables."""
        r, w, lock, *_ = parse_sql_access("LOCK TABLES t1 READ, t2 WRITE")
        assert "t1" in r, "READ-locked table should be in read_tables"
        assert "t1" not in w, "READ-locked table should NOT be in write_tables"
        assert "t2" in w, "WRITE-locked table should be in write_tables"
        assert lock is LockIntent.UPDATE

    def test_lock_tables_all_read(self):
        """LOCK TABLES t1 READ, t2 READ: both in read_tables, none in write_tables."""
        r, w, lock, *_ = parse_sql_access("LOCK TABLES t1 READ, t2 READ")
        assert "t1" in r
        assert "t2" in r
        assert not w, "READ-only LOCK TABLES should have empty write_tables"
        assert lock is LockIntent.SHARE

    def test_lock_tables_with_alias(self):
        """LOCK TABLES t1 AS a READ should extract table name 't1', not alias."""
        r, w, lock, *_ = parse_sql_access("LOCK TABLES t1 AS a READ")
        assert "t1" in r
        assert "a" not in r
        assert lock is LockIntent.SHARE


class TestDjangoPlaceholders:
    def test_delete_with_placeholders(self):
        # DELETE: formerly failed because of %s and IN (
        sql = 'DELETE FROM "t" WHERE "t"."id" IN (%s)'
        result = parse_sql_access(sql)
        assert "t" in result.read_tables
        assert "t" in result.write_tables

    def test_insert_with_placeholders(self):
        # INSERT: formerly failed because of %s and RETURNING
        sql = 'INSERT INTO "t" ("a", "b") VALUES (%s, %s) RETURNING "t"."id"'
        result = parse_sql_access(sql)
        assert result.read_tables == set()
        assert "t" in result.write_tables

    def test_named_placeholders(self):
        # Test %(name)s style
        sql = "SELECT * FROM t WHERE id = %(id)s"
        result = parse_sql_access(sql)
        assert "t" in result.read_tables

    def test_escaped_percent_not_treated_as_placeholder(self):
        """%%s in SQL (escaped percent) must not corrupt the SQL before parsing."""
        # Real-world case: LIKE '%%smith' contains %%s which must not become ?
        sql = "SELECT * FROM users WHERE name LIKE '%%smith' AND id = %s"
        result = parse_sql_access(sql)
        assert "users" in result.read_tables

    def test_escaped_percent_with_real_placeholder(self):
        """Mix of real %s placeholder and %%s escaped percent."""
        sql = "SELECT * FROM t WHERE x = %s AND y LIKE '%%something'"
        result = parse_sql_access(sql)
        assert "t" in result.read_tables

    def test_escaped_named_percent_not_treated_as_placeholder(self):
        """%%(name)s must not be converted to a placeholder."""
        sql = "SELECT * FROM t WHERE x = '%%(id)s'"
        result = parse_sql_access(sql)
        assert "t" in result.read_tables


class TestLockTableMultiTable:
    def test_lock_multiple_tables(self):
        """LOCK TABLE t1, t2 IN ... MODE should report both tables."""
        result = parse_sql_access("LOCK TABLE users, orders IN EXCLUSIVE MODE")
        assert result.write_tables == {"users", "orders"}

    def test_lock_multiple_tables_no_mode(self):
        """LOCK TABLE t1, t2 (no mode clause) should report both tables."""
        result = parse_sql_access("LOCK TABLE users, orders")
        assert result.write_tables == {"users", "orders"}

    def test_lock_multiple_tables_share_mode(self):
        """LOCK TABLE ... IN SHARE MODE should use SHARE intent for all tables."""
        result = parse_sql_access("LOCK TABLE users, orders IN SHARE MODE")
        assert result.write_tables == {"users", "orders"}
        assert result.lock_intent == LockIntent.SHARE


class TestLockTableQuotedSchema:
    def test_lock_table_quoted_schema_unquoted_table(self):
        """LOCK TABLE "schema".table should extract the table name, not truncate it.

        Bug: _strip_quotes incorrectly applies [1:-1] when the name starts
        with a quote but doesn't end with one, stripping the last char of
        the table name instead of a closing quote.
        """
        result = parse_sql_access('LOCK TABLE "public".users IN EXCLUSIVE MODE')
        assert "users" in result.write_tables, f"Expected 'users' in write_tables, got {result.write_tables}"

    def test_lock_table_quoted_schema_quoted_table(self):
        """LOCK TABLE "schema"."table" should work correctly (regression check)."""
        result = parse_sql_access('LOCK TABLE "public"."users" IN EXCLUSIVE MODE')
        assert "users" in result.write_tables, f"Expected 'users' in write_tables, got {result.write_tables}"


class TestInsertTableRegex:
    def test_schema_qualified_insert(self):
        """INSERT INTO schema.table should capture the table, not the schema."""
        from frontrun._sql_cursor import _RE_INSERT_TABLE

        m = _RE_INSERT_TABLE.match("INSERT INTO public.users (name) VALUES ('alice')")
        assert m is not None
        assert m.group(1) == "users"

    def test_insert_or_replace(self):
        """INSERT OR REPLACE INTO table should match."""
        from frontrun._sql_cursor import _RE_INSERT_TABLE

        m = _RE_INSERT_TABLE.match("INSERT OR REPLACE INTO users (id, name) VALUES (1, 'alice')")
        assert m is not None
        assert m.group(1) == "users"

    def test_insert_or_ignore(self):
        """INSERT OR IGNORE INTO table should match."""
        from frontrun._sql_cursor import _RE_INSERT_TABLE

        m = _RE_INSERT_TABLE.match("INSERT OR IGNORE INTO users (id, name) VALUES (1, 'alice')")
        assert m is not None
        assert m.group(1) == "users"

    def test_insert_ignore_mysql(self):
        """INSERT IGNORE INTO table should match (MySQL)."""
        from frontrun._sql_cursor import _RE_INSERT_TABLE

        m = _RE_INSERT_TABLE.match("INSERT IGNORE INTO users (name) VALUES ('alice')")
        assert m is not None
        assert m.group(1) == "users"

    def test_basic_insert_still_works(self):
        """Ensure basic INSERT INTO still works after regex change."""
        from frontrun._sql_cursor import _RE_INSERT_TABLE

        m = _RE_INSERT_TABLE.match("INSERT INTO users (name) VALUES ('alice')")
        assert m is not None
        assert m.group(1) == "users"


class TestUpdateTableRegex:
    def test_schema_qualified_update(self):
        """UPDATE schema.table should capture the table, not the schema."""
        from frontrun._sql_cursor import _RE_UPDATE_TABLE

        m = _RE_UPDATE_TABLE.match("UPDATE public.users SET name = 'bob' WHERE id = 1")
        assert m is not None
        assert m.group(1) == "users"

    def test_schema_qualified_update_quoted(self):
        """UPDATE `schema`.`table` should capture the table."""
        from frontrun._sql_cursor import _RE_UPDATE_TABLE

        m = _RE_UPDATE_TABLE.match("UPDATE `mydb`.`users` SET name = 'bob'")
        assert m is not None
        assert m.group(1) == "users"

    def test_basic_update_still_works(self):
        """Ensure basic UPDATE table still works after regex change."""
        from frontrun._sql_cursor import _RE_UPDATE_TABLE

        m = _RE_UPDATE_TABLE.match("UPDATE users SET name = 'bob' WHERE id = 1")
        assert m is not None
        assert m.group(1) == "users"


class TestPyformatEscapedPercentWithPlaceholder:
    def test_escaped_percent_followed_by_placeholder(self):
        """%%%s (literal percent + placeholder) in LIKE must parse correctly.

        In pyformat, '%%%s' means: literal %% (escaped percent) followed by
        %s (placeholder).  The preprocessing should produce '%?' inside the
        string literal so sqlglot sees a LIKE pattern, not a stray '%s'.
        """
        sql = "SELECT * FROM t WHERE name LIKE '%%%s'"
        result = parse_sql_access(sql)
        assert "t" in result.read_tables


# ---------------------------------------------------------------------------
# Bug: LOCK TABLES fallthrough when 0 tables extracted
# ---------------------------------------------------------------------------


class TestLockTablesFallthroughBug:
    """When LOCK TABLES fails to extract any tables (pathological input),
    it should still return an explicit result rather than falling through
    to subsequent handlers or sqlglot.

    The LOCK TABLES handler at lines 200-222 only returns when it
    extracts at least one table.  When 0 tables are extracted (all tokens
    are lock keywords), it falls through past its own guard, bypassing
    the explicit LOCK handling and delegating to sqlglot which loses
    the lock_intent information.
    """

    def test_lock_tables_no_tables_preserves_lock_intent(self):
        """LOCK TABLES with no extractable table names should still return
        with an explicit lock_intent rather than falling through to sqlglot.

        When LOCK TABLES extracts 0 tables, the handler should return an
        explicit empty SqlAccessResult.  Currently it falls through to
        sqlglot which returns lock_intent=None, losing the information
        that this was a LOCK statement.
        """
        # "WRITE" is a lock_word so name_tokens is empty -> tbl_name is None -> continue
        # Both read_tables and write_tables stay empty -> falls past the guard
        # -> falls through to sqlglot -> lock_intent becomes None (wrong)
        result = parse_sql_access("LOCK TABLES WRITE")
        assert result.read_tables == set(), f"Expected empty read_tables, got {result.read_tables}"
        assert result.write_tables == set(), f"Expected empty write_tables, got {result.write_tables}"
        # The key assertion: a recognized LOCK statement should preserve lock_intent
        # even when it extracts 0 tables.  The current code falls through to sqlglot
        # which returns lock_intent=None.
        assert result.lock_intent is not None, (
            "LOCK TABLES should return an explicit lock_intent even when 0 tables "
            "are extracted, not fall through to sqlglot (which loses lock_intent)"
        )
