"""SQL syntax coverage: locking clauses, dialect extensions and nested queries.

These tests assert table extraction, not schema introspection or wire-protocol behavior.
"""

from __future__ import annotations

import pytest

from frontrun._sql_parsing import LockIntent, parse_sql_access


class TestSelectForUpdate:
    @pytest.mark.parametrize(
        "sql, table, intent",
        [
            ("SELECT * FROM users WHERE id = 1 FOR UPDATE", "users", LockIntent.UPDATE),
            ("SELECT * FROM accounts WHERE user_id = ? FOR SHARE", "accounts", LockIntent.SHARE),
            ("SELECT * FROM orders WHERE id = ? FOR UPDATE NOWAIT", "orders", LockIntent.UPDATE),
            (
                "SELECT * FROM inventory WHERE product_id IN (1, 2, 3) FOR UPDATE SKIP LOCKED",
                "inventory",
                LockIntent.UPDATE_SKIP_LOCKED,
            ),
        ],
        ids=["update", "share", "nowait", "skip-locked"],
    )
    def test_lock_intent(self, sql, table, intent):
        access = parse_sql_access(sql)
        assert access.read_tables == {table}
        assert access.write_tables == set()
        assert access.lock_intent is intent


class TestLockTable:
    @pytest.mark.parametrize(
        "sql, table, intent",
        [
            ("LOCK TABLE users IN EXCLUSIVE MODE", "users", LockIntent.UPDATE),
            ("LOCK TABLE orders IN SHARE MODE", "orders", LockIntent.SHARE),
            ("LOCK TABLE inventory IN ROW EXCLUSIVE MODE", "inventory", LockIntent.UPDATE),
        ],
        ids=["exclusive", "share", "row-exclusive"],
    )
    def test_lock_intent(self, sql, table, intent):
        access = parse_sql_access(sql)
        assert access.read_tables == set()
        assert access.write_tables == {table}
        assert access.lock_intent is intent


class TestAdvisoryLocks:
    @pytest.mark.parametrize(
        "sql, resource, intent",
        [
            ("SELECT pg_advisory_lock(12345)", "advisory_lock:12345", LockIntent.UPDATE),
            ("SELECT pg_advisory_xact_lock(999)", "advisory_lock:999", LockIntent.UPDATE),
            ("SELECT pg_advisory_lock_shared(111)", "advisory_lock:111", LockIntent.SHARE),
            ("SELECT GET_LOCK('my_lock', 10)", "advisory_lock:my_lock", LockIntent.UPDATE),
        ],
        ids=["session", "transaction", "shared", "mysql"],
    )
    def test_lock_resource(self, sql, resource, intent):
        access = parse_sql_access(sql)
        assert resource in access.write_tables
        assert access.lock_intent is intent


class TestUnionOptimization:
    def test_union_select_should_be_reads_not_writes(self):
        sql = "SELECT id FROM users UNION SELECT id FROM archived_users"
        (r, w, *_) = parse_sql_access(sql)
        assert "users" in r and "archived_users" in r
        assert w == set(), "UNION reads should not be classified as writes"

    def test_intersect_should_be_reads(self):
        sql = "SELECT id FROM users INTERSECT SELECT id FROM admins"
        (r, w, *_) = parse_sql_access(sql)
        assert "users" in r and "admins" in r
        assert w == set()

    def test_except_should_be_reads(self):
        sql = "SELECT id FROM all_users EXCEPT SELECT id FROM banned_users"
        (r, w, *_) = parse_sql_access(sql)
        assert "all_users" in r and "banned_users" in r
        assert w == set()

    def test_union_all_should_be_reads(self):
        sql = "SELECT * FROM orders UNION ALL SELECT * FROM archived_orders"
        (r, w, *_) = parse_sql_access(sql)
        assert "orders" in r and "archived_orders" in r
        assert w == set()

    def test_insert_union_target_is_write(self):
        sql = "INSERT INTO summary SELECT * FROM users UNION SELECT * FROM archived_users"
        (r, w, *_) = parse_sql_access(sql)
        assert w == {"summary"}
        assert "users" in r and "archived_users" in r


class TestRelatedTableStatements:
    def test_insert_and_delete_extract_written_tables(self):
        insert_sql = "INSERT INTO orders (user_id, amount) VALUES (?, ?)"
        delete_sql = "DELETE FROM users WHERE id = ?"
        (r_insert, w_insert, *_) = parse_sql_access(insert_sql)
        (r_delete, w_delete, *_) = parse_sql_access(delete_sql)
        assert w_insert == {"orders"}
        assert w_delete == {"users"}


class TestTemporalTables:
    def test_for_system_time_as_of(self):
        sql = "SELECT * FROM users FOR SYSTEM_TIME AS OF '2024-01-01' WHERE id = 1"
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"users"}

    def test_for_system_time_between(self):
        sql = "SELECT * FROM accounts FOR SYSTEM_TIME BETWEEN '2024-01-01' AND '2024-01-31'"
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"accounts"}

    def test_system_versioned_table_insert(self):
        sql = "INSERT INTO audit_log (event, valid_from) VALUES (?, NOW())"
        (r, w, *_) = parse_sql_access(sql)
        assert w == {"audit_log"}


class TestExpressionColumns:
    def test_comparison_extracts_read_table(self):
        sql = "SELECT * FROM orders WHERE id = ? AND total > ?"
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"orders"}


class TestWindowFunctions:
    def test_window_function_partition_by(self):
        sql = """
        SELECT id, salary,
               RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rank
        FROM employees
        """
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"employees"}

    def test_window_function_frame(self):
        sql = """
        SELECT id, salary,
               AVG(salary) OVER (ORDER BY salary ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS rolling_avg
        FROM employees
        """
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"employees"}


class TestMultiDialect:
    def test_mysql_insert_or_replace(self):
        sql = "INSERT INTO users (id, name) VALUES (?, ?) ON DUPLICATE KEY UPDATE name = VALUES(name)"
        (r, w, *_) = parse_sql_access(sql)
        assert w == {"users"}

    def test_sqlite_insert_or_replace(self):
        sql = "INSERT OR REPLACE INTO accounts (id, balance) VALUES (?, ?)"
        (r, w, *_) = parse_sql_access(sql)
        assert w == {"accounts"}

    def test_postgres_on_conflict(self):
        sql = "INSERT INTO users (id, name) VALUES (?, ?) ON CONFLICT (id) DO UPDATE SET name = ?"
        (r, w, *_) = parse_sql_access(sql)
        assert w == {"users"}


class TestCorrelatedSubqueries:
    def test_correlated_subquery_in_where(self):
        sql = """
        SELECT * FROM users u
        WHERE balance > (SELECT AVG(balance) FROM accounts WHERE user_id = u.id)
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "users" in r and "accounts" in r

    def test_correlated_subquery_in_select_list(self):
        sql = """
        SELECT u.id, u.name,
               (SELECT COUNT(*) FROM orders WHERE user_id = u.id) AS order_count
        FROM users u
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "users" in r and "orders" in r


class TestCaseExpressions:
    def test_case_in_where_clause(self):
        sql = """
        SELECT * FROM orders
        WHERE CASE WHEN status = 'pending' THEN amount > 100
                   ELSE amount > 500 END
        """
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"orders"}

    def test_case_in_update_set(self):
        sql = """
        UPDATE accounts SET balance = balance +
          CASE WHEN type = 'premium' THEN 50 ELSE 10 END
        WHERE id = ?
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "accounts" in w and "accounts" in r


class TestExistsNotExists:
    def test_exists_subquery(self):
        sql = """
        SELECT * FROM users u
        WHERE EXISTS (SELECT 1 FROM orders WHERE user_id = u.id)
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "users" in r and "orders" in r

    def test_not_exists_subquery(self):
        sql = """
        DELETE FROM accounts
        WHERE NOT EXISTS (SELECT 1 FROM transactions WHERE account_id = accounts.id)
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "accounts" in w
        assert "transactions" in r


class TestDistinct:
    def test_distinct_on(self):
        sql = """
        SELECT DISTINCT ON (user_id) * FROM events
        ORDER BY user_id, created_at DESC
        """
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"events"}


class TestSelfJoins:
    def test_self_join_employees(self):
        sql = """
        SELECT a.id, a.name, b.id AS manager_id
        FROM employees a
        JOIN employees b ON a.manager_id = b.id
        """
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"employees"}

    def test_self_referential_update(self):
        sql = """
        UPDATE categories SET parent_id = ? WHERE id = ?
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "categories" in r and "categories" in w


class TestLimitOffset:
    def test_delete_with_limit(self):
        sql = """
        DELETE FROM sessions ORDER BY created_at LIMIT 10
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "sessions" in w and "sessions" in r

    def test_select_with_limit_offset(self):
        sql = """
        SELECT * FROM orders ORDER BY id LIMIT 20 OFFSET 100
        """
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"orders"}


class TestOuterJoinWhereSemantics:
    def test_left_join_with_where_on_outer_table(self):
        sql = """
        SELECT * FROM orders o
        LEFT JOIN users u ON o.user_id = u.id
        WHERE u.id IS NOT NULL
        """
        (r, w, *_) = parse_sql_access(sql)
        assert r == {"orders", "users"}


class TestLateralJoins:
    def test_lateral_join_with_correlation(self):
        sql = """
        SELECT * FROM users u,
        LATERAL (SELECT * FROM orders WHERE user_id = u.id LIMIT 1) o
        """
        (r, w, *_) = parse_sql_access(sql)
        assert "users" in r and "orders" in r


class TestUpsertEdgeCases:
    def test_insert_on_conflict_do_update_with_where(self):
        sql = """
        INSERT INTO users (id, name) VALUES (?, ?)
        ON CONFLICT (id) DO UPDATE SET name = ? WHERE is_active = true
        """
        (r, w, *_) = parse_sql_access(sql)
        assert w == {"users"}

    def test_insert_on_conflict_do_nothing(self):
        sql = """
        INSERT INTO unique_tokens (token, user_id) VALUES (?, ?)
        ON CONFLICT (token) DO NOTHING
        """
        (r, w, *_) = parse_sql_access(sql)
        assert w == {"unique_tokens"}


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            """
        DO $$
        BEGIN
            PERFORM pg_advisory_lock(1);
            UPDATE accounts SET balance = balance - 100 WHERE id = 1;
            PERFORM pg_advisory_unlock(1);
        END $$
        """,
            id="pg_advisory_lock_in_do_block",
        ),
        pytest.param("DELETE FROM users WHERE id = ?", id="fk_chain_dependencies"),
        pytest.param("DELETE FROM orders WHERE user_id = ?", id="fk_chain_dependencies"),
        pytest.param("DELETE FROM shipments WHERE order_id = ?", id="fk_chain_dependencies"),
        pytest.param("UPDATE employees SET manager_id = ? WHERE id = ?", id="self_referential_fk"),
        pytest.param("UPDATE orders SET total = 100 WHERE id = 1", id="generated_column_not_writable"),
        pytest.param(
            "SELECT * FROM users WHERE id = ? FOR UPDATE", id="prepared_statement_different_params_independent"
        ),
        pytest.param("CALL sp_update_user(?, ?)", id="stored_procedure_call_introspection"),
        pytest.param(
            """
        DO $$
        DECLARE
            v_table_name TEXT := 'orders';
        BEGIN
            EXECUTE 'DELETE FROM ' || v_table_name || ' WHERE status = ''cancelled''';
        END $$
        """,
            id="do_block_dynamic_sql",
        ),
    ],
)
def test_unmodeled_syntax_does_not_raise(sql):
    """Smoke coverage only; schema and dynamic SQL introspection are not asserted."""
    parse_sql_access(sql)
