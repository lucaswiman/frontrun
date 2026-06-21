"""Tests for COPY and PREPARE/EXECUTE statement parsing."""

from __future__ import annotations

from frontrun._sql_parsing import parse_sql_access


class TestCopyStatementParsing:
    """Tests for PostgreSQL COPY statement detection."""

    def test_copy_from_stdin(self):
        """COPY table FROM STDIN should be a write."""
        r = parse_sql_access("COPY users FROM STDIN")
        assert r.write_tables == {"users"}
        assert r.read_tables == set()

    def test_copy_from_stdin_with_format(self):
        """COPY table FROM STDIN WITH (FORMAT csv) should be a write."""
        r = parse_sql_access("COPY users FROM STDIN WITH (FORMAT csv)")
        assert r.write_tables == {"users"}
        assert r.read_tables == set()

    def test_copy_to_stdout(self):
        """COPY table TO STDOUT should be a read."""
        r = parse_sql_access("COPY accounts TO STDOUT")
        assert r.read_tables == {"accounts"}
        assert r.write_tables == set()

    def test_copy_to_file(self):
        """COPY table TO '/path/file.csv' should be a read."""
        r = parse_sql_access("COPY orders TO '/tmp/orders.csv'")
        assert r.read_tables == {"orders"}
        assert r.write_tables == set()

    def test_copy_from_file(self):
        """COPY table FROM '/path/file.csv' should be a write."""
        r = parse_sql_access("COPY inventory FROM '/tmp/inventory.csv'")
        assert r.write_tables == {"inventory"}
        assert r.read_tables == set()

    def test_copy_with_columns(self):
        """COPY table (col1, col2) FROM STDIN."""
        # Has parens after table name, but the table name itself isn't parenthesized
        r = parse_sql_access("COPY users (name, email) FROM STDIN")
        # This might fall through to sqlglot or be handled by regex
        assert "users" in r.write_tables

    def test_copy_case_insensitive(self):
        """COPY should be case-insensitive."""
        r = parse_sql_access("copy users from stdin")
        assert r.write_tables == {"users"}

    def test_copy_schema_qualified(self):
        """COPY with schema-qualified table."""
        r = parse_sql_access("COPY public.users FROM STDIN")
        assert r.write_tables == {"users"}

    def test_copy_no_lock_intent(self):
        """COPY should not set lock intent."""
        r = parse_sql_access("COPY users FROM STDIN")
        assert r.lock_intent is None

    def test_copy_no_tx_op(self):
        """COPY should not be a transaction control statement."""
        r = parse_sql_access("COPY users FROM STDIN")
        assert r.tx_op is None


class TestPrepareStatementParsing:
    """Tests for PostgreSQL PREPARE statement parsing."""

    def test_prepare_select(self):
        """PREPARE with SELECT should extract read tables from inner SQL."""
        r = parse_sql_access("PREPARE my_query AS SELECT * FROM users WHERE id = $1")
        assert r.read_tables == {"users"}
        assert r.write_tables == set()

    def test_prepare_insert(self):
        """PREPARE with INSERT should extract write tables."""
        r = parse_sql_access("PREPARE ins_user AS INSERT INTO users (name) VALUES ($1)")
        assert r.write_tables == {"users"}

    def test_prepare_update(self):
        """PREPARE with UPDATE should extract read+write tables."""
        r = parse_sql_access("PREPARE upd AS UPDATE accounts SET balance = $1 WHERE id = $2")
        assert r.write_tables == {"accounts"}
        assert r.read_tables == {"accounts"}

    def test_prepare_delete(self):
        """PREPARE with DELETE should extract read+write tables."""
        r = parse_sql_access("PREPARE del AS DELETE FROM sessions WHERE id = $1")
        assert r.write_tables == {"sessions"}
        assert r.read_tables == {"sessions"}

    def test_prepare_no_as(self):
        """PREPARE without AS clause (malformed) should return empty."""
        r = parse_sql_access("PREPARE my_stmt")
        assert r.read_tables == set()
        assert r.write_tables == set()


class TestExecuteStatementParsing:
    """Tests for EXECUTE statement parsing (opaque without prepared stmt registry)."""

    def test_execute_returns_empty(self):
        """EXECUTE stmt_name returns empty sets (opaque)."""
        r = parse_sql_access("EXECUTE my_query(1)")
        assert r.read_tables == set()
        assert r.write_tables == set()

    def test_execute_no_params(self):
        """EXECUTE stmt_name without params."""
        r = parse_sql_access("EXECUTE my_query")
        assert r.read_tables == set()
        assert r.write_tables == set()


class TestDeallocateStatementParsing:
    """Tests for DEALLOCATE statement parsing."""

    def test_deallocate(self):
        """DEALLOCATE stmt_name should be a no-op (no table access)."""
        r = parse_sql_access("DEALLOCATE my_query")
        assert r.read_tables == set()
        assert r.write_tables == set()

    def test_deallocate_prepare(self):
        """DEALLOCATE PREPARE stmt_name."""
        r = parse_sql_access("DEALLOCATE PREPARE my_query")
        assert r.read_tables == set()
        assert r.write_tables == set()

    def test_deallocate_all(self):
        """DEALLOCATE ALL."""
        r = parse_sql_access("DEALLOCATE ALL")
        assert r.read_tables == set()
        assert r.write_tables == set()
