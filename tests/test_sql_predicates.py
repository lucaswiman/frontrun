"""Tests for SQL row-level predicate extraction (Algorithm 5).

Tests cover:
- EqualityPredicate dataclass
- extract_equality_predicates() — WHERE clause parsing
- can_use_row_level() — decision function
- pk_predicates_disjoint() — row disjointness check
- End-to-end: parameter resolution → predicate extraction
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

from frontrun._sql_predicates import (
    EqualityPredicate,
    InListPredicate,
    can_use_row_level,
    expand_predicate_rows,
    extract_equality_predicates,
    extract_row_level_access,
    pk_predicates_disjoint,
)

# ---------------------------------------------------------------------------
# extract_row_level_access
# ---------------------------------------------------------------------------


class TestExtractRowLevelAccess:
    def test_insert_single_row(self):
        sql = "INSERT INTO users (id, name) VALUES (1, 'alice')"
        rows = extract_row_level_access(sql)
        assert rows == [[EqualityPredicate("id", "1"), EqualityPredicate("name", "alice")]]

    def test_insert_multiple_rows(self):
        sql = "INSERT INTO users (id, name) VALUES (1, 'alice'), (2, 'bob')"
        rows = extract_row_level_access(sql)
        assert len(rows) == 2
        assert rows[0] == [EqualityPredicate("id", "1"), EqualityPredicate("name", "alice")]
        assert rows[1] == [EqualityPredicate("id", "2"), EqualityPredicate("name", "bob")]

    def test_insert_with_expressions_skips_columns(self):
        """Columns with non-literal values (like NOW()) are skipped in row predicates."""
        sql = "INSERT INTO users (id, created_at) VALUES (1, NOW())"
        rows = extract_row_level_access(sql)
        assert rows == [[EqualityPredicate("id", "1")]]

    def test_insert_without_columns_returns_none(self):
        """INSERT without explicit column list cannot be mapped to predicates."""
        sql = "INSERT INTO users VALUES (1, 'alice')"
        assert extract_row_level_access(sql) is None

    def test_where_clause_equality(self):
        sql = "SELECT * FROM users WHERE id = 1"
        rows = extract_row_level_access(sql)
        assert rows == [[EqualityPredicate("id", "1")]]

    def test_where_clause_in_list(self):
        sql = "SELECT * FROM users WHERE id IN (1, 2)"
        rows = extract_row_level_access(sql)
        assert rows == [[EqualityPredicate("id", "1")], [EqualityPredicate("id", "2")]]


# ---------------------------------------------------------------------------
# extract_equality_predicates
# ---------------------------------------------------------------------------


class TestExtractEqualityPredicates:
    def test_simple_equality(self):
        preds = extract_equality_predicates("SELECT * FROM users WHERE id = 42")
        assert preds == [EqualityPredicate("id", "42")]

    def test_string_literal(self):
        preds = extract_equality_predicates("SELECT * FROM users WHERE name = 'alice'")
        assert preds == [EqualityPredicate("name", "alice")]

    def test_compound_and(self):
        preds = extract_equality_predicates("UPDATE t SET x = 1 WHERE id = 1 AND region = 'us'")
        assert EqualityPredicate("id", "1") in preds
        assert EqualityPredicate("region", "us") in preds
        assert len(preds) == 2

    def test_triple_and(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE a = 1 AND b = 2 AND c = 3")
        assert len(preds) == 3
        assert EqualityPredicate("a", "1") in preds
        assert EqualityPredicate("b", "2") in preds
        assert EqualityPredicate("c", "3") in preds

    def test_or_returns_empty(self):
        """OR predicates are not conjunctive — return empty."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = 1 OR id = 2")
        assert preds == []

    def test_no_where_clause(self):
        preds = extract_equality_predicates("SELECT * FROM users")
        assert preds == []

    def test_insert_no_where(self):
        preds = extract_equality_predicates("INSERT INTO t (id) VALUES (1)")
        assert preds == []

    def test_inequality_skipped(self):
        """Non-equality predicates (>, <, !=) are skipped."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE id > 5")
        assert preds == []

    def test_in_clause_extracted(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id IN (1, 2, 3)")
        assert len(preds) == 1
        assert isinstance(preds[0], InListPredicate)
        assert preds[0].column == "id"
        assert preds[0].values == frozenset({"1", "2", "3"})

    def test_in_clause_single_value(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id IN (42)")
        assert len(preds) == 1
        assert isinstance(preds[0], InListPredicate)
        assert preds[0].values == frozenset({"42"})

    def test_in_clause_with_strings(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE status IN ('active', 'pending')")
        assert len(preds) == 1
        assert isinstance(preds[0], InListPredicate)
        assert preds[0].column == "status"
        assert preds[0].values == frozenset({"active", "pending"})

    def test_in_clause_with_subquery_skipped(self):
        """IN with subquery is not a literal list — should be skipped."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE id IN (SELECT id FROM other)")
        assert preds == []

    def test_in_clause_mixed_with_equality(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE region = 'us' AND id IN (1, 2)")
        assert len(preds) == 2
        assert EqualityPredicate("region", "us") in preds
        in_pred = [p for p in preds if isinstance(p, InListPredicate)][0]
        assert in_pred.column == "id"
        assert in_pred.values == frozenset({"1", "2"})

    def test_between_skipped(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id BETWEEN 1 AND 10")
        assert preds == []

    def test_like_skipped(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE name LIKE '%alice%'")
        assert preds == []

    def test_subquery_skipped(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = (SELECT MAX(id) FROM t)")
        # The subquery's = is not column = literal, so empty
        assert preds == []

    def test_mixed_equality_and_range(self):
        """Only equality predicates extracted, range predicates skipped."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = 42 AND age > 18")
        assert preds == [EqualityPredicate("id", "42")]

    def test_update_where(self):
        preds = extract_equality_predicates("UPDATE accounts SET balance = 0 WHERE id = 7")
        assert EqualityPredicate("id", "7") in preds

    def test_delete_where(self):
        preds = extract_equality_predicates("DELETE FROM sessions WHERE session_id = 'abc123'")
        assert EqualityPredicate("session_id", "abc123") in preds

    def test_reversed_operands(self):
        """Handles literal = column (reversed order)."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE 42 = id")
        assert preds == [EqualityPredicate("id", "42")]

    def test_parse_error_returns_empty(self):
        preds = extract_equality_predicates("NOT VALID SQL {{{")
        assert preds == []

    def test_function_call_in_predicate_skipped(self):
        """Function calls like NOW() are not literals."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE created_at = NOW()")
        assert preds == []

    def test_null_comparison(self):
        """IS NULL is not an equality — should be empty."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE deleted_at IS NULL")
        assert preds == []

    def test_qualified_column(self):
        """Table-qualified column (t.id) should extract just the column name."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE t.id = 42")
        assert preds == [EqualityPredicate("id", "42")]

    def test_subquery_where_does_not_leak_to_outer(self):
        """A WHERE inside a subquery must not define the outer statement's row identity.

        ``UPDATE counters SET value = (SELECT value FROM counters WHERE id = 1)``
        is a full-table write — every row of counters is updated. The subquery's
        ``id = 1`` must NOT be reported as the UPDATE's predicate (finding 2),
        otherwise a full-table write is mistaken for a single-row write.
        """
        sql = "UPDATE counters SET value = (SELECT value FROM counters WHERE id = 1)"
        preds = extract_equality_predicates(sql)
        assert preds == [], "subquery WHERE must not leak to the outer statement"

    def test_subquery_where_leak_select(self):
        """SELECT with no top-level WHERE but a subquery WHERE → no predicates."""
        sql = "SELECT * FROM t WHERE x IN (SELECT y FROM u WHERE id = 5)"
        preds = extract_equality_predicates(sql)
        # IN (subquery) is not a literal list, and the subquery's id=5 must not leak.
        assert EqualityPredicate("id", "5") not in preds


# ---------------------------------------------------------------------------
# expand_predicate_rows
# ---------------------------------------------------------------------------


class TestExpandPredicateRows:
    def test_equalities_only(self):
        preds = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        rows = expand_predicate_rows(preds)
        assert rows == [preds]

    def test_empty_predicates(self):
        assert expand_predicate_rows([]) is None

    def test_single_in_list(self):
        preds = [InListPredicate("id", frozenset({"1", "2", "3"}))]
        rows = expand_predicate_rows(preds)
        assert rows is not None
        assert len(rows) == 3
        values = {rows[i][0].value for i in range(3)}
        assert values == {"1", "2", "3"}

    def test_in_list_with_equality(self):
        preds = [EqualityPredicate("region", "us"), InListPredicate("id", frozenset({"1", "2"}))]
        rows = expand_predicate_rows(preds)
        assert rows is not None
        assert len(rows) == 2
        for row in rows:
            assert EqualityPredicate("region", "us") in row
            assert len(row) == 2

    def test_two_in_lists_cross_product(self):
        preds = [
            InListPredicate("id", frozenset({"1", "2"})),
            InListPredicate("region", frozenset({"us", "eu"})),
        ]
        rows = expand_predicate_rows(preds)
        assert rows is not None
        assert len(rows) == 4  # 2 x 2

    def test_large_in_list_returns_none(self):
        """IN-list cross product exceeding _MAX_EXPANSION returns None (table-level fallback)."""
        preds = [
            InListPredicate("id", frozenset(str(i) for i in range(10))),
            InListPredicate("region", frozenset(str(i) for i in range(10))),
        ]
        # 10 x 10 = 100 > _MAX_EXPANSION (64)
        assert expand_predicate_rows(preds) is None


# ---------------------------------------------------------------------------
# can_use_row_level
# ---------------------------------------------------------------------------


class TestCanUseRowLevel:
    def test_both_have_pk_predicates(self):
        preds_a = [EqualityPredicate("id", "1")]
        preds_b = [EqualityPredicate("id", "2")]
        assert can_use_row_level("users", preds_a, preds_b, {"id"})

    def test_empty_predicates_a(self):
        preds_b = [EqualityPredicate("id", "1")]
        assert not can_use_row_level("users", [], preds_b, {"id"})

    def test_empty_predicates_b(self):
        preds_a = [EqualityPredicate("id", "1")]
        assert not can_use_row_level("users", preds_a, [], {"id"})

    def test_unknown_pk(self):
        preds_a = [EqualityPredicate("id", "1")]
        preds_b = [EqualityPredicate("id", "2")]
        assert not can_use_row_level("users", preds_a, preds_b, None)

    def test_missing_pk_column_a(self):
        """Predicates don't cover all PK columns."""
        preds_a = [EqualityPredicate("name", "alice")]
        preds_b = [EqualityPredicate("id", "2")]
        assert not can_use_row_level("users", preds_a, preds_b, {"id"})

    def test_composite_pk_both_complete(self):
        preds_a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        preds_b = [EqualityPredicate("id", "2"), EqualityPredicate("region", "eu")]
        assert can_use_row_level("users", preds_a, preds_b, {"id", "region"})

    def test_composite_pk_one_incomplete(self):
        preds_a = [EqualityPredicate("id", "1")]
        preds_b = [EqualityPredicate("id", "2"), EqualityPredicate("region", "eu")]
        assert not can_use_row_level("users", preds_a, preds_b, {"id", "region"})

    def test_extra_predicates_ok(self):
        """Extra non-PK predicates don't prevent row-level."""
        preds_a = [EqualityPredicate("id", "1"), EqualityPredicate("name", "alice")]
        preds_b = [EqualityPredicate("id", "2"), EqualityPredicate("name", "bob")]
        assert can_use_row_level("users", preds_a, preds_b, {"id"})


# ---------------------------------------------------------------------------
# pk_predicates_disjoint
# ---------------------------------------------------------------------------


class TestPkPredicatesDisjoint:
    def test_different_pk_values(self):
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("id", "2")]
        assert pk_predicates_disjoint(a, b)

    def test_same_pk_values(self):
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("id", "1")]
        assert not pk_predicates_disjoint(a, b)

    def test_composite_pk_one_differs(self):
        a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        b = [EqualityPredicate("id", "1"), EqualityPredicate("region", "eu")]
        assert pk_predicates_disjoint(a, b)

    def test_composite_pk_all_same(self):
        a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        b = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        assert not pk_predicates_disjoint(a, b)

    def test_no_shared_columns(self):
        """Different column sets → can't prove disjoint."""
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("name", "alice")]
        assert not pk_predicates_disjoint(a, b)

    def test_subset_columns_differ(self):
        a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        b = [EqualityPredicate("id", "2")]
        assert pk_predicates_disjoint(a, b)

    def test_empty_a(self):
        a: list[EqualityPredicate] = []
        b = [EqualityPredicate("id", "1")]
        assert not pk_predicates_disjoint(a, b)

    def test_empty_b(self):
        a = [EqualityPredicate("id", "1")]
        b: list[EqualityPredicate] = []
        assert not pk_predicates_disjoint(a, b)

    def test_both_empty(self):
        assert not pk_predicates_disjoint([], [])

    def test_in_list_disjoint(self):
        a = [InListPredicate("id", frozenset({"1", "2", "3"}))]
        b = [InListPredicate("id", frozenset({"4", "5", "6"}))]
        assert pk_predicates_disjoint(a, b)

    def test_in_list_overlapping(self):
        a = [InListPredicate("id", frozenset({"1", "2", "3"}))]
        b = [InListPredicate("id", frozenset({"3", "4", "5"}))]
        assert not pk_predicates_disjoint(a, b)

    def test_in_list_vs_equality_disjoint(self):
        a = [InListPredicate("id", frozenset({"1", "2", "3"}))]
        b = [EqualityPredicate("id", "4")]
        assert pk_predicates_disjoint(a, b)

    def test_in_list_vs_equality_overlapping(self):
        a = [InListPredicate("id", frozenset({"1", "2", "3"}))]
        b = [EqualityPredicate("id", "2")]
        assert not pk_predicates_disjoint(a, b)

    def test_in_list_composite_pk(self):
        """Different id sets but same region — disjoint on id."""
        a = [InListPredicate("id", frozenset({"1", "2"})), EqualityPredicate("region", "us")]
        b = [InListPredicate("id", frozenset({"3", "4"})), EqualityPredicate("region", "us")]
        assert pk_predicates_disjoint(a, b)


# ---------------------------------------------------------------------------
# End-to-end: parameter resolution → predicate extraction
# ---------------------------------------------------------------------------


class TestParameterizedPredicateExtraction:
    """End-to-end: resolve params → extract predicates."""

    def test_qmark_select(self):
        from frontrun._sql_params import resolve_parameters

        sql = "SELECT * FROM users WHERE id = ?"
        resolved = resolve_parameters(sql, (42,), "qmark")
        preds = extract_equality_predicates(resolved)
        assert preds == [EqualityPredicate("id", "42")]

    def test_format_update(self):
        from frontrun._sql_params import resolve_parameters

        sql = "UPDATE accounts SET balance = balance - %s WHERE id = %s"
        resolved = resolve_parameters(sql, (100, 7), "format")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "7") in preds

    def test_named_compound(self):
        from frontrun._sql_params import resolve_parameters

        sql = "SELECT * FROM t WHERE id = :id AND region = :region"
        resolved = resolve_parameters(sql, {"id": 1, "region": "us"}, "named")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "1") in preds
        assert EqualityPredicate("region", "us") in preds

    def test_pyformat_orm_style(self):
        from frontrun._sql_params import resolve_parameters

        sql = "SELECT users.id FROM users WHERE users.id = %s"
        resolved = resolve_parameters(sql, (42,), "pyformat")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "42") in preds
