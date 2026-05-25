from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from frontrun._sql_params import _python_to_sql_literal, resolve_parameters


class TestPythonToSqlLiteral:
    def test_none(self):
        assert _python_to_sql_literal(None) == "NULL"

    def test_true(self):
        assert _python_to_sql_literal(True) == "TRUE"

    def test_false(self):
        assert _python_to_sql_literal(False) == "FALSE"

    def test_bool_before_int(self):
        # bool is a subclass of int; must be handled before int check
        assert _python_to_sql_literal(True) == "TRUE"
        assert _python_to_sql_literal(False) == "FALSE"
        assert _python_to_sql_literal(True) != "1"
        assert _python_to_sql_literal(False) != "0"

    def test_int_zero(self):
        assert _python_to_sql_literal(0) == "0"

    def test_int_positive(self):
        assert _python_to_sql_literal(42) == "42"

    def test_int_negative(self):
        assert _python_to_sql_literal(-17) == "-17"

    def test_int_large(self):
        assert _python_to_sql_literal(10**18) == "1000000000000000000"

    def test_float_positive(self):
        assert _python_to_sql_literal(3.14) == "3.14"

    def test_float_negative(self):
        assert _python_to_sql_literal(-2.5) == "-2.5"

    def test_float_zero(self):
        assert _python_to_sql_literal(0.0) == "0.0"

    def test_float_inf(self):
        assert _python_to_sql_literal(float("inf")) == "inf"

    def test_float_neg_inf(self):
        assert _python_to_sql_literal(float("-inf")) == "-inf"

    def test_float_nan(self):
        result = _python_to_sql_literal(float("nan"))
        assert math.isnan(float(result))

    def test_bytes_empty(self):
        assert _python_to_sql_literal(b"") == "X''"

    def test_bytes_simple(self):
        assert _python_to_sql_literal(b"\xde\xad\xbe\xef") == "X'deadbeef'"

    def test_bytes_long(self):
        data = bytes(range(256))
        result = _python_to_sql_literal(data)
        assert result.startswith("X'")
        assert result.endswith("'")
        assert len(result) == 2 + 512 + 1  # X' + 512 hex chars + '

    def test_bytearray(self):
        assert _python_to_sql_literal(bytearray(b"\x01\x02\x03")) == "X'010203'"

    def test_memoryview(self):
        assert _python_to_sql_literal(memoryview(b"\xca\xfe")) == "X'cafe'"

    def test_string_simple(self):
        assert _python_to_sql_literal("hello") == "'hello'"

    def test_string_empty(self):
        assert _python_to_sql_literal("") == "''"

    def test_string_single_quote(self):
        assert _python_to_sql_literal("O'Brien") == "'O''Brien'"

    def test_string_multiple_quotes(self):
        assert _python_to_sql_literal("it's a 'test'") == "'it''s a ''test'''"

    def test_string_only_quotes(self):
        # "'''" has 3 single quotes; each becomes '' → 6 chars, wrapped = 8 chars total
        assert _python_to_sql_literal("'''") == "''''''''"

    def test_string_nested_double_quotes(self):
        assert _python_to_sql_literal('say "hello"') == "'say \"hello\"'"

    def test_string_unicode(self):
        result = _python_to_sql_literal("caf\u00e9")
        assert result == "'caf\u00e9'"

    def test_string_emoji(self):
        result = _python_to_sql_literal("\U0001f600")
        assert result == "'\U0001f600'"

    def test_datetime_uses_str(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _python_to_sql_literal(dt)
        assert "2024-01-15" in result
        assert result.startswith("'")
        assert result.endswith("'")

    def test_date_uses_str(self):
        d = date(2024, 6, 1)
        result = _python_to_sql_literal(d)
        assert "2024-06-01" in result
        assert result.startswith("'")
        assert result.endswith("'")

    def test_decimal(self):
        result = _python_to_sql_literal(Decimal("123.456"))
        assert result == "'123.456'"

    def test_uuid(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = _python_to_sql_literal(uid)
        assert "12345678-1234-5678-1234-567812345678" in result
        assert result.startswith("'")


class TestResolveParameters:
    # -----------------------------------------------------------------
    # qmark paramstyle
    # -----------------------------------------------------------------

    def test_qmark_basic(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = ? AND name = ?",
            (42, "alice"),
            "qmark",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_qmark_single_param(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE id = ?",
            (1,),
            "qmark",
        )
        assert resolved == "SELECT * FROM t WHERE id = 1"

    def test_qmark_more_placeholders_than_params(self):
        sql = "SELECT ? AND ? AND ?"
        resolved = resolve_parameters(sql, (1, 2), "qmark")
        assert resolved == "SELECT 1 AND 2 AND ?"

    def test_qmark_fewer_placeholders_than_params(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE id = ?",
            (10, 20, 30),
            "qmark",
        )
        assert resolved == "SELECT * FROM t WHERE id = 10"

    def test_qmark_list_params(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE a = ? AND b = ?",
            [1, 2],
            "qmark",
        )
        assert resolved == "SELECT * FROM t WHERE a = 1 AND b = 2"

    def test_qmark_none(self):
        resolved = resolve_parameters(
            "UPDATE t SET x = ? WHERE id = ?",
            (None, 1),
            "qmark",
        )
        assert "NULL" in resolved
        assert "1" in resolved

    def test_qmark_bool_true(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE active = ?",
            (True,),
            "qmark",
        )
        assert "TRUE" in resolved

    def test_qmark_bool_false(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE active = ?",
            (False,),
            "qmark",
        )
        assert "FALSE" in resolved

    def test_qmark_string_with_quotes(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE name = ?",
            ("O'Brien",),
            "qmark",
        )
        assert "O''Brien" in resolved

    def test_qmark_float(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE score = ?",
            (3.14,),
            "qmark",
        )
        assert "3.14" in resolved

    def test_qmark_bytes(self):
        resolved = resolve_parameters(
            "INSERT INTO t (data) VALUES (?)",
            (b"\xde\xad",),
            "qmark",
        )
        assert "X'dead'" in resolved

    def test_qmark_mixed_types(self):
        resolved = resolve_parameters(
            "INSERT INTO t VALUES (?, ?, ?, ?)",
            (1, "hello", None, True),
            "qmark",
        )
        assert "1" in resolved
        assert "'hello'" in resolved
        assert "NULL" in resolved
        assert "TRUE" in resolved

    def test_qmark_no_placeholders(self):
        sql = "SELECT 1 FROM dual"
        resolved = resolve_parameters(sql, (42,), "qmark")
        assert resolved == sql

    def test_qmark_empty_sql(self):
        resolved = resolve_parameters("", (1,), "qmark")
        assert resolved == ""

    def test_qmark_many_params(self):
        n = 50
        placeholders = ", ".join("?" * n)
        sql = f"INSERT INTO t VALUES ({placeholders})"
        params = list(range(n))
        resolved = resolve_parameters(sql, params, "qmark")
        for i in range(n):
            assert str(i) in resolved

    # -----------------------------------------------------------------
    # numeric paramstyle
    # -----------------------------------------------------------------

    def test_numeric_basic(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = :1 AND name = :2",
            (42, "alice"),
            "numeric",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_numeric_out_of_order(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE b = :2 AND a = :1",
            ("first", "second"),
            "numeric",
        )
        assert resolved == "SELECT * FROM t WHERE b = 'second' AND a = 'first'"

    def test_numeric_repeated_index(self):
        resolved = resolve_parameters(
            "SELECT :1, :1, :2",
            ("x", "y"),
            "numeric",
        )
        assert resolved == "SELECT 'x', 'x', 'y'"

    def test_numeric_single(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE id = :1",
            (99,),
            "numeric",
        )
        assert resolved == "SELECT * FROM t WHERE id = 99"

    def test_numeric_list_params(self):
        resolved = resolve_parameters(
            "SELECT :1, :2",
            [10, 20],
            "numeric",
        )
        assert resolved == "SELECT 10, 20"

    def test_numeric_none(self):
        resolved = resolve_parameters(
            "UPDATE t SET val = :1",
            (None,),
            "numeric",
        )
        assert "NULL" in resolved

    def test_numeric_out_of_bounds_returns_sql(self):
        sql = "SELECT * FROM t WHERE id = :5"
        resolved = resolve_parameters(sql, (1, 2), "numeric")
        assert resolved == sql

    def test_dollar_zero_not_resolved_to_last_param(self):
        resolved = resolve_parameters("SELECT $0, $1", ("first", "second"), "dollar")
        assert "'second'" not in resolved or "$0" in resolved

    # -----------------------------------------------------------------
    # named paramstyle
    # -----------------------------------------------------------------

    def test_named_basic(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = :id AND name = :name",
            {"id": 42, "name": "alice"},
            "named",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_named_underscore(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE user_id = :user_id",
            {"user_id": 7},
            "named",
        )
        assert resolved == "SELECT * FROM t WHERE user_id = 7"

    def test_named_repeated(self):
        resolved = resolve_parameters(
            "SELECT :id, :id",
            {"id": 5},
            "named",
        )
        assert resolved == "SELECT 5, 5"

    def test_named_pg_cast_not_matched(self):
        resolved = resolve_parameters(
            "SELECT id::text FROM users WHERE id = :id",
            {"id": 42},
            "named",
        )
        assert "::text" in resolved
        assert "42" in resolved

    def test_named_pg_cast_only(self):
        sql = "SELECT id::text FROM users"
        resolved = resolve_parameters(sql, {"id": 42}, "named")
        assert "::text" in resolved

    def test_named_missing_key_returns_sql(self):
        sql = "SELECT * FROM users WHERE id = :missing"
        resolved = resolve_parameters(sql, {"other": 1}, "named")
        assert resolved == sql

    def test_named_none_value(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE x = :x",
            {"x": None},
            "named",
        )
        assert "NULL" in resolved

    def test_named_bool_value(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE flag = :flag",
            {"flag": True},
            "named",
        )
        assert "TRUE" in resolved

    def test_named_multiple_names(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE a = :alpha AND b = :beta AND c = :gamma",
            {"alpha": 1, "beta": "two", "gamma": None},
            "named",
        )
        assert "1" in resolved
        assert "'two'" in resolved
        assert "NULL" in resolved

    def test_named_numeric_adjacent(self):
        # :name should not accidentally match :123 style
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE name = :name",
            {"name": "test"},
            "named",
        )
        assert "'test'" in resolved

    # -----------------------------------------------------------------
    # format paramstyle
    # -----------------------------------------------------------------

    def test_format_basic(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = %s AND name = %s",
            (42, "alice"),
            "format",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_format_escaped_percent_not_matched(self):
        resolved = resolve_parameters(
            "SELECT '%%s' FROM users WHERE id = %s",
            (42,),
            "format",
        )
        assert "%%s" in resolved
        assert "42" in resolved

    def test_format_no_placeholders(self):
        sql = "SELECT 1 FROM t"
        resolved = resolve_parameters(sql, (42,), "format")
        assert resolved == sql

    def test_format_list_params(self):
        resolved = resolve_parameters(
            "SELECT %s, %s",
            [10, 20],
            "format",
        )
        assert resolved == "SELECT 10, 20"

    def test_format_none(self):
        resolved = resolve_parameters(
            "INSERT INTO t VALUES (%s)",
            (None,),
            "format",
        )
        assert "NULL" in resolved

    def test_format_with_dict_uses_pyformat(self):
        # When format paramstyle is given with a dict, it uses %(name)s style
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE id = %(id)s",
            {"id": 99},
            "format",
        )
        assert "99" in resolved

    # -----------------------------------------------------------------
    # pyformat paramstyle
    # -----------------------------------------------------------------

    def test_pyformat_with_dict(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = %(id)s AND name = %(name)s",
            {"id": 42, "name": "alice"},
            "pyformat",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_pyformat_with_tuple_falls_back_to_format(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = %s",
            (42,),
            "pyformat",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42"

    def test_pyformat_escaped_percent_not_matched(self):
        resolved = resolve_parameters(
            "SELECT '%%' FROM t WHERE id = %(id)s",
            {"id": 10},
            "pyformat",
        )
        assert "'%%'" in resolved
        assert "10" in resolved

    def test_pyformat_missing_key_returns_sql(self):
        sql = "SELECT * FROM t WHERE id = %(missing)s"
        resolved = resolve_parameters(sql, {"other": 1}, "pyformat")
        assert resolved == sql

    def test_pyformat_none_value(self):
        resolved = resolve_parameters(
            "UPDATE t SET x = %(x)s",
            {"x": None},
            "pyformat",
        )
        assert "NULL" in resolved

    def test_pyformat_repeated_key(self):
        resolved = resolve_parameters(
            "SELECT %(val)s, %(val)s",
            {"val": 7},
            "pyformat",
        )
        assert resolved == "SELECT 7, 7"

    # -----------------------------------------------------------------
    # None parameters
    # -----------------------------------------------------------------

    def test_none_params_returns_same_object(self):
        sql = "SELECT * FROM users WHERE id = 1"
        result = resolve_parameters(sql, None, "qmark")
        assert result is sql

    # -----------------------------------------------------------------
    # Unknown paramstyle
    # -----------------------------------------------------------------

    def test_unknown_paramstyle_returns_sql(self):
        sql = "SELECT * FROM t WHERE id = ?"
        resolved = resolve_parameters(sql, (1,), "unknown_style")
        assert resolved == sql

    def test_empty_paramstyle_returns_sql(self):
        sql = "SELECT * FROM t WHERE id = ?"
        resolved = resolve_parameters(sql, (1,), "")
        assert resolved == sql

    # -----------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------

    def test_empty_sql_qmark(self):
        assert resolve_parameters("", (1,), "qmark") == ""

    def test_empty_sql_named(self):
        assert resolve_parameters("", {"id": 1}, "named") == ""

    def test_empty_sql_format(self):
        assert resolve_parameters("", (1,), "format") == ""

    def test_empty_params_tuple_qmark(self):
        sql = "SELECT 1"
        resolved = resolve_parameters(sql, (), "qmark")
        assert resolved == sql

    def test_empty_params_dict_named(self):
        sql = "SELECT 1"
        resolved = resolve_parameters(sql, {}, "named")
        assert resolved == sql

    def test_qmark_with_negative_int(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE score > ?",
            (-100,),
            "qmark",
        )
        assert "-100" in resolved

    def test_format_with_float(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE val > %s",
            (1.23,),
            "format",
        )
        assert "1.23" in resolved

    def test_named_with_bytes(self):
        resolved = resolve_parameters(
            "INSERT INTO t (data) VALUES (:data)",
            {"data": b"\xff\x00"},
            "named",
        )
        assert "X'ff00'" in resolved

    def test_qmark_datetime(self):
        dt = datetime(2024, 3, 15, 9, 0, 0)
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE created_at = ?",
            (dt,),
            "qmark",
        )
        assert "2024-03-15" in resolved

    def test_qmark_decimal(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE price = ?",
            (Decimal("9.99"),),
            "qmark",
        )
        assert "9.99" in resolved

    def test_numeric_with_boolean(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE flag = :1",
            (False,),
            "numeric",
        )
        assert "FALSE" in resolved

    def test_named_with_uuid(self):
        uid = UUID("aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb")
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE uid = :uid",
            {"uid": uid},
            "named",
        )
        assert "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb" in resolved

    def test_qmark_all_none(self):
        resolved = resolve_parameters(
            "INSERT INTO t VALUES (?, ?, ?)",
            (None, None, None),
            "qmark",
        )
        assert resolved == "INSERT INTO t VALUES (NULL, NULL, NULL)"

    def test_format_many_placeholders(self):
        n = 100
        sql = "SELECT " + ", ".join(["%s"] * n)
        params = list(range(n))
        resolved = resolve_parameters(sql, params, "format")
        for i in range(n):
            assert str(i) in resolved

    def test_named_alphanumeric_names(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE x1 = :x1 AND y2 = :y2",
            {"x1": 10, "y2": 20},
            "named",
        )
        assert "10" in resolved
        assert "20" in resolved

    def test_pyformat_all_types(self):
        resolved = resolve_parameters(
            "INSERT INTO t VALUES (%(i)s, %(s)s, %(n)s, %(b)s, %(f)s)",
            {"i": 1, "s": "hello", "n": None, "b": True, "f": 2.5},
            "pyformat",
        )
        assert "1" in resolved
        assert "'hello'" in resolved
        assert "NULL" in resolved
        assert "TRUE" in resolved
        assert "2.5" in resolved

    def test_qmark_string_no_quotes_needed_roundtrip(self):
        # Result should parse as valid SQL literal
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE name = ?",
            ("simple",),
            "qmark",
        )
        assert resolved == "SELECT * FROM t WHERE name = 'simple'"

    def test_numeric_large_index(self):
        sql = "SELECT :10"
        resolved = resolve_parameters(sql, tuple(range(1, 11)), "numeric")
        assert "10" in resolved

    def test_qmark_unicode_string(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE city = ?",
            ("\u6771\u4eac",),  # Tokyo in kanji
            "qmark",
        )
        assert "\u6771\u4eac" in resolved

    def test_format_tuple_with_none_first(self):
        resolved = resolve_parameters(
            "INSERT INTO t (a, b) VALUES (%s, %s)",
            (None, "val"),
            "format",
        )
        assert "NULL" in resolved
        assert "'val'" in resolved

    def test_named_no_placeholders(self):
        sql = "SELECT COUNT(*) FROM t"
        resolved = resolve_parameters(sql, {"id": 1}, "named")
        assert resolved == sql

    def test_numeric_no_placeholders(self):
        sql = "SELECT COUNT(*) FROM t"
        resolved = resolve_parameters(sql, (1, 2), "numeric")
        assert resolved == sql

    def test_qmark_bool_is_not_int(self):
        # bool check must happen before int since bool is subclass of int
        resolved_true = resolve_parameters("SELECT ?", (True,), "qmark")
        resolved_one = resolve_parameters("SELECT ?", (1,), "qmark")
        assert "TRUE" in resolved_true
        assert "1" in resolved_one
        assert "TRUE" not in resolved_one
        assert "1" not in resolved_true

    def test_pyformat_with_list_params_uses_format(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE a = %s AND b = %s",
            [5, 6],
            "pyformat",
        )
        assert resolved == "SELECT * FROM t WHERE a = 5 AND b = 6"

    def test_numeric_with_float(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE val = :1",
            (3.14159,),
            "numeric",
        )
        assert "3.14159" in resolved

    def test_named_with_string_containing_colon(self):
        # A value containing a colon should not confuse the resolver
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE x = :x",
            {"x": "a:b"},
            "named",
        )
        assert "'a:b'" in resolved

    def test_format_only_escaped_percent(self):
        sql = "SELECT '100%%' FROM t"
        resolved = resolve_parameters(sql, (), "format")
        assert resolved == sql

    def test_qmark_empty_string_param(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE name = ?",
            ("",),
            "qmark",
        )
        assert resolved == "SELECT * FROM t WHERE name = ''"

    def test_named_with_int_zero(self):
        resolved = resolve_parameters(
            "UPDATE t SET count = :count",
            {"count": 0},
            "named",
        )
        assert "0" in resolved

    def test_numeric_index_out_of_bounds_returns_sql(self):
        sql = "SELECT * FROM t WHERE id = :3"
        resolved = resolve_parameters(sql, (1, 2), "numeric")
        assert resolved == sql

    def test_qmark_bytearray_param(self):
        resolved = resolve_parameters(
            "INSERT INTO t (data) VALUES (?)",
            (bytearray(b"\x01\x02\x03"),),
            "qmark",
        )
        assert "X'010203'" in resolved

    def test_qmark_memoryview_param(self):
        resolved = resolve_parameters(
            "INSERT INTO t (data) VALUES (?)",
            (memoryview(b"\xab\xcd"),),
            "qmark",
        )
        assert "X'abcd'" in resolved

    def test_named_with_none_among_others(self):
        resolved = resolve_parameters(
            "INSERT INTO t (a, b, c) VALUES (:a, :b, :c)",
            {"a": 1, "b": None, "c": "x"},
            "named",
        )
        assert "1" in resolved
        assert "NULL" in resolved
        assert "'x'" in resolved

    def test_pyformat_with_escaped_percent_and_dict(self):
        resolved = resolve_parameters(
            "SELECT 100%% AS pct, %(id)s AS id FROM t",
            {"id": 5},
            "pyformat",
        )
        assert "100%%" in resolved
        assert "5" in resolved

    def test_qmark_sql_with_literals(self):
        # Ensure already-quoted string literals in SQL don't get mangled
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE type = 'fixed' AND id = ?",
            (10,),
            "qmark",
        )
        assert "'fixed'" in resolved
        assert "10" in resolved

    def test_named_deeply_nested_query(self):
        sql = "SELECT * FROM (SELECT id FROM users WHERE id = :uid) sub WHERE sub.id > :min_id"
        resolved = resolve_parameters(sql, {"uid": 1, "min_id": 0}, "named")
        assert "1" in resolved
        assert "0" in resolved
        assert ":uid" not in resolved
        assert ":min_id" not in resolved

    def test_format_with_bool_false(self):
        resolved = resolve_parameters(
            "UPDATE t SET active = %s",
            (False,),
            "format",
        )
        assert "FALSE" in resolved

    def test_numeric_boolean_params(self):
        resolved = resolve_parameters(
            "SELECT :1, :2",
            (True, False),
            "numeric",
        )
        assert "TRUE" in resolved
        assert "FALSE" in resolved

    def test_named_string_with_single_quote(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE phrase = :phrase",
            {"phrase": "it's fine"},
            "named",
        )
        assert "it''s fine" in resolved

    def test_qmark_negative_float(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE delta > ?",
            (-0.001,),
            "qmark",
        )
        assert "-0.001" in resolved

    def test_pyformat_no_placeholders_dict(self):
        sql = "SELECT 1 FROM t"
        resolved = resolve_parameters(sql, {"x": 1}, "pyformat")
        assert resolved == sql

    def test_format_more_placeholders_than_params(self):
        sql = "SELECT %s AND %s AND %s"
        resolved = resolve_parameters(sql, (1, 2), "format")
        assert resolved == "SELECT 1 AND 2 AND %s"

    def test_named_with_date_value(self):
        d = date(2024, 12, 31)
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE created = :dt",
            {"dt": d},
            "named",
        )
        assert "2024-12-31" in resolved

    def test_qmark_decimal_value(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE amount = ?",
            (Decimal("1234.56"),),
            "qmark",
        )
        assert "1234.56" in resolved

    def test_all_paramstyles_return_str(self):
        sql_q = "SELECT ?"
        sql_n = "SELECT :1"
        sql_named = "SELECT :val"
        sql_f = "SELECT %s"
        sql_pf = "SELECT %(val)s"

        assert isinstance(resolve_parameters(sql_q, (1,), "qmark"), str)
        assert isinstance(resolve_parameters(sql_n, (1,), "numeric"), str)
        assert isinstance(resolve_parameters(sql_named, {"val": 1}, "named"), str)
        assert isinstance(resolve_parameters(sql_f, (1,), "format"), str)
        assert isinstance(resolve_parameters(sql_pf, {"val": 1}, "pyformat"), str)

    def test_numeric_pg_cast_not_matched(self):
        """PostgreSQL :: cast syntax must not be matched by :N numeric placeholders."""
        sql = "SELECT val::123 FROM t WHERE id = :1"
        resolved = resolve_parameters(sql, (42,), "numeric")
        assert "::123" in resolved, f"PG cast ::123 was incorrectly resolved: {resolved}"
        assert "42" in resolved

    def test_numeric_pg_cast_double_colon_only(self):
        """::N should be left alone; only :N should be resolved."""
        sql = "SELECT col::2 FROM t"
        resolved = resolve_parameters(sql, ("x",), "numeric")
        assert "::2" in resolved
