"""Membership-changing SQL must conflict with concurrent predicate reads."""

from unittest.mock import Mock

import pytest

from frontrun import _sql_cursor


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("SELECT * FROM users", "read"),
        ("UPDATE users SET age = age + 1", "read"),
        ("DELETE FROM users", "write"),
        ("INSERT INTO users VALUES (1, 'Alice', 30)", "write"),
        ("INSERT INTO users SELECT * FROM users", "write"),
        ("SELECT * FROM users; INSERT INTO users VALUES (1, 'Alice', 30)", "write"),
        ("WITH added AS (INSERT INTO users VALUES (1, 'Alice', 30) RETURNING *) SELECT * FROM users", "write"),
        (
            "MERGE INTO users USING (SELECT 1 AS id) AS src ON users.id = src.id "
            "WHEN NOT MATCHED THEN INSERT (id) VALUES (src.id)",
            "write",
        ),
        (
            "MERGE INTO users USING (SELECT 1 AS id) AS src ON users.id = src.id WHEN MATCHED THEN DELETE",
            "write",
        ),
        (
            "MERGE INTO users USING (SELECT 1 AS id) AS src ON users.id = src.id WHEN MATCHED THEN UPDATE SET age = 30",
            "read",
        ),
    ],
)
def test_membership_access(sql: str, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    reporter = Mock()
    monkeypatch.setattr(_sql_cursor, "get_io_reporter", lambda: reporter)
    _sql_cursor._report_sql_access(sql)
    membership = [call.args for call in reporter.call_args_list if call.args[0].endswith(":seq")]
    assert membership == [("sql:users:seq", kind)]
