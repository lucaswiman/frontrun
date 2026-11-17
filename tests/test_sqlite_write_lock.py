"""SQLite database writer arbitration and transaction boundaries."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from collections.abc import Callable
from itertools import count
from pathlib import Path
from typing import Any

import pytest

import frontrun

# Long enough that a worker blocked in SQLite's busy handler outlasts the
# per-execution timeout below: an unmodeled wait shows up as a timeout.
_BUSY_TIMEOUT = 30.0
_JOURNAL_MODES = ["wal", "delete"]


class _Db:
    def __init__(self, path: Path, journal_mode: str) -> None:
        self.path = path
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute(f"PRAGMA journal_mode={journal_mode}")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)")
        conn.execute("INSERT INTO t VALUES (1, 0)")
        conn.close()

    def connect(self, **kwargs: Any) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT, check_same_thread=False, **kwargs)

    def value(self) -> int:
        conn = sqlite3.connect(self.path)
        try:
            (v,) = conn.execute("SELECT v FROM t WHERE id = 1").fetchone()
            return int(v)
        finally:
            conn.close()


def _explore(
    tmp_path: Path,
    journal_mode: str,
    workers: list[Callable[[_Db], None]],
    invariant: Callable[[_Db], bool],
) -> Any:
    runs = count()
    return frontrun.explore(
        setup=lambda: _Db(tmp_path / f"{next(runs)}.sqlite", journal_mode),
        workers=workers,
        invariant=invariant,
        max_executions=40,
        timeout_per_run=3.0,
        total_timeout=120,
        reproduce_on_failure=3,
    )


def _increment_in_immediate_transaction(db: _Db) -> None:
    conn = db.connect(isolation_level=None)
    conn.execute("BEGIN IMMEDIATE")
    (v,) = conn.execute("SELECT v FROM t WHERE id = 1").fetchone()
    conn.execute("UPDATE t SET v = ? WHERE id = 1", (v + 1,))
    conn.execute("COMMIT")
    conn.close()


def _increment_in_deferred_transaction(db: _Db) -> None:
    conn = db.connect(isolation_level=None)
    conn.execute("BEGIN")
    conn.execute("UPDATE t SET v = v + 1 WHERE id = 1")
    (v,) = conn.execute("SELECT v FROM t WHERE id = 1").fetchone()
    conn.execute("UPDATE t SET v = ? WHERE id = 1", (v,))
    conn.execute("COMMIT")
    conn.close()


def _increment_in_implicit_transaction(db: _Db) -> None:
    conn = db.connect()  # legacy transaction control: DML opens a transaction
    conn.execute("UPDATE t SET v = v + 1 WHERE id = 1")
    conn.execute("UPDATE t SET v = v + 0 WHERE id = 1")
    conn.commit()
    conn.close()


def _increment_in_context_manager(db: _Db) -> None:
    conn = db.connect()
    with conn:
        conn.execute("UPDATE t SET v = v + 1 WHERE id = 1")
        conn.execute("UPDATE t SET v = v + 0 WHERE id = 1")
    conn.close()


@pytest.mark.parametrize("journal_mode", _JOURNAL_MODES)
@pytest.mark.parametrize(
    "worker",
    [
        _increment_in_immediate_transaction,
        _increment_in_deferred_transaction,
        _increment_in_implicit_transaction,
        _increment_in_context_manager,
    ],
    ids=["immediate", "deferred", "implicit", "context-manager"],
)
def test_write_transactions_serialize(tmp_path: Path, journal_mode: str, worker: Callable[[_Db], None]) -> None:
    result = _explore(tmp_path, journal_mode, [worker, worker], lambda db: db.value() == 2)
    result.assert_holds()


def test_immediate_and_deferred_writers_share_database_lock(tmp_path: Path) -> None:
    result = _explore(
        tmp_path,
        "wal",
        [_increment_in_immediate_transaction, _increment_in_deferred_transaction],
        lambda db: db.value() == 2,
    )
    result.assert_holds()


def _two_transactions(mode: str) -> Callable[[_Db], None]:
    def worker(db: _Db) -> None:
        if mode == "pep249":
            conn = db.connect(autocommit=False)
        elif mode in {"implicit", "context-manager"}:
            conn = db.connect()
        else:
            conn = db.connect(isolation_level=None)
        for sql in ("UPDATE t SET v = 1 WHERE id = 1", "UPDATE t SET v = v + 1 WHERE id = 1"):
            if mode == "context-manager":
                with conn:
                    conn.execute(sql)
            else:
                if mode.startswith("BEGIN"):
                    conn.execute(mode)
                conn.execute(sql)
                if mode.startswith("BEGIN"):
                    conn.execute("COMMIT")
                elif mode != "autocommit":
                    conn.commit()
        conn.close()

    return worker


def _times_ten(db: _Db) -> None:
    conn = db.connect(isolation_level=None)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE t SET v = v * 10 WHERE id = 1")
    conn.execute("COMMIT")
    conn.close()


@pytest.mark.parametrize("journal_mode", _JOURNAL_MODES)
@pytest.mark.parametrize(
    "mode",
    [
        "BEGIN IMMEDIATE",
        "BEGIN",
        "implicit",
        pytest.param("pep249", marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 249 mode")),
        "context-manager",
        "autocommit",
    ],
    ids=["immediate", "deferred", "implicit", "pep249", "context-manager", "autocommit"],
)
def test_write_lock_is_released_when_the_transaction_ends(tmp_path: Path, journal_mode: str, mode: str) -> None:
    # Serial outcomes are 2 (times-ten first) and 20; 11 needs the second
    # worker's transaction to run between the first worker's two.
    result = _explore(tmp_path, journal_mode, [_two_transactions(mode), _times_ten], lambda db: db.value() in (2, 20))
    assert result.property_holds is False, getattr(result, "inconclusive_reason", None)
    assert result.reproduction_successes == result.reproduction_attempts == 3


def _increment_after_failed_insert(db: _Db) -> None:
    conn = db.connect(isolation_level=None)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE t SET v = v + 1 WHERE id = 1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO t VALUES (1, 0)")
    conn.execute("UPDATE t SET v = v + 0 WHERE id = 1")
    conn.execute("COMMIT")
    conn.close()


@pytest.mark.parametrize("journal_mode", _JOURNAL_MODES)
def test_failed_statement_keeps_the_transaction_write_lock(tmp_path: Path, journal_mode: str) -> None:
    worker = _increment_after_failed_insert
    result = _explore(tmp_path, journal_mode, [worker, worker], lambda db: db.value() == 2)
    result.assert_holds()


async def _aio_increment_in_immediate_transaction(db: _Db) -> None:
    import aiosqlite  # type: ignore[import-untyped]

    async with aiosqlite.connect(db.path, timeout=_BUSY_TIMEOUT, isolation_level=None) as conn:
        await conn.execute("BEGIN IMMEDIATE")
        cursor = await conn.execute("SELECT v FROM t WHERE id = 1")
        row = await cursor.fetchone()
        await conn.execute("UPDATE t SET v = ? WHERE id = 1", (row[0] + 1,))
        await conn.execute("COMMIT")


async def _aio_increment_in_implicit_transaction(db: _Db) -> None:
    import aiosqlite  # type: ignore[import-untyped]

    async with aiosqlite.connect(db.path, timeout=_BUSY_TIMEOUT) as conn:
        await conn.execute("UPDATE t SET v = v + 1 WHERE id = 1")
        await conn.execute("UPDATE t SET v = v + 0 WHERE id = 1")
        await conn.commit()


async def _aio_increment_after_failed_insert(db: _Db) -> None:
    import aiosqlite  # type: ignore[import-untyped]

    async with aiosqlite.connect(db.path, timeout=_BUSY_TIMEOUT, isolation_level=None) as conn:
        await conn.execute("BEGIN IMMEDIATE")
        await conn.execute("UPDATE t SET v = v + 1 WHERE id = 1")
        with pytest.raises(sqlite3.IntegrityError):
            await conn.execute("INSERT INTO t VALUES (1, 0)")
        await conn.execute("UPDATE t SET v = v + 0 WHERE id = 1")
        await conn.execute("COMMIT")


@pytest.mark.parametrize("journal_mode", _JOURNAL_MODES)
@pytest.mark.parametrize(
    "worker",
    [
        _aio_increment_in_immediate_transaction,
        _aio_increment_in_implicit_transaction,
        _aio_increment_after_failed_insert,
    ],
    ids=["immediate", "implicit", "failed-statement"],
)
def test_aiosqlite_write_transactions_serialize(
    tmp_path: Path, journal_mode: str, worker: Callable[[_Db], Any]
) -> None:
    pytest.importorskip("aiosqlite")

    result = asyncio.run(_explore(tmp_path, journal_mode, [worker, worker], lambda db: db.value() == 2))
    result.assert_holds()


@pytest.mark.parametrize(
    ("sql", "needs_lock"),
    [
        ("SELECT v FROM t WHERE id = 1", False),
        ("select count(*) from t", False),
        ("BEGIN", False),
        ("BEGIN DEFERRED TRANSACTION", False),
        ("BEGIN IMMEDIATE", True),
        ("begin exclusive transaction", True),
        ("COMMIT", False),
        ("END", False),
        ("END /* transaction */", False),
        ("-- mode\nBEGIN /* writer */ IMMEDIATE", True),
        ("ROLLBACK", False),
        ("SAVEPOINT sp", False),
        ("RELEASE sp", False),
        ("ROLLBACK TO sp", False),
        ("INSERT INTO t VALUES (2, 0)", True),
        ("UPDATE t SET v = 1", True),
        ("DELETE FROM t", True),
        ("REPLACE INTO t VALUES (1, 0)", True),
        ("CREATE TABLE u (x)", True),
        ("PRAGMA journal_mode=WAL", True),
        ("WITH c AS (SELECT 1) DELETE FROM t", True),
        ("this is not sql", True),
    ],
)
def test_statement_classification(sql: str, needs_lock: bool) -> None:
    from frontrun._sql_sqlite_write_lock import _sqlite_statement_needs_write_lock

    assert _sqlite_statement_needs_write_lock(sql) is needs_lock
