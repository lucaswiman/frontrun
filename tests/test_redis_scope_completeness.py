"""Scope-completeness safety net for Redis command modeling.

Soundness invariant: over-merging is allowed, under-merging is forbidden.
A Redis command whose semantic accesses are not positively known must be
reported with a COARSE conservative access (database-wide keyspace write
plus server-wide keyspace write) — never with zero accesses and never with
a silently-narrow scope.  Zero accesses are only permitted for commands on
the explicit curated allowlist of genuinely stateless commands.

This file iterates the ENTIRE Redis command table and asserts every command
is exactly one of:

(a) explicitly modeled with keys/scope (key-spec table, keyspace intent
    sets, transaction control, EVAL family, pub/sub, SORT, WATCH, SELECT),
(b) on the explicit no-access allowlist of genuinely stateless commands, or
(c) classified coarse-conservative.

A future command added to the table without scope modeling must fail here,
and a command missing from the table entirely must land in the coarse
bucket.  Cross-scope commands (MOVE, COPY, MIGRATE, SWAPDB, FLUSHDB,
FLUSHALL, SELECT, and the script/function-invoking commands) each have a
named test so a regression in any of them fails this file by name.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from frontrun import _redis_client
from frontrun._io_detection import set_io_reporter
from frontrun._redis_command_data import (
    _COMMAND_KEY_SPECS,
    _EVAL_CMDS,
    _KEYSPACE_ENUMERATION_CMDS,
    _KEYSPACE_WRITE_CMDS,
    _NO_KEY_CMDS,
    _TX_CONTROL_CMDS,
)
from frontrun._redis_parsing import _PUBSUB_CMDS, parse_redis_access

# Resource IDs produced for a client configured as host="source", port=6379, db=0.
_DB_KEYSPACE_WRITE = ("redis:keyspace:db=redis:source:6379/0", "write")
_SERVER_KEYSPACE_WRITE = ("redis:server-keyspace:server=redis:source:6379", "write")

# Commands handled by dedicated report-time logic rather than the key-spec
# table or the classification sets imported above.
_SPECIAL_MODELED = frozenset({"WATCH", "SELECT", "PUBSUB", "SORT", "SORT_RO"})


def _ok(*_args: Any, **_kwargs: Any) -> str:
    return "OK"


def _client(db: int = 0) -> Any:
    return SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": db})
    )


def _capture(command: str, args: tuple[object, ...], client: Any = None) -> tuple[bool, list[tuple[str, str]]]:
    events: list[tuple[str, str]] = []
    if client is None:
        client = _client()
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        reported = _redis_client._report_redis_access(command, args, client=client)
    finally:
        set_io_reporter(None)
    return reported, events


def _assert_coarse(command: str, args: tuple[object, ...]) -> None:
    """The command must produce the conservative database+server-wide write."""
    reported, events = _capture(command, args)
    assert reported, f"{command} {args!r}: not reported at all (zero accesses — false certification risk)"
    assert _DB_KEYSPACE_WRITE in events, f"{command} {args!r}: missing database-wide conservative write: {events}"
    assert _SERVER_KEYSPACE_WRITE in events, f"{command} {args!r}: missing server-wide conservative write: {events}"


def _stateless_allowlist() -> frozenset[str]:
    from frontrun._redis_command_data import _STATELESS_NO_ACCESS_CMDS

    return _STATELESS_NO_ACCESS_CMDS


# ---------------------------------------------------------------------------
# Coarse-by-default: unknown and unparseable commands
# ---------------------------------------------------------------------------


def test_unknown_command_without_args_is_coarse_write() -> None:
    """A command missing from the table entirely must land in the coarse bucket."""
    _assert_coarse("TOTALLYUNKNOWNCMD", ())


def test_unknown_command_with_args_is_coarse_write() -> None:
    """First-arg key guessing is not positive evidence — must also be coarse."""
    _assert_coarse("TOTALLYUNKNOWNCMD", ("k", "v"))


def test_unknown_subcommand_of_modeled_parent_is_coarse_write() -> None:
    """A new subcommand of a dispatching parent (e.g. XGROUP) must be coarse."""
    _assert_coarse("XGROUP", ("SOMETHINGNEW", "stream", "grp"))


@pytest.mark.parametrize(
    ("command", "args"),
    [
        pytest.param("XREAD", ("COUNT", "10"), id="xread-missing-streams-keyword"),
        pytest.param("GET", (), id="get-without-key"),
        pytest.param("MEMORY", ("USAGE",), id="memory-usage-without-key"),
        pytest.param("BLMPOP", ("0", "not-a-number", "k", "LEFT"), id="blmpop-malformed-numkeys"),
    ],
)
def test_key_extraction_failure_is_coarse_write(command: str, args: tuple[object, ...]) -> None:
    """A modeled command whose key extraction fails must not yield zero accesses."""
    _assert_coarse(command, args)


def test_every_key_spec_command_goes_coarse_when_extraction_fails() -> None:
    """Degenerate arity must never produce zero accesses for any table command.

    WATCH is excluded: it is overridden as transaction control, and plain
    transaction control legitimately carries no key accesses.
    """
    for command in sorted(_COMMAND_KEY_SPECS):
        if command in _TX_CONTROL_CMDS or command == "WATCH":
            continue
        reported, events = _capture(command, ())
        assert reported, f"{command}: extraction failure produced zero accesses"
        assert _DB_KEYSPACE_WRITE in events, f"{command}: extraction failure did not fall back to coarse: {events}"


def test_pipeline_unknown_command_is_coarse_write() -> None:
    events: list[tuple[str, str]] = []
    pipeline = SimpleNamespace(
        command_stack=[(("BLORPCMD", "k"), {})],
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
    )
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        assert _redis_client._report_pipeline_commands(pipeline)
    finally:
        set_io_reporter(None)
    assert _DB_KEYSPACE_WRITE in events


# ---------------------------------------------------------------------------
# Server-state commands: known but stateful → coarse, never zero accesses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "args"),
    [
        pytest.param("SCRIPT", ("FLUSH",), id="script"),
        pytest.param("FUNCTION", ("FLUSH",), id="function"),
        pytest.param("CONFIG", ("SET", "maxmemory", "100mb"), id="config"),
        pytest.param("DEBUG", ("SLEEP", "0"), id="debug"),
        pytest.param("ACL", ("SETUSER", "u", "off"), id="acl"),
        pytest.param("MODULE", ("LOAD", "/x.so"), id="module"),
        pytest.param("SHUTDOWN", ("NOSAVE",), id="shutdown"),
        pytest.param("REPLICAOF", ("NO", "ONE"), id="replicaof"),
        pytest.param("SLAVEOF", ("NO", "ONE"), id="slaveof"),
        pytest.param("FAILOVER", (), id="failover"),
        pytest.param("PSYNC", ("?", "-1"), id="psync"),
        pytest.param("SYNC", (), id="sync"),
    ],
)
def test_server_state_commands_are_coarse_write(command: str, args: tuple[object, ...]) -> None:
    """Server-state mutations (script cache, config, replication) affect data-plane
    outcomes (e.g. SCRIPT FLUSH races EVALSHA into NOSCRIPT) and must be coarse."""
    _assert_coarse(command, args)


# ---------------------------------------------------------------------------
# Stateless allowlist: the ONLY route to zero accesses
# ---------------------------------------------------------------------------


def test_stateless_allowlist_commands_report_no_accesses() -> None:
    allowlist = _stateless_allowlist()
    assert allowlist, "stateless allowlist must exist and be non-empty"
    for command in sorted(allowlist):
        reported, events = _capture(command, ())
        assert events == [], f"{command}: allowlisted as stateless but reported accesses: {events}"
        assert not reported, f"{command}: allowlisted as stateless but claimed reporting"


def test_stateless_allowlist_excludes_stateful_commands() -> None:
    """Each allowlist entry must be genuinely stateless; these are provably not."""
    allowlist = _stateless_allowlist()
    stateful = {
        # keyspace-wide mutations / scans
        "FLUSHDB",
        "FLUSHALL",
        "SWAPDB",
        "KEYS",
        "SCAN",
        "RANDOMKEY",
        "DBSIZE",
        # connection scope
        "SELECT",
        # server state that changes data-plane outcomes
        "SCRIPT",
        "FUNCTION",
        "CONFIG",
        "DEBUG",
        "ACL",
        "MODULE",
        "SHUTDOWN",
        "REPLICAOF",
        "SLAVEOF",
        "FAILOVER",
        "PSYNC",
        "SYNC",
    }
    overlap = allowlist & stateful
    assert not overlap, f"stateful commands on the stateless allowlist: {sorted(overlap)}"


def test_stateless_allowlist_is_disjoint_from_modeled_sets() -> None:
    allowlist = _stateless_allowlist()
    table_parents = {cmd.split(" ", 1)[0] for cmd in _COMMAND_KEY_SPECS}
    for name, modeled in [
        ("key-spec table", table_parents),
        ("tx control", _TX_CONTROL_CMDS),
        ("eval family", _EVAL_CMDS),
        ("keyspace writes", _KEYSPACE_WRITE_CMDS),
        ("keyspace reads", _KEYSPACE_ENUMERATION_CMDS),
        ("pub/sub", _PUBSUB_CMDS),
        ("special modeled", _SPECIAL_MODELED),
    ]:
        overlap = allowlist & set(modeled)
        assert not overlap, f"allowlist overlaps {name}: {sorted(overlap)}"


# ---------------------------------------------------------------------------
# Exhaustive partition over the entire command universe
# ---------------------------------------------------------------------------


def _command_universe() -> set[str]:
    universe = {cmd.split(" ", 1)[0] for cmd in _COMMAND_KEY_SPECS}
    universe |= set(_NO_KEY_CMDS)
    universe |= set(_TX_CONTROL_CMDS)
    universe |= set(_EVAL_CMDS)
    universe |= set(_KEYSPACE_WRITE_CMDS)
    universe |= set(_KEYSPACE_ENUMERATION_CMDS)
    universe |= set(_PUBSUB_CMDS)
    universe |= set(_SPECIAL_MODELED)
    return universe


def test_every_command_in_table_is_classified_exactly_once() -> None:
    """Every command is (a) explicitly modeled, (b) allowlisted stateless, or
    (c) coarse-conservative — and never silently none of the three."""
    from frontrun._redis_client import _needs_coarse_fallback

    allowlist = _stateless_allowlist()
    table_parents = {cmd.split(" ", 1)[0] for cmd in _COMMAND_KEY_SPECS}
    modeled = (
        table_parents
        | set(_TX_CONTROL_CMDS)
        | set(_EVAL_CMDS)
        | set(_KEYSPACE_WRITE_CMDS)
        | set(_KEYSPACE_ENUMERATION_CMDS)
        | set(_PUBSUB_CMDS)
        | _SPECIAL_MODELED
    )

    for command in sorted(_command_universe()):
        in_modeled = command in modeled
        in_allowlist = command in allowlist
        assert not (in_modeled and in_allowlist), f"{command}: both modeled and allowlisted — ambiguous"
        if in_modeled or in_allowlist:
            continue
        # Residual bucket: the command must be coarse-conservative, both by
        # classification and by observed reporting behavior.
        access = parse_redis_access(command, ())
        assert _needs_coarse_fallback(command, (), access), (
            f"{command}: not modeled, not allowlisted, not coarse — unsound zero-access default"
        )
        _assert_coarse(command, ())


# ---------------------------------------------------------------------------
# Cross-scope commands: named regression tests
# ---------------------------------------------------------------------------


def test_move_reports_destination_database() -> None:
    _, events = _capture("MOVE", ("k", 1))
    assert ("redis:k:db=redis:source:6379/1", "write") in events


def test_copy_reports_destination_database() -> None:
    _, events = _capture("COPY", ("src", "dst", "DB", 2))
    assert ("redis:dst:db=redis:source:6379/2", "write") in events


def test_migrate_reports_destination_server() -> None:
    _, events = _capture("MIGRATE", ("desthost", 6380, "k", 3, 1000))
    assert ("redis:k:db=redis:desthost:6380/3", "write") in events


def test_swapdb_reports_both_database_keyspaces() -> None:
    _, events = _capture("SWAPDB", (1, 2))
    assert ("redis:keyspace:db=redis:source:6379/1", "write") in events
    assert ("redis:keyspace:db=redis:source:6379/2", "write") in events


def test_flushdb_reports_database_keyspace_write() -> None:
    _, events = _capture("FLUSHDB", ())
    assert _DB_KEYSPACE_WRITE in events


def test_flushall_reports_server_keyspace_write() -> None:
    _, events = _capture("FLUSHALL", ())
    assert _SERVER_KEYSPACE_WRITE in events


def test_select_rescopes_later_commands() -> None:
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
        connection=object(),  # single_connection_client
    )
    assert _redis_client._intercept_execute_command(_ok, client, "SELECT", 1) == "OK"
    _, events = _capture("SET", ("k", "v"), client=client)
    assert ("redis:k:db=redis:source:6379/1", "write") in events


def test_reset_restores_default_database_candidate() -> None:
    """RESET returns the connection to database 0; later accesses must cover it."""
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
        connection=object(),
    )
    assert _redis_client._intercept_execute_command(_ok, client, "SELECT", 1) == "OK"
    assert _redis_client._intercept_execute_command(_ok, client, "RESET") == "OK"
    _, events = _capture("SET", ("k", "v"), client=client)
    assert ("redis:k:db=redis:source:6379/0", "write") in events


@pytest.mark.parametrize(
    ("command", "args", "expected_key", "expected_kind"),
    [
        pytest.param("EVAL", ("return 1", 1, "k"), "redis:k:db=redis:source:6379/0", "write", id="eval"),
        pytest.param("EVALSHA", ("sha", 1, "k"), "redis:k:db=redis:source:6379/0", "write", id="evalsha"),
        pytest.param("EVAL_RO", ("return 1", 1, "k"), "redis:k:db=redis:source:6379/0", "read", id="eval-ro"),
        pytest.param("EVALSHA_RO", ("sha", 1, "k"), "redis:k:db=redis:source:6379/0", "read", id="evalsha-ro"),
        pytest.param("FCALL", ("fn", 1, "k"), "redis:k:db=redis:source:6379/0", "write", id="fcall"),
        pytest.param("FCALL_RO", ("fn", 1, "k"), "redis:k:db=redis:source:6379/0", "read", id="fcall-ro"),
    ],
)
def test_script_invocation_reports_declared_keys(
    command: str, args: tuple[object, ...], expected_key: str, expected_kind: str
) -> None:
    _, events = _capture(command, args)
    assert (expected_key, expected_kind) in events


@pytest.mark.parametrize("command", ["EVAL", "EVALSHA", "FCALL"])
def test_script_invocation_with_no_declared_keys_is_coarse(command: str) -> None:
    """A script declaring zero KEYS may still touch anything — coarse, not zero."""
    _assert_coarse(command, ("script-or-sha-or-fn", 0))


@pytest.mark.parametrize("command", ["SCRIPT", "FUNCTION"])
def test_script_management_commands_are_coarse(command: str) -> None:
    """SCRIPT/FUNCTION FLUSH+LOAD race EVALSHA/FCALL — they must be coarse."""
    _assert_coarse(command, ("FLUSH",))


# ---------------------------------------------------------------------------
# Performance guard: well-modeled commands must NOT become coarse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "args"),
    [
        pytest.param("GET", ("k",), id="get"),
        pytest.param("SET", ("k", "v"), id="set"),
        pytest.param("HSET", ("h", "f", "v"), id="hset"),
        pytest.param("DEL", ("k1", "k2"), id="del"),
        pytest.param("MULTI", (), id="multi"),
        pytest.param("WATCH", ("k",), id="watch"),
        pytest.param("PUBLISH", ("chan", "msg"), id="publish"),
        pytest.param("EVAL", ("return 1", 1, "k"), id="eval-declared"),
    ],
)
def test_well_modeled_commands_are_not_coarse(command: str, args: tuple[object, ...]) -> None:
    _, events = _capture(command, args)
    assert _DB_KEYSPACE_WRITE not in events, f"{command}: modeled command escalated to coarse: {events}"
    assert _SERVER_KEYSPACE_WRITE not in events


@pytest.mark.parametrize("command", ["PING", "ECHO", "AUTH", "HELLO", "CLIENT"])
def test_connection_handshake_commands_stay_no_access(command: str) -> None:
    """redis-py issues these on every connection; they must not gain accesses."""
    reported, events = _capture(command, ("x",))
    assert events == []
    assert not reported
