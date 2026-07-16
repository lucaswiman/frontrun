"""Focused tests for Redis client scheduling envelopes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from frontrun import _redis_client, _redis_client_async
from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id, set_io_reporter
from frontrun._redis_client import _intercept_pipeline_execute, set_redis_replay_mode


def _capture_accesses(
    command: str, args: tuple[object, ...], *, db: int = 0, host: str = "source"
) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    client = SimpleNamespace(connection_pool=SimpleNamespace(connection_kwargs={"host": host, "port": 6379, "db": db}))
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        assert _redis_client._report_redis_access(command, args, client=client)
    finally:
        set_io_reporter(None)
    return events


@pytest.mark.parametrize(
    ("command", "args", "expected"),
    [
        pytest.param("MOVE", ("k", 1), ("redis:k:db=redis:source:6379/1", "write"), id="move"),
        pytest.param(
            "COPY",
            ("source-key", "destination-key", "DB", 2),
            ("redis:destination-key:db=redis:source:6379/2", "write"),
            id="copy-db",
        ),
        pytest.param(
            "MIGRATE",
            ("destination", 6380, "k", 3, 1000),
            ("redis:k:db=redis:destination:6380/3", "write"),
            id="migrate",
        ),
    ],
)
def test_cross_scope_redis_commands_report_destination_access(
    command: str, args: tuple[object, ...], expected: tuple[str, str]
) -> None:
    """Cross-database/server writes must conflict with destination traffic."""
    assert expected in _capture_accesses(command, args)


def test_swapdb_reports_both_database_keyspaces() -> None:
    events = _capture_accesses("SWAPDB", (1, 2))

    assert ("redis:keyspace:db=redis:source:6379/1", "write") in events
    assert ("redis:keyspace:db=redis:source:6379/2", "write") in events


def test_flushall_conflicts_with_other_database_traffic() -> None:
    """FLUSHALL mutates every database on its server, not just the selected one."""
    flush = _capture_accesses("FLUSHALL", (), db=0)
    other_db_write = _capture_accesses("SET", ("k", "value"), db=1)

    flush_writes = {resource for resource, kind in flush if kind == "write"}
    other_reads_or_writes = {resource for resource, _kind in other_db_write}
    assert flush_writes & other_reads_or_writes


def test_pubsub_channels_are_scoped_to_server_not_database() -> None:
    """Redis Pub/Sub ignores the selected database."""
    db0 = _capture_accesses("PUBLISH", ("updates", "zero"), db=0)
    db1 = _capture_accesses("PUBLISH", ("updates", "one"), db=1)

    db0_writes = {resource for resource, kind in db0 if kind == "write"}
    db1_writes = {resource for resource, kind in db1 if kind == "write"}
    assert db0_writes & db1_writes


def test_host_aliases_share_server_identity() -> None:
    """localhost and 127.0.0.1 reach the same server and must share resources."""
    via_name = _capture_accesses("SET", ("k", "value"), host="localhost")
    via_ip = _capture_accesses("SET", ("k", "value"), host="127.0.0.1")

    name_writes = {resource for resource, kind in via_name if kind == "write"}
    ip_writes = {resource for resource, kind in via_ip if kind == "write"}
    assert name_writes & ip_writes


def test_migrate_destination_host_alias_shares_identity() -> None:
    """MIGRATE destination identity must canonicalize like client identity."""
    migrate = _capture_accesses("MIGRATE", ("localhost", 6379, "k", 0, 1000), host="source")
    destination_write = _capture_accesses("SET", ("k", "value"), host="127.0.0.1")

    migrate_writes = {resource for resource, kind in migrate if kind == "write"}
    destination_writes = {resource for resource, kind in destination_write if kind == "write"}
    assert migrate_writes & destination_writes


def test_migrate_unix_socket_destination_matches_direct_socket_client(tmp_path: Any) -> None:
    """MIGRATE port 0 names a Unix socket and must reuse its resolved server identity."""
    path = str(tmp_path / "redis.sock")
    real_path = str((tmp_path / "redis.sock").resolve())
    _redis_client._unix_path_server_parts[real_path] = ("localhost", "6380")
    migrate = _capture_accesses("MIGRATE", (path, 0, "k", 2, 1000), host="source")
    destination_write = _capture_accesses("SET", ("k", "value"), db=2, host="127.0.0.1")
    destination_write = [(resource.replace(":6379/", ":6380/"), kind) for resource, kind in destination_write]

    migrate_writes = {resource for resource, kind in migrate if kind == "write"}
    direct_writes = {resource for resource, kind in destination_write if kind == "write"}
    assert migrate_writes & direct_writes


def test_migrate_unix_socket_auth_parser_consumes_values_and_stops_at_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """AUTH-like credential/key values are data, not additional options."""
    seen: list[dict[str, Any]] = []

    def query(_path: str, connection_kwargs: dict[str, Any]) -> str:
        seen.append(connection_kwargs)
        return "6380"

    monkeypatch.setattr(_redis_client, "_query_unix_socket_tcp_port", query)
    path = str(tmp_path / "redis.sock")

    _redis_client._migrate_destination_scope((path, 0, "k", 2, 1000, "AUTH", "AUTH", "COPY"))
    assert seen.pop() == {"password": "AUTH"}

    _redis_client._unix_path_server_parts.clear()
    _redis_client._migrate_destination_scope((path, 0, "", 2, 1000, "KEYS", "AUTH", "key"))
    assert seen.pop() == {}


def test_migrate_copy_does_not_report_source_write() -> None:
    """MIGRATE COPY retains the source key, so only the destination is written."""
    events = _capture_accesses("MIGRATE", ("destination", 6380, "k", 3, 1000, "COPY"), host="source")

    assert ("redis:k:db=redis:source:6379/0", "read") in events
    assert ("redis:k:db=redis:source:6379/0", "write") not in events
    assert ("redis:k:db=redis:destination:6380/3", "write") in events


def test_unresolved_unix_socket_identity_fails_closed(tmp_path: Any) -> None:
    """An unresolved socket may alias TCP, so a per-path scope is unsound."""
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"path": str(tmp_path / "absent.sock"), "db": 0})
    )
    _redis_client._unix_path_server_parts.clear()

    with pytest.raises(RuntimeError, match="unix-socket.*identity"):
        _redis_client._get_redis_scope_parts(client)


def test_select_on_single_connection_client_updates_db_scope() -> None:
    """A live SELECT moves a single-connection client's later accesses."""
    events: list[tuple[str, str]] = []
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
        connection=object(),  # redis-py sets .connection for single_connection_client
    )
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        _redis_client._record_client_select(client, 1)
        assert _redis_client._report_redis_access("SET", ("k", "value"), client=client)
    finally:
        set_io_reporter(None)

    writes = {resource for resource, kind in events if kind == "write"}
    assert "redis:k:db=redis:source:6379/1" in writes
    assert "redis:k:db=redis:source:6379/0" not in writes


def test_select_on_pooled_client_reports_both_db_scopes() -> None:
    """Only one pooled connection switched; both databases stay candidates."""
    events: list[tuple[str, str]] = []
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
    )
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        _redis_client._record_client_select(client, 1)
        assert _redis_client._report_redis_access("SET", ("k", "value"), client=client)
    finally:
        set_io_reporter(None)

    writes = {resource for resource, kind in events if kind == "write"}
    assert "redis:k:db=redis:source:6379/0" in writes
    assert "redis:k:db=redis:source:6379/1" in writes


def test_select_state_is_not_keyed_by_reusable_id_alone() -> None:
    """SELECT state must live on the client (or a weakref-evicted entry).

    id() values are reusable: a bare id-keyed registry entry for a dead
    client would hand its selected database to whatever object next occupies
    the address, silently mis-scoping an unrelated client.
    """
    settable = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
        connection=object(),
    )
    _redis_client._record_client_select(settable, 1)
    assert id(settable) not in _redis_client._client_exact_dbs
    assert _redis_client._get_redis_scope_parts(settable) == ("source", "6379", "1")

    class _Untrackable:  # neither attribute-settable nor weakref-able
        __slots__ = ("connection", "connection_pool")

    untrackable = _Untrackable()
    untrackable.connection_pool = SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0})
    untrackable.connection = object()
    _redis_client._record_client_select(untrackable, 1)
    assert id(untrackable) not in _redis_client._client_exact_dbs
    # Untrackable clients keep the configured scope rather than risking a
    # stale one for a future client at the same address.
    assert _redis_client._get_redis_scope_parts(untrackable) == ("source", "6379", "0")


def test_select_recorded_without_reporter_for_replay_alignment() -> None:
    """A SELECT seen while no reporter is installed (setup/replay) must still
    move the scope, or recorded anchors and replay anchors diverge."""
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
        connection=object(),
    )
    set_io_reporter(None)
    assert _redis_client._intercept_execute_command(lambda *_args, **_kwargs: "OK", client, "SELECT", 1) == "OK"

    events: list[tuple[str, str]] = []
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        assert _redis_client._report_redis_access("SET", ("k", "value"), client=client)
    finally:
        set_io_reporter(None)

    assert ("redis:k:db=redis:source:6379/1", "write") in events


def test_failed_select_does_not_change_single_connection_scope() -> None:
    """A rejected SELECT leaves the physical connection in its old database."""
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
        connection=object(),
    )

    def fail(_client: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("invalid database")

    with pytest.raises(RuntimeError, match="invalid database"):
        _redis_client._intercept_execute_command(fail, client, "SELECT", 1)

    assert _redis_client._get_redis_scope_parts(client) == ("source", "6379", "0")


def test_successful_select_state_is_shared_through_connection_pool() -> None:
    """A selected pooled connection may be reused by a sibling Redis client."""
    pool = SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0})
    selecting_client = SimpleNamespace(connection_pool=pool)
    sibling_client = SimpleNamespace(connection_pool=pool)

    assert (
        _redis_client._intercept_execute_command(lambda *_args, **_kwargs: "OK", selecting_client, "SELECT", 1) == "OK"
    )
    events: list[tuple[str, str]] = []
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        assert _redis_client._report_redis_access("GET", ("k",), client=sibling_client)
    finally:
        set_io_reporter(None)

    assert ("redis:k:db=redis:source:6379/1", "read") in events


def test_async_failed_select_does_not_change_single_connection_scope() -> None:
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
        connection=object(),
    )

    async def fail(_client: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("invalid database")

    with pytest.raises(RuntimeError, match="invalid database"):
        asyncio.run(_redis_client_async._intercept_execute_command_async(fail, client, "SELECT", 1))

    assert _redis_client._get_redis_scope_parts(client) == ("source", "6379", "0")


def test_bytes_flushall_command_reports_server_scope() -> None:
    """redis-py may pass command tokens as bytes; scope semantics must survive normalization."""
    events: list[tuple[str, str]] = []
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0})
    )
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        parsed = _redis_client._parse_and_report_execute_command((b"FLUSHALL",), client)
    finally:
        set_io_reporter(None)

    assert parsed is not None
    assert ("redis:server-keyspace:server=redis:source:6379", "write") in events


def test_coredis_command_request_reports_serialized_command_and_key() -> None:
    """coredis 6 passes one CommandRequest instead of command + arguments."""
    events: list[tuple[str, str]] = []
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0})
    )
    request = SimpleNamespace(name=b"SET", serialized_arguments=(b"key", b"value"))
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        parsed = _redis_client._parse_and_report_execute_command((request,), client)
    finally:
        set_io_reporter(None)

    assert parsed == ("SET", (b"key", b"value"), True)
    assert ("redis:key:db=redis:source:6379/0", "write") in events


def test_bytes_pipeline_move_command_reports_destination_scope() -> None:
    """Pipeline byte command tokens must retain cross-database destination dependencies."""
    events: list[tuple[str, str]] = []
    pipeline = SimpleNamespace(
        command_stack=[((b"MOVE", b"k", 1), {})],
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0}),
    )
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        assert _redis_client._report_pipeline_commands(pipeline)
    finally:
        set_io_reporter(None)

    assert ("redis:k:db=redis:source:6379/1", "write") in events


def test_pattern_subscription_conflicts_with_matching_publish() -> None:
    """Pattern subscriptions overlap dynamically named matching channels."""
    pattern = _capture_accesses("PSUBSCRIBE", ("events:*",))
    publish = _capture_accesses("PUBLISH", ("events:created", "payload"))

    pattern_reads = {resource for resource, kind in pattern if kind == "read"}
    publish_writes = {resource for resource, kind in publish if kind == "write"}
    assert pattern_reads & publish_writes


@pytest.mark.parametrize("command", ["UNSUBSCRIBE", "PUNSUBSCRIBE", "SUNSUBSCRIBE", "PUBSUB"])
def test_pubsub_state_commands_conflict_with_publish(command: str) -> None:
    """Unknown subscription sets and server introspection need conservative scope."""
    state_access = _capture_accesses(command, ())
    publish = _capture_accesses("PUBLISH", ("events", "payload"))

    state_resources = {resource for resource, _kind in state_access}
    publish_resources = {resource for resource, _kind in publish}
    assert state_resources & publish_resources


def test_sync_pubsub_execute_command_is_intercepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """redis-py subscriptions execute on PubSub rather than Redis itself."""

    class PubSub:
        connection_pool = SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0})

        def execute_command(self, *args: Any, **_kwargs: Any) -> tuple[Any, ...]:
            return args

    def fake_import(name: str) -> Any:
        if name == "redis.client":
            return SimpleNamespace(PubSub=PubSub)
        raise ImportError(name)

    events: list[tuple[str, str]] = []
    _redis_client.unpatch_redis()
    monkeypatch.setattr(_redis_client.importlib, "import_module", fake_import)
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        _redis_client.patch_redis()
        result = PubSub().execute_command("SUBSCRIBE", "events")
    finally:
        _redis_client.unpatch_redis()
        set_io_reporter(None)

    assert result == ("SUBSCRIBE", "events")
    assert any(resource.startswith("redis:channel:events") and kind == "read" for resource, kind in events)


def test_async_pubsub_execute_command_is_intercepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """async redis-py subscriptions also execute on a dedicated PubSub class."""

    class PubSub:
        connection_pool = SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": 0})

        async def execute_command(self, *args: Any, **_kwargs: Any) -> tuple[Any, ...]:
            return args

    def fake_import(name: str) -> Any:
        if name == "redis.asyncio.client":
            return SimpleNamespace(PubSub=PubSub)
        raise ImportError(name)

    async def run() -> tuple[Any, ...]:
        _redis_client_async.unpatch_redis_async()
        monkeypatch.setattr(_redis_client_async.importlib, "import_module", fake_import)
        _redis_client_async.patch_redis_async()
        try:
            return await PubSub().execute_command("PSUBSCRIBE", "events:*")
        finally:
            _redis_client_async.unpatch_redis_async()

    events: list[tuple[str, str]] = []
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        result = asyncio.run(run())
    finally:
        set_io_reporter(None)

    assert result == ("PSUBSCRIBE", "events:*")
    assert any(resource.startswith("redis:pubsub:server=") and kind == "read" for resource, kind in events)


def test_memoryview_key_matches_equivalent_bytes_key() -> None:
    """redis-py accepts memoryview values and sends their bytes unchanged."""
    write = _capture_accesses("SET", (memoryview(b"shared"), "value"))
    read = _capture_accesses("GET", (b"shared",))

    write_resources = {resource for resource, kind in write if kind == "write"}
    read_resources = {resource for resource, kind in read if kind == "read"}
    assert write_resources & read_resources


def test_copy_source_named_db_does_not_masquerade_as_option() -> None:
    """COPY options start after the required source and destination keys."""
    copy = _capture_accesses("COPY", ("DB", "destination"), db=0)
    destination_write = _capture_accesses("SET", ("destination", "value"), db=0)

    copy_writes = {resource for resource, kind in copy if kind == "write"}
    destination_resources = {resource for resource, _kind in destination_write}
    assert copy_writes & destination_resources


def test_sync_pipeline_watch_immediate_command_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """redis-py WATCH bypasses Pipeline.execute and must be intercepted directly."""

    class Pipeline:
        command_stack: list[Any] = []

        def execute(self) -> str:
            return "pipeline"

        def immediate_execute_command(self, *args: Any, **_kwargs: Any) -> tuple[Any, ...]:
            return args

    def fake_import(name: str) -> Any:
        if name == "redis.client":
            return SimpleNamespace(Pipeline=Pipeline)
        raise ImportError(name)

    events: list[tuple[str, str]] = []
    _redis_client.unpatch_redis()
    monkeypatch.setattr(_redis_client.importlib, "import_module", fake_import)
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        _redis_client.patch_redis()
        result = Pipeline().immediate_execute_command("WATCH", "watched-key")
    finally:
        _redis_client.unpatch_redis()
        set_io_reporter(None)

    assert result == ("WATCH", "watched-key")
    assert ("redis:watched-key", "read") in events


def test_async_pipeline_watch_immediate_command_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async redis-py WATCH wire call needs the same key-level report."""

    class Pipeline:
        command_stack: list[Any] = []

        async def execute(self) -> str:
            return "pipeline"

        async def immediate_execute_command(self, *args: Any, **_kwargs: Any) -> tuple[Any, ...]:
            return args

    def fake_import(name: str) -> Any:
        if name in {"redis.asyncio", "redis.asyncio.client"}:
            return SimpleNamespace(Pipeline=Pipeline)
        raise ImportError(name)

    async def run() -> tuple[Any, ...]:
        _redis_client_async.unpatch_redis_async()
        monkeypatch.setattr(_redis_client_async.importlib, "import_module", fake_import)
        _redis_client_async.patch_redis_async()
        try:
            return await Pipeline().immediate_execute_command("WATCH", "watched-key")
        finally:
            _redis_client_async.unpatch_redis_async()

    events: list[tuple[str, str]] = []
    set_io_reporter(lambda resource_id, kind: events.append((resource_id, kind)))
    try:
        result = asyncio.run(run())
    finally:
        set_io_reporter(None)

    assert result == ("WATCH", "watched-key")
    assert ("redis:watched-key", "read") in events


@pytest.mark.parametrize(
    ("command_stack", "expected_boundaries"),
    [
        pytest.param([], 0, id="empty"),
        pytest.param([(("PING",), {})], 0, id="keyless"),
        pytest.param([(("GET", "cache-key"), {})], 1, id="keyed"),
    ],
)
def test_pipeline_replay_recreates_only_exploration_boundaries(
    command_stack: list[Any], expected_boundaries: int
) -> None:
    """Replay must not invent a boundary for a pipeline exploration skipped."""

    class Scheduler:
        def __init__(self) -> None:
            self.before: list[str] = []
            self.after: list[str] = []

        def before_io(self, _thread_id: int, resource_id: str) -> None:
            self.before.append(resource_id)

        def after_io(self, _thread_id: int, resource_id: str) -> None:
            self.after.append(resource_id)

    scheduler = Scheduler()
    pipeline = SimpleNamespace(command_stack=command_stack)
    set_io_reporter(None)  # replay has no access reporter
    set_dpor_scheduler(scheduler)
    set_dpor_thread_id(0)
    set_redis_replay_mode(True)
    try:
        result = _intercept_pipeline_execute(lambda _pipeline: "executed", pipeline)
    finally:
        set_redis_replay_mode(False)
        set_dpor_scheduler(None)
        set_dpor_thread_id(None)

    assert result == "executed"
    assert scheduler.before == ["redis:PIPELINE"] * expected_boundaries
    assert scheduler.after == ["redis:PIPELINE"] * expected_boundaries
