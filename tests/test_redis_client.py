"""Focused tests for Redis client scheduling envelopes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from frontrun import _redis_client, _redis_client_async
from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id, set_io_reporter
from frontrun._redis_client import _intercept_pipeline_execute, set_redis_replay_mode


def _capture_accesses(command: str, args: tuple[object, ...], *, db: int = 0) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    client = SimpleNamespace(
        connection_pool=SimpleNamespace(connection_kwargs={"host": "source", "port": 6379, "db": db})
    )
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
