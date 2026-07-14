"""Focused tests for Redis client scheduling envelopes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from frontrun import _redis_client, _redis_client_async
from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id, set_io_reporter
from frontrun._redis_client import _intercept_pipeline_execute, set_redis_replay_mode


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
