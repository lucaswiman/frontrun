"""Focused tests for Redis client scheduling envelopes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id, set_io_reporter
from frontrun._redis_client import _intercept_pipeline_execute, set_redis_replay_mode


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
