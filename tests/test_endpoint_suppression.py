"""Correctness tests for SQL/Redis endpoint-I/O suppression scopes."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest

from frontrun._io_detection import _emit_socket_io, is_endpoint_io_suppressed, set_io_reporter
from frontrun._redis_client import (
    _suppress_endpoint_io as suppress_redis_endpoint_io,
)
from frontrun._redis_client import is_redis_tid_suppressed
from frontrun._sql_endpoint_suppression import (
    _suppress_endpoint_io as suppress_sql_endpoint_io,
)
from frontrun._sql_endpoint_suppression import is_tid_suppressed as is_sql_tid_suppressed

SuppressionScope = Callable[..., AbstractContextManager[None]]


@pytest.fixture(params=["sql", "redis"])
def endpoint_suppression(request: pytest.FixtureRequest) -> tuple[SuppressionScope, Callable[[int], bool]]:
    if request.param == "sql":
        return suppress_sql_endpoint_io, is_sql_tid_suppressed
    return suppress_redis_endpoint_io, is_redis_tid_suppressed


def test_endpoint_suppression_is_nested(
    endpoint_suppression: tuple[SuppressionScope, Callable[[int], bool]],
) -> None:
    suppress, is_tid_suppressed = endpoint_suppression
    events: list[tuple[str, str]] = []
    tid = threading.get_native_id()
    set_io_reporter(lambda resource, kind: events.append((resource, kind)))
    try:
        with suppress():
            with suppress():
                _emit_socket_io("socket:nested", "read")
            assert is_tid_suppressed(tid)
            _emit_socket_io("socket:outer", "read")
        _emit_socket_io("socket:visible", "read")
    finally:
        set_io_reporter(None)

    assert events == [("socket:visible", "read")]
    assert not is_tid_suppressed(tid)


def test_endpoint_suppression_does_not_leak_to_spawned_thread(
    endpoint_suppression: tuple[SuppressionScope, Callable[[int], bool]],
) -> None:
    suppress, _ = endpoint_suppression
    suppress_in_thread: list[bool] = []

    with suppress():
        assert is_endpoint_io_suppressed()
        thread = threading.Thread(target=lambda: suppress_in_thread.append(is_endpoint_io_suppressed()))
        thread.start()
        thread.join()

    assert suppress_in_thread == [False]


@pytest.mark.asyncio
async def test_async_endpoint_suppression_does_not_hide_sibling_task_io(
    endpoint_suppression: tuple[SuppressionScope, Callable[[int], bool]],
) -> None:
    suppress, is_tid_suppressed = endpoint_suppression
    entered = asyncio.Event()
    sibling_reported = asyncio.Event()
    events: list[tuple[str, str]] = []
    tid = threading.get_native_id()
    set_io_reporter(lambda resource, kind: events.append((resource, kind)))

    async def database_task() -> None:
        with suppress(suppress_native_tid=False):
            assert not is_tid_suppressed(tid)
            entered.set()
            await sibling_reported.wait()
            _emit_socket_io("socket:database", "write")

    async def sibling_task() -> None:
        await entered.wait()
        _emit_socket_io("socket:sibling", "read")
        sibling_reported.set()

    try:
        await asyncio.gather(database_task(), sibling_task())
    finally:
        set_io_reporter(None)

    assert events == [("socket:sibling", "read")]
