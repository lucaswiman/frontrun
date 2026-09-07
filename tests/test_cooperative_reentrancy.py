"""Scheduler callbacks and GC must not recursively enter cooperative scheduling."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontrun._cooperative import (
    CooperativeLock,
    CooperativeRLock,
    CooperativeSemaphore,
    _scheduler_tls,
    set_context,
    set_sync_reporter,
)
from frontrun._deadlock import WaitForGraph, install_wait_for_graph, uninstall_wait_for_graph


@pytest.fixture
def scheduler_context():
    scheduler = SimpleNamespace(_finished=False, _error=None, wait_for_turn=Mock(), report_error=Mock())
    graph = install_wait_for_graph()
    events = []
    set_context(scheduler, 0)
    set_sync_reporter(lambda event, obj_id, lock_obj: events.append(event))
    try:
        yield scheduler, graph, events
    finally:
        _scheduler_tls._in_dpor_machinery = False
        set_sync_reporter(None)
        set_context(None, None)
        uninstall_wait_for_graph()


@pytest.mark.parametrize("machinery", [False, True], ids=["normal", "in-machinery"])
def test_rlock_release_clears_holding_edge_without_reentering_scheduler(scheduler_context, machinery):
    _, graph, events = scheduler_context
    lock = CooperativeRLock()
    node = ("lock", lock._object_id)
    lock.acquire()
    assert ("thread", 0) in graph._edges.get(node, set())
    _scheduler_tls._in_dpor_machinery = machinery
    lock.release()
    # Inspect before fixture teardown clears the graph, avoiding a vacuous pass.
    assert ("thread", 0) not in graph._edges.get(node, set())
    assert ("lock_release" in events) is not machinery
    assert not lock._is_owned()


def test_rlock_final_release_during_machinery(scheduler_context):
    lock = CooperativeRLock()
    lock.acquire()
    lock.acquire()
    lock.release()
    assert lock._is_owned()
    _scheduler_tls._in_dpor_machinery = True
    lock.release()
    assert not lock._is_owned()


def test_contested_rlock_in_machinery_uses_native_wait(scheduler_context):
    scheduler, _, _ = scheduler_context
    lock = CooperativeRLock()
    ready = threading.Event()
    release = threading.Event()

    def holder():
        with lock._lock:
            ready.set()
            release.wait(timeout=5.0)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert ready.wait(timeout=5.0)
        _scheduler_tls._in_dpor_machinery = True
        scheduler.wait_for_turn.reset_mock()  # Ignore cooperative Event setup above.
        assert not lock.acquire(timeout=0.1)
        scheduler.wait_for_turn.assert_not_called()
    finally:
        release.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()


@pytest.mark.parametrize("primitive", [CooperativeLock, CooperativeRLock, CooperativeSemaphore])
def test_lock_inside_sync_reporter_does_not_recurse(scheduler_context, primitive):
    acquired = []

    def reporter(event, obj_id, lock_obj):
        inner = CooperativeLock()
        acquired.append(inner.acquire(timeout=1.0))
        if acquired[-1]:
            inner.release()

    set_sync_reporter(reporter)
    with primitive():
        pass
    assert acquired and all(acquired)


def test_lock_in_machinery_bypasses_scheduler(scheduler_context):
    scheduler, _, events = scheduler_context
    _scheduler_tls._in_dpor_machinery = True
    lock = CooperativeLock()
    assert lock.acquire(timeout=1.0)
    lock.release()
    scheduler.wait_for_turn.assert_not_called()
    assert events == []


def test_wait_for_graph_allows_reentrant_operations():
    graph = WaitForGraph()
    with graph._lock:
        # A bounded nested acquire detects a non-reentrant lock without hanging.
        assert graph._lock.acquire(timeout=0.1)
        try:
            graph.add_waiting(0, 100)
            graph.add_holding(1, 200)
            graph.remove_waiting(0, 100)
            graph.remove_holding(1, 200)
            assert graph._edges == {}
        finally:
            graph._lock.release()
