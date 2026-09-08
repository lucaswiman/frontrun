"""Patching rollback and compatibility with native threading consumers."""

import concurrent.futures
import threading
from unittest.mock import Mock

import pytest

from frontrun import _patching
from frontrun._cooperative import patch_locks, unpatch_locks


@pytest.mark.parametrize("failure", ["factory", "assignment"])
def test_failed_patch_can_be_retried(failure, monkeypatch):
    class Target:
        method = object()

    original, replacement = Target.method, object()
    originals, patches = {}, []
    factory = Mock(return_value=replacement)
    with monkeypatch.context() as scoped:
        if failure == "factory":
            factory.side_effect = RuntimeError("patch failed")
        else:
            scoped.setattr(_patching, "setattr", Mock(side_effect=RuntimeError("patch failed")), raising=False)
        with pytest.raises(RuntimeError, match="patch failed"):
            _patching.patch_method(Target, "method", originals=originals, patches=patches, make_wrapper=factory)
    assert originals == {} and patches == []
    assert Target.method is original
    factory.side_effect = None
    assert _patching.patch_method(Target, "method", originals=originals, patches=patches, make_wrapper=factory)
    assert Target.method is replacement
    _patching.restore_patches(patches)
    assert Target.method is original


def test_patch_locks_concurrent_futures_threadpool():
    patch_locks()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(lambda: 42).result(timeout=5.0) == 42
    finally:
        unpatch_locks()


def test_patch_locks_thread_start_event_handshake():
    started, finished = threading.Event(), threading.Event()

    def worker():
        started.set()
        finished.wait(1.0)

    patch_locks()
    try:
        thread = threading.Thread(target=worker)
        try:
            thread.start()
            assert started.wait(1.0)
        finally:
            finished.set()
            if thread.ident is not None:
                thread.join(timeout=1.0)
        assert not thread.is_alive()
    finally:
        unpatch_locks()
