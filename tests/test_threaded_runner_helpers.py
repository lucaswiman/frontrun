"""Finding 9b/9e: PatchScope.close robustness and scheduler error preservation."""

from __future__ import annotations

import time
from types import ModuleType

import pytest
from frontrun._dpor import PyDporEngine

import frontrun._dpor_runtime.runner as dpor_runner_module
import frontrun.bytecode as bytecode_module
from frontrun._dpor_runtime.runner import DporBytecodeRunner
from frontrun._real_threading import condition as _real_condition
from frontrun._real_threading import event as _real_event
from frontrun._real_threading import lock as _real_lock
from frontrun._threaded_runner import PatchScope, notify_scheduler_timeout
from frontrun.bytecode import BytecodeShuffler, OpcodeScheduler
from frontrun.dpor import DporScheduler


@pytest.mark.parametrize("strategy", ["random", "dpor"])
def test_runner_timeout_is_one_deadline_for_all_threads(strategy):
    if strategy == "random":
        scheduler = OpcodeScheduler([0] * 10, num_threads=3)
        runner = BytecodeShuffler(scheduler)
    else:
        engine = PyDporEngine(num_threads=3)
        scheduler = DporScheduler(engine, engine.begin_execution(), num_threads=3)
        runner = DporBytecodeRunner(scheduler)
    release = _real_event()

    def hang():
        release.wait(100)

    started = time.monotonic()
    try:
        runner.run([hang, hang, hang], timeout=1.0)
        elapsed = time.monotonic() - started
        assert len(runner.threads) == 3
        assert isinstance(scheduler._error, TimeoutError)
        assert elapsed < 2.0, f"per-thread deadlines accumulated: {elapsed:.1f}s"
    finally:
        release.set()
        for thread in runner.threads:
            thread.join(timeout=2.0)


def test_patch_scope_runs_all_cleanups_even_if_one_raises():
    """One raising unpatch must not skip the remaining LIFO cleanups (9e).

    Otherwise a failure tearing down one patch leaves threading primitives
    patched process-wide.
    """
    calls: list[str] = []
    scope = PatchScope()

    scope.add(lambda: None, lambda: calls.append("first"))

    def _raises() -> None:
        calls.append("second")
        raise RuntimeError("boom")

    scope.add(lambda: None, _raises)
    scope.add(lambda: None, lambda: calls.append("third"))

    with pytest.raises(RuntimeError, match="boom"):
        scope.close()

    # LIFO order: third, second (raises), first — all must run.
    assert calls == ["third", "second", "first"], calls


class _FakeScheduler:
    def __init__(self) -> None:
        self._error: Exception | None = None
        self._lock = _real_lock()
        self._condition = _real_condition(self._lock)


def test_notify_scheduler_timeout_preserves_first_error():
    """A pre-existing DeadlockError must not be clobbered by the timeout (9b)."""

    class DeadlockError(Exception):
        pass

    sched = _FakeScheduler()
    original = DeadlockError("real cause of the hang")
    sched._error = original

    notify_scheduler_timeout(sched, [])

    assert sched._error is original, f"notify_scheduler_timeout overwrote the first error with {sched._error!r}"


_COMMON_PATCHES = [
    ("install_wait_for_graph", "uninstall_wait_for_graph", "graph"),
    ("patch_locks", "unpatch_locks", "locks"),
    ("patch_io", "unpatch_io", "io"),
    ("patch_sql", "unpatch_sql", "sql"),
]


@pytest.mark.parametrize(
    ("module", "runner_type", "patches"),
    [
        (bytecode_module, BytecodeShuffler, [*_COMMON_PATCHES, ("_patch_sleep", "unpatch_sleep", "sleep")]),
        (
            dpor_runner_module,
            DporBytecodeRunner,
            [*_COMMON_PATCHES, ("patch_redis", "unpatch_redis", "redis"), ("_patch_sleep", "unpatch_sleep", "sleep")],
        ),
    ],
)
def test_runner_patch_scope_rolls_back_each_install_boundary(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    runner_type: type[BytecodeShuffler] | type[DporBytecodeRunner],
    patches: list[tuple[str, str, str]],
) -> None:
    for fail_at in range(len(patches)):
        calls: list[str] = []
        with monkeypatch.context() as patcher:
            for index, (patch_name, unpatch_name, label) in enumerate(patches):

                def install(*, _index: int = index, _label: str = label) -> None:
                    calls.append(f"+{_label}")
                    if _index == fail_at:
                        raise RuntimeError(f"failed {_label}")

                effective_patch_name = patch_name if hasattr(module, patch_name) else "patch_sleep"
                patcher.setattr(module, effective_patch_name, install)
                patcher.setattr(module, unpatch_name, lambda _label=label: calls.append(f"-{_label}"))

            scope = runner_type(_FakeScheduler()).patch_scope()
            assert calls == []
            with pytest.raises(RuntimeError, match="failed"):
                with scope:
                    pass

        installed = [label for _, _, label in patches[: fail_at + 1]]
        cleaned = [label for _, _, label in reversed(patches[:fail_at])]
        assert calls == [*(f"+{label}" for label in installed), *(f"-{label}" for label in cleaned)]


@pytest.mark.parametrize(
    ("module", "runner_type"),
    [(bytecode_module, BytecodeShuffler), (dpor_runner_module, DporBytecodeRunner)],
)
def test_runner_patch_scope_honors_disabled_io_and_sleep(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    runner_type: type[BytecodeShuffler] | type[DporBytecodeRunner],
) -> None:
    calls: list[str] = []
    patch_names = [
        *_COMMON_PATCHES,
        ("patch_redis", "unpatch_redis", "redis"),
        ("_patch_sleep", "unpatch_sleep", "sleep"),
    ]
    for patch_name, unpatch_name, label in patch_names:
        if hasattr(module, patch_name):
            monkeypatch.setattr(module, patch_name, lambda _label=label: calls.append(f"+{_label}"))
            monkeypatch.setattr(module, unpatch_name, lambda _label=label: calls.append(f"-{_label}"))

    with runner_type(_FakeScheduler(), detect_io=False).patch_scope(patch_sleep=False):
        assert calls == ["+graph", "+locks"]
    assert calls == ["+graph", "+locks", "-locks", "-graph"]
