"""Tests for time.sleep and asyncio.sleep patching.

Verifies that:
- time.sleep is replaced with a no-op during exploration
- asyncio.sleep is replaced with a no-op (but still yields) during async exploration
- Sleep patching can be disabled via patch_sleep=False
- Sleep calls act as scheduling points (threads can be preempted at sleep)
"""

from __future__ import annotations

import asyncio
import time

from frontrun import explore
from frontrun._cooperative import patch_sleep, unpatch_sleep


class TestSleepPatching:
    """Test the low-level patch_sleep / unpatch_sleep API."""

    def test_patch_replaces_time_sleep(self) -> None:
        original = time.sleep
        patch_sleep()
        try:
            assert time.sleep is not original
        finally:
            unpatch_sleep()
        assert time.sleep is original

    def test_patched_sleep_does_not_delay(self) -> None:
        patch_sleep()
        try:
            start = time.monotonic()
            time.sleep(10)  # Would take 10s if not patched
            elapsed = time.monotonic() - start
            assert elapsed < 1.0, f"Patched sleep took {elapsed:.2f}s — should be instant"
        finally:
            unpatch_sleep()

    def test_unpatch_restores_original(self) -> None:
        original = time.sleep
        patch_sleep()
        unpatch_sleep()
        assert time.sleep is original

    def test_patch_is_refcounted(self) -> None:
        """Multiple patch/unpatch calls are reference-counted like patch_locks."""
        original = time.sleep
        patch_sleep()
        patch_sleep()
        unpatch_sleep()
        # Still patched (one outstanding)
        assert time.sleep is not original
        unpatch_sleep()
        # Now restored
        assert time.sleep is original


class TestSleepInDpor:
    """Test sleep patching integrated with explore(strategy="dpor")."""

    def test_sleep_in_thread_does_not_delay(self) -> None:
        """time.sleep inside explored threads should not actually sleep."""
        import threading

        class State:
            def __init__(self) -> None:
                self.value = 0
                self.lock = threading.Lock()

        def thread_with_sleep(state: State) -> None:
            with state.lock:
                state.value += 1
            time.sleep(10)  # Would be very slow if not patched
            with state.lock:
                state.value += 1

        start = time.monotonic()
        result = explore(
            setup=State,
            workers=[thread_with_sleep, thread_with_sleep],
            invariant=lambda s: s.value == 4,
            max_executions=5,
            strategy="dpor",
        )
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"Exploration took {elapsed:.2f}s — sleep was not patched"
        assert result.property_holds

    def test_sleep_is_scheduling_point(self) -> None:
        """Sleep should be a point where the scheduler can switch threads."""

        class State:
            value: int = 0

        def thread_a(state: State) -> None:
            temp = state.value
            time.sleep(0.1)  # Scheduling point — other thread can run here
            state.value = temp + 1

        result = explore(
            setup=State,
            workers=[thread_a, thread_a],
            invariant=lambda s: s.value == 2,
            strategy="dpor",
        )
        # The classic lost-update bug: both threads read 0, then both write 1.
        # DPOR should find this because sleep is a scheduling point.
        assert not result.property_holds, "DPOR should find lost-update bug across sleep"

    def test_patch_sleep_false_disables_patching(self) -> None:
        """With patch_sleep=False, time.sleep should not be patched."""

        class State:
            slept: bool = False

        def thread_func(state: State) -> None:
            start = time.monotonic()
            time.sleep(0.05)  # Should actually sleep
            elapsed = time.monotonic() - start
            if elapsed >= 0.01:
                state.slept = True

        result = explore(
            setup=State,
            workers=[thread_func],
            invariant=lambda s: s.slept,
            max_executions=1,
            patch_sleep=False,
            strategy="dpor",
        )
        assert result.property_holds, "With patch_sleep=False, sleep should actually execute"


class TestAsyncSleepPatching:
    """Test asyncio.sleep patching in async DPOR."""

    def test_async_sleep_does_not_delay(self) -> None:
        """asyncio.sleep inside explored tasks should not actually sleep."""

        class State:
            value: int = 0

        async def task_with_sleep(state: State) -> None:
            state.value += 1
            await asyncio.sleep(10)  # Would be very slow if not patched
            state.value += 1

        async def run() -> None:
            start = time.monotonic()
            result = await explore(
                setup=State,
                workers=[task_with_sleep, task_with_sleep],
                invariant=lambda s: s.value == 4,
                max_executions=5,
                strategy="dpor",
            )
            elapsed = time.monotonic() - start
            assert elapsed < 10.0, f"Async exploration took {elapsed:.2f}s — sleep not patched"
            assert result.property_holds

        asyncio.run(run())
