from __future__ import annotations

import contextlib
import threading

import pytest

import frontrun


class TestReplayHarness:
    def test_reproduction_uses_dpor_runner_not_bytecode_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DPOR counterexample replay should not depend on bytecode.run_with_schedule."""
        import frontrun.bytecode as bytecode

        def _unexpected_bytecode_replay(*args: object, **kwargs: object) -> object:
            raise AssertionError("DPOR replay should not call frontrun.bytecode.run_with_schedule")

        monkeypatch.setattr(bytecode, "run_with_schedule", _unexpected_bytecode_replay)

        class State:
            def __init__(self) -> None:
                self.value = 0

        def thread_fn(state: State) -> None:
            temp = state.value
            state.value = temp + 1

        result = frontrun.explore(
            setup=State,
            workers=[thread_fn, thread_fn],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            reproduce_on_failure=3,
            max_executions=50,
            preemption_bound=2,
        )

        assert not result.property_holds, "DPOR should find the lost-update race"
        assert result.reproduction_attempts == 3
        assert result.reproduction_successes == 3, (
            f"Expected DPOR-native replay to reproduce 3/3 times, got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_replay_shadow_stack_sync_for_c_method_calls(self) -> None:
        """Replay must keep the shadow stack in sync so _call_might_report_access
        produces the same results during replay as during exploration.

        Without the shadow stack fix, LOAD_ATTR opcodes (shared) wouldn't update
        the shadow stack during replay because _ReplayDporScheduler.report_and_wait
        ignored the frame.  Then CALL opcodes for C methods (like list.append,
        list.pop) wouldn't be recognized as needing scheduling points, causing
        the schedule to desynchronize and the race to not reproduce (0/10).

        This test uses a shared list (like eth-ape's _DEFAULT_SENDERS pattern)
        where the race depends on CALL scheduling points for C methods on mutable
        objects.
        """

        class _State:
            def __init__(self) -> None:
                self.stack: list[str] = []
                self.seen = [None, None]

        @contextlib.contextmanager
        def _push_pop(state: _State, value: str):
            try:
                state.stack.append(value)
                yield
            finally:
                state.stack.pop()

        def _make_fn(idx: int):
            def fn(s: _State) -> None:
                with _push_pop(s, f"v{idx}"):
                    s.seen[idx] = s.stack[-1] if s.stack else None

            return fn

        def invariant(s: _State) -> bool:
            # Thread 0 must see its own value when inside its context
            return s.seen[0] is None or s.seen[0] == "v0"

        result = frontrun.explore(
            setup=_State,
            workers=[_make_fn(0), _make_fn(1)],
            invariant=invariant,
            detect_io=False,
            reproduce_on_failure=10,
        )

        assert not result.property_holds, "DPOR should find the shared-stack race"
        assert result.reproduction_successes == 10, (
            f"Expected 10/10 reproduction with shadow stack sync, got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )


class TestIOAnchoredReplayScheduler:
    def test_after_io_sets_finished_when_all_done(self) -> None:
        from frontrun.dpor import _IOAnchoredReplayScheduler

        sched = object.__new__(_IOAnchoredReplayScheduler)
        sched._condition = threading.Condition(threading.Lock())
        sched._finished = False
        sched._error = None
        sched._active_io_thread = 42
        sched._next_thread_after_io = None
        sched._current_thread = None
        sched._threads_done = {0, 1, 42}
        sched.num_threads = 3
        sched._io_trace = []

        sched.after_io(42, "redis://cmd")

        assert sched._finished is True
