from __future__ import annotations

import contextlib
import threading
import time

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

    def test_done_thread_anchor_does_not_livelock(self) -> None:
        """Defect #16 divergence: the next IO anchor references a finished thread.

        When a thread takes a state-dependent early return (the exact divergence
        this scheduler exists to tolerate) it finishes without doing its last
        recorded IO. The next anchor then names an already-done thread. Replay
        must skip that anchor and proceed with the remaining live thread rather
        than busy-spinning on the done thread while HOLDING the condition lock
        (a 100% CPU livelock that starves every other thread until the outer
        join timeout).
        """
        from frontrun.dpor import _IOAnchoredReplayScheduler

        io_schedule = [(0, "r"), (1, "r"), (0, "r2")]
        sched = _IOAnchoredReplayScheduler(io_schedule, num_threads=2, deadlock_timeout=0.5)

        # Thread 0 does its first IO, then diverges and finishes early (skips 'r2').
        sched.before_io(0, "r")
        sched.after_io(0, "r")
        sched.mark_done(0)
        # Thread 1 does its IO; the next anchor (0, 'r2') names finished thread 0.
        sched.before_io(1, "r")
        sched.after_io(1, "r")

        result: dict[str, bool] = {}
        t = threading.Thread(
            target=lambda: result.__setitem__("ret", sched._wait_for_turn(1)),
            daemon=True,
        )
        t.start()
        t.join(timeout=3.0)

        # Pre-fix: the thread busy-spins forever holding the condition lock.
        assert not t.is_alive(), "replay livelocked on a finished thread's IO anchor"
        assert result.get("ret") is True
        # The condition lock must be free — no busy-spin holding it.
        assert sched._lock.acquire(timeout=1.0), "condition lock still held by a busy-spinner"
        sched._lock.release()

    @pytest.mark.parametrize("io_anchored", [False, True])
    def test_sync_retry_confirms_replay_deadlock_candidate(self, io_anchored: bool) -> None:
        from frontrun._deadlock import DeadlockError
        from frontrun._dpor_runtime.scheduler import _IOAnchoredReplayScheduler, _ReplayDporScheduler
        from frontrun._virtual_clock import VirtualClock

        if io_anchored:
            scheduler = _IOAnchoredReplayScheduler(
                [(0, "dummy")], 2, deadlock_timeout=0.01, virtual_clock=VirtualClock()
            )
        else:
            scheduler = _ReplayDporScheduler([0], 2, deadlock_timeout=0.01, virtual_clock=VirtualClock())
        scheduler._lock_waiters[1] = {0, 1}
        scheduler._current_thread = 0
        scheduler._exact_deadlock_candidate_at = time.monotonic() - 1.0

        assert not scheduler.before_sync_retry(1)
        assert isinstance(scheduler._error, DeadlockError)

    def test_positional_replay_detects_exact_event_deadlock(self) -> None:
        """Replaying a schedule that ends in an Event-wait cycle must raise
        DeadlockError promptly, not spin out the op budget and die with a
        plain TimeoutError after deadlock_timeout."""
        from frontrun._deadlock import DeadlockError
        from frontrun._dpor_runtime.replay import _run_dpor_schedule
        from frontrun._virtual_clock import VirtualClock

        class State:
            def __init__(self) -> None:
                self.e1 = threading.Event()
                self.e2 = threading.Event()

        def w1(s: State) -> None:
            s.e1.wait()
            s.e2.set()

        def w2(s: State) -> None:
            s.e2.wait()
            s.e1.set()

        wall_start = time.monotonic()
        with pytest.raises(DeadlockError):
            _run_dpor_schedule(
                [0, 1],
                State,
                [w1, w2],
                timeout=5.0,
                detect_io=False,
                deadlock_timeout=2.0,
                clock="virtual",
                virtual_clock=VirtualClock(),
            )
        wall_elapsed = time.monotonic() - wall_start
        assert wall_elapsed < 1.5, f"replay deadlock detection took {wall_elapsed:.1f}s (fallback timeout burned?)"

    def test_io_anchored_replay_detects_exact_event_deadlock(self) -> None:
        """Same as above through the IO-anchored replay scheduler (defect #16 path)."""
        from frontrun._deadlock import DeadlockError
        from frontrun._dpor_runtime.replay import _run_dpor_schedule
        from frontrun._virtual_clock import VirtualClock

        class State:
            def __init__(self) -> None:
                self.e1 = threading.Event()
                self.e2 = threading.Event()

        def w1(s: State) -> None:
            s.e1.wait()
            s.e2.set()

        def w2(s: State) -> None:
            s.e2.wait()
            s.e1.set()

        wall_start = time.monotonic()
        with pytest.raises(DeadlockError):
            _run_dpor_schedule(
                [0, 1],
                State,
                [w1, w2],
                timeout=5.0,
                detect_io=True,
                deadlock_timeout=2.0,
                io_schedule=[(0, "dummy")],
                clock="virtual",
                virtual_clock=VirtualClock(),
            )
        wall_elapsed = time.monotonic() - wall_start
        assert wall_elapsed < 1.5, f"replay deadlock detection took {wall_elapsed:.1f}s (fallback timeout burned?)"

    def test_io_anchored_replay_expires_timed_wait_virtually(self) -> None:
        """A timeout-kind deadline (Event.wait(timeout=...)) must expire during
        IO-anchored replay.  The io_schedule carries no clock-actor entries, so
        the scheduler itself must advance the virtual clock when the thread it
        waits on is blocked in a timed wait."""
        from frontrun._dpor_runtime.replay import _run_dpor_schedule
        from frontrun._virtual_clock import VIRTUAL_EPOCH, VirtualClock

        class State:
            def __init__(self) -> None:
                self.event = threading.Event()
                self.elapsed = 0.0
                self.timed_out: bool | None = None

        def worker(s: State) -> None:
            start = time.monotonic()
            got = s.event.wait(timeout=1.0)  # nobody sets it; must expire virtually
            s.elapsed = time.monotonic() - start
            s.timed_out = not got

        clock = VirtualClock()
        wall_start = time.monotonic()
        state = _run_dpor_schedule(
            [0],
            State,
            [worker],
            timeout=1.0,
            detect_io=True,
            deadlock_timeout=0.1,
            io_schedule=[(0, "dummy")],
            clock="virtual",
            virtual_clock=clock,
        )
        wall_elapsed = time.monotonic() - wall_start

        assert state.timed_out is True
        assert state.elapsed >= 1.0
        assert clock.now() >= VIRTUAL_EPOCH + 1.0
        assert wall_elapsed < 0.5, f"virtual replay timed wait burned wall time ({wall_elapsed:.3f}s)"

    def test_io_anchored_replay_timed_acquire_expiry_with_holder(self) -> None:
        """A recorded lock.acquire(timeout=...) expiry must replay under the
        IO-anchored scheduler when the lock is held by an event-blocked thread.

        Exercises the two FIX pieces together: the scheduler must hand the
        turn from the blocked anchor owner to the live waiter, then advance
        the virtual clock to the waiter's timeout-kind deadline (the
        io_schedule carries no clock-actor entries to replay the expiry).
        """
        from frontrun._dpor_runtime.replay import _run_dpor_schedule
        from frontrun._virtual_clock import VIRTUAL_EPOCH, VirtualClock

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.done = threading.Event()
                self.got: bool | None = None

        def holder(s: State) -> None:
            with s.lock:
                s.done.wait()  # hold the lock until the waiter gave up

        def waiter(s: State) -> None:
            got = s.lock.acquire(timeout=1.0)
            s.got = got
            if got:
                s.lock.release()
            s.done.set()

        clock = VirtualClock()
        wall_start = time.monotonic()
        state = _run_dpor_schedule(
            [0, 1],
            State,
            [holder, waiter],
            timeout=5.0,
            detect_io=True,
            deadlock_timeout=2.0,
            io_schedule=[(0, "dummy-a"), (1, "dummy-b")],
            clock="virtual",
            virtual_clock=clock,
        )
        wall_elapsed = time.monotonic() - wall_start

        assert state.got is False, "the recorded timeout branch must replay"
        assert clock.now() >= VIRTUAL_EPOCH + 1.0
        assert wall_elapsed < 1.5, f"timed-acquire replay burned wall time ({wall_elapsed:.3f}s)"

    def test_reproduction_replays_timed_acquire_expiry_end_to_end(self) -> None:
        """End-to-end: a counterexample whose failing schedule includes a timed
        lock.acquire(timeout=...) expiry must reproduce N/N under clock="virtual"
        with detect_io=True (SQL activity in the run).

        Note: SQL cursors do not emit before_io/after_io anchors (only Redis
        does), so this pipeline reproduces via the positional replay scheduler
        and its recorded clock-actor entries; the IO-anchored twin is covered
        by the unit tests above.
        """
        import sqlite3
        import uuid

        db_uri = f"file:fix2_timed_{uuid.uuid4().hex}?mode=memory&cache=shared"
        keeper = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
        keeper.execute("CREATE TABLE hits (worker TEXT, got INTEGER)")
        keeper.commit()
        try:

            class State:
                def __init__(self) -> None:
                    self.lock = threading.Lock()
                    self.done = threading.Event()
                    self.got: bool | None = None
                    conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
                    conn.execute("DELETE FROM hits")
                    conn.commit()
                    conn.close()

            def holder(s: State) -> None:
                with s.lock:
                    conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
                    conn.execute("INSERT INTO hits VALUES ('holder', 1)")
                    conn.commit()
                    conn.close()
                    s.done.wait()  # hold the lock until the waiter gave up

            def waiter(s: State) -> None:
                got = s.lock.acquire(timeout=1.0)
                s.got = got
                conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
                conn.execute("INSERT INTO hits VALUES ('waiter', ?)", (1 if got else 0,))
                conn.commit()
                conn.close()
                if got:
                    s.lock.release()
                s.done.set()

            wall_start = time.monotonic()
            result = frontrun.explore(
                setup=State,
                workers=[holder, waiter],
                invariant=lambda s: s.got is True,
                clock="virtual",
                detect_io=True,
                deadlock_timeout=2.0,
                timeout_per_run=5.0,
                reproduce_on_failure=3,
            )
            wall_elapsed = time.monotonic() - wall_start
            assert not result.property_holds, "the holder-first interleaving must fail the invariant"
            assert result.reproduction_attempts == 3
            assert result.reproduction_successes == 3, (
                f"timed-acquire expiry reproduced {result.reproduction_successes}/{result.reproduction_attempts} "
                "under IO-anchored replay"
            )
            assert wall_elapsed < 20.0, f"reproduction took {wall_elapsed:.1f}s (deadlock_timeout burned per attempt?)"
        finally:
            keeper.close()

    def test_io_anchored_replay_advances_virtual_sleep(self) -> None:
        from frontrun._dpor_runtime.replay import _run_dpor_schedule
        from frontrun._virtual_clock import VIRTUAL_EPOCH, VirtualClock

        class State:
            def __init__(self) -> None:
                self.elapsed = 0.0

        def worker(s: State) -> None:
            start = time.monotonic()
            time.sleep(1.0)
            s.elapsed = time.monotonic() - start

        clock = VirtualClock()
        wall_start = time.monotonic()
        state = _run_dpor_schedule(
            [0],
            State,
            [worker],
            timeout=1.0,
            detect_io=True,
            deadlock_timeout=0.1,
            io_schedule=[(0, "dummy")],
            clock="virtual",
            virtual_clock=clock,
        )
        wall_elapsed = time.monotonic() - wall_start

        assert state.elapsed >= 1.0
        assert clock.now() >= VIRTUAL_EPOCH + 1.0
        assert wall_elapsed < 0.5, f"virtual replay sleep burned wall time ({wall_elapsed:.3f}s)"
