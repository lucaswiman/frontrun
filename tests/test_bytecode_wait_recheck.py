"""Finding 9a: re-check terminal/bounds state after a wait() timeout.

After ``condition.wait(timeout=...)`` returns False, another thread may have
consumed the remaining schedule entries and finished, leaving ``self._index``
at (or past) the end.  Indexing ``schedule[self._index]`` then raises
IndexError instead of cleanly returning / looping.
"""

from __future__ import annotations

from frontrun._marker_coordination import ThreadCoordinator
from frontrun.bytecode import OpcodeScheduler
from frontrun.common import Schedule, Step


def test_opcode_scheduler_wait_timeout_past_end_no_indexerror():
    # Force the not-our-turn path: schedule wants thread 0, ask as thread 1.
    sched = OpcodeScheduler([0, 0], num_threads=2)
    sched.deadlock_timeout = 0.01

    def _fake_wait(timeout=None):
        # Simulate another thread consuming the schedule and finishing the run
        # while we were blocked in wait().
        sched._index = len(sched.schedule)
        sched._finished = True
        return False

    sched._condition.wait = _fake_wait  # type: ignore[method-assign]
    # Should return False cleanly, not raise IndexError.
    assert sched.wait_for_turn(1) is False


def test_thread_coordinator_wait_timeout_past_end_no_indexerror():
    schedule = Schedule([Step("t1", "m"), Step("t2", "m")])
    coord = ThreadCoordinator(schedule, deadlock_timeout=0.01)

    def _fake_wait(timeout=None):
        # Another thread completed the schedule while we waited.
        coord.current_step = len(schedule.steps)
        coord.completed = True
        return False

    coord.condition.wait = _fake_wait  # type: ignore[method-assign]

    # t2 asks for its turn at step 0 (expects t1) -> not our turn -> wait()
    # returns False with current_step already at the end.  Must not IndexError.
    coord.wait_for_turn("t2", "m")
    assert coord.error is None
