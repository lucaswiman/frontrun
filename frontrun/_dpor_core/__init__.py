"""Pure helpers shared by sync and async DPOR drivers (no threading / asyncio)."""

from __future__ import annotations

from frontrun._dpor_core.clock_actor import (
    advance_and_dispatch,
    can_autojump,
    report_clock_sleep_wake,
    retire_actor_if_done,
    sync_clock_actor,
)
from frontrun._dpor_core.clock_port import VirtualClockPort, noop_on_wake
from frontrun._dpor_core.concurrency import (
    ExplorationStep,
    NoOpLock,
    ReplayEngine,
    ReplayExecution,
    dpor_exploration_iter,
    event_wake_sync_id,
    wake_sync_id,
)
from frontrun._dpor_core.engine import make_dpor_engine
from frontrun._dpor_core.failures import format_exact_deadlock_desc, record_dpor_failure
from frontrun._dpor_core.invariants import (
    compute_serializable_baseline_async,
    compute_serializable_baseline_sync,
    format_race_failure_explanation,
)
from frontrun._dpor_core.row_locks import RowLockRegistry
from frontrun._dpor_core.scheduling import apply_lock_blocked_override
from frontrun._dpor_core.utils import (
    advance_replay_index,
    extend_replay_schedule,
    group_schedule_runs,
    is_reproduction_run,
    make_deadline,
    reset_execution_state,
)
from frontrun._dpor_core.worker import IterationCustomizer, LivenessProbe, TerminableWorkerSet, WorkerSet, WorkerTarget

__all__ = [
    "ExplorationStep",
    "IterationCustomizer",
    "LivenessProbe",
    "NoOpLock",
    "ReplayEngine",
    "ReplayExecution",
    "RowLockRegistry",
    "TerminableWorkerSet",
    "VirtualClockPort",
    "WorkerSet",
    "WorkerTarget",
    "advance_and_dispatch",
    "advance_replay_index",
    "apply_lock_blocked_override",
    "can_autojump",
    "compute_serializable_baseline_async",
    "compute_serializable_baseline_sync",
    "dpor_exploration_iter",
    "event_wake_sync_id",
    "extend_replay_schedule",
    "format_exact_deadlock_desc",
    "format_race_failure_explanation",
    "group_schedule_runs",
    "is_reproduction_run",
    "make_deadline",
    "make_dpor_engine",
    "noop_on_wake",
    "record_dpor_failure",
    "report_clock_sleep_wake",
    "reset_execution_state",
    "retire_actor_if_done",
    "sync_clock_actor",
    "wake_sync_id",
]
