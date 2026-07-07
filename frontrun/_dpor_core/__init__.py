"""Pure helpers shared by sync and async DPOR drivers (no threading / asyncio)."""

from __future__ import annotations

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
from frontrun._dpor_core.failures import record_dpor_failure
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
from frontrun._dpor_core.worker import IterationCustomizer, LivenessProbe, WorkerSet, WorkerTarget

__all__ = [
    "ExplorationStep",
    "IterationCustomizer",
    "LivenessProbe",
    "NoOpLock",
    "ReplayEngine",
    "ReplayExecution",
    "RowLockRegistry",
    "WorkerSet",
    "WorkerTarget",
    "advance_replay_index",
    "apply_lock_blocked_override",
    "compute_serializable_baseline_async",
    "compute_serializable_baseline_sync",
    "dpor_exploration_iter",
    "event_wake_sync_id",
    "extend_replay_schedule",
    "format_race_failure_explanation",
    "group_schedule_runs",
    "is_reproduction_run",
    "make_deadline",
    "make_dpor_engine",
    "record_dpor_failure",
    "reset_execution_state",
    "wake_sync_id",
]
