"""
Pure-Python DPOR (Dynamic Partial Order Reduction) prototype for frontrun.

This implements the classic DPOR algorithm (Flanagan & Godefroid, POPL 2005)
with optional preemption bounding. It is a faithful Python translation of the
Rust prototype in src/, designed for experimentation and integration testing.

The module provides two levels of API:

1. Low-level engine API (DporEngine, Execution) for fine-grained control.
2. High-level model-checking API (explore_dpor) for common patterns.

Usage example::

    from dpor_prototype import explore_dpor, AccessKind, Step

    result = explore_dpor(
        setup=lambda: {"counter": 0, "local": [0, 0]},
        thread_steps=[
            [  # Thread 0: read-modify-write
                Step(object_id=0, kind=AccessKind.READ,
                     apply=lambda s: s.__setitem__("local", [s["counter"], s["local"][1]])),
                Step(object_id=0, kind=AccessKind.WRITE,
                     apply=lambda s: s.__setitem__("counter", s["local"][0] + 1)),
            ],
            [  # Thread 1: read-modify-write
                Step(object_id=0, kind=AccessKind.READ,
                     apply=lambda s: s.__setitem__("local", [s["local"][0], s["counter"]])),
                Step(object_id=0, kind=AccessKind.WRITE,
                     apply=lambda s: s.__setitem__("counter", s["local"][1] + 1)),
            ],
        ],
        invariant=lambda s: s["counter"] == 2,
    )
    assert not result.all_passed  # Lost-update bug detected!
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Vector Clock
# ---------------------------------------------------------------------------

class VersionVec:
    """Vector clock for happens-before tracking."""

    __slots__ = ("_clocks",)

    def __init__(self, num_threads: int) -> None:
        self._clocks = [0] * num_threads

    def __repr__(self) -> str:
        return f"VersionVec({list(self._clocks)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionVec):
            return NotImplemented
        return self._clocks == other._clocks

    def clone(self) -> VersionVec:
        vv = VersionVec(0)
        vv._clocks = list(self._clocks)
        return vv

    @property
    def size(self) -> int:
        return len(self._clocks)

    def get(self, thread_id: int) -> int:
        return self._clocks[thread_id]

    def set(self, thread_id: int, value: int) -> None:
        self._clocks[thread_id] = value

    def increment(self, thread_id: int) -> None:
        self._clocks[thread_id] += 1

    def join(self, other: VersionVec) -> None:
        """Point-wise maximum: self = max(self, other)."""
        if len(other._clocks) > len(self._clocks):
            self._clocks.extend([0] * (len(other._clocks) - len(self._clocks)))
        for i in range(len(other._clocks)):
            self._clocks[i] = max(self._clocks[i], other._clocks[i])

    def partial_le(self, other: VersionVec) -> bool:
        """Returns True if self <= other (component-wise)."""
        max_len = max(len(self._clocks), len(other._clocks))
        for i in range(max_len):
            a = self._clocks[i] if i < len(self._clocks) else 0
            b = other._clocks[i] if i < len(other._clocks) else 0
            if a > b:
                return False
        return True

    def concurrent_with(self, other: VersionVec) -> bool:
        """Returns True if neither self <= other nor other <= self."""
        return not self.partial_le(other) and not other.partial_le(self)


# ---------------------------------------------------------------------------
# Access Tracking
# ---------------------------------------------------------------------------

class AccessKind(Enum):
    READ = auto()
    WRITE = auto()


@dataclass
class Access:
    """Records an access to a shared object for DPOR dependency detection."""
    path_id: int
    dpor_vv: VersionVec
    thread_id: int

    def happens_before(self, later_vv: VersionVec) -> bool:
        return self.dpor_vv.partial_le(later_vv)


@dataclass
class ObjectState:
    """Tracks the last accesses to a shared object for DPOR."""
    last_access: Optional[Access] = None
    last_write_access: Optional[Access] = None

    def last_dependent_access(self, kind: AccessKind) -> Optional[Access]:
        """
        Returns the last dependent access for the given kind.
        - Read depends on last Write (reads are independent of each other).
        - Write depends on any prior access.
        """
        if kind == AccessKind.READ:
            return self.last_write_access
        else:
            return self.last_access

    def record_access(self, access: Access, kind: AccessKind) -> None:
        if kind == AccessKind.WRITE:
            self.last_write_access = access
        # Clone for last_access to avoid aliasing
        self.last_access = Access(
            path_id=access.path_id,
            dpor_vv=access.dpor_vv.clone(),
            thread_id=access.thread_id,
        )


# ---------------------------------------------------------------------------
# Thread State
# ---------------------------------------------------------------------------

class ThreadStatus(Enum):
    DISABLED = auto()
    PENDING = auto()
    BACKTRACK = auto()
    YIELD = auto()
    ACTIVE = auto()
    VISITED = auto()
    BLOCKED = auto()

    def is_runnable(self) -> bool:
        return self in (
            ThreadStatus.PENDING,
            ThreadStatus.BACKTRACK,
            ThreadStatus.YIELD,
            ThreadStatus.ACTIVE,
        )


@dataclass
class Thread:
    """State of a single thread within one execution."""
    id: int
    causality: VersionVec
    dpor_vv: VersionVec
    finished: bool = False
    blocked: bool = False

    @staticmethod
    def new(thread_id: int, num_threads: int) -> Thread:
        return Thread(
            id=thread_id,
            causality=VersionVec(num_threads),
            dpor_vv=VersionVec(num_threads),
        )

    def is_runnable(self) -> bool:
        return not self.finished and not self.blocked


# ---------------------------------------------------------------------------
# Exploration Tree (Path)
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    """A single branch point in the exploration tree."""
    threads: list[ThreadStatus]
    active_thread: int
    preemptions: int = 0


class Path:
    """The exploration tree: manages DFS over scheduling decisions."""

    def __init__(self, preemption_bound: Optional[int] = None) -> None:
        self._branches: list[Branch] = []
        self._pos: int = 0
        self._preemption_bound = preemption_bound

    @property
    def depth(self) -> int:
        return len(self._branches)

    @property
    def current_position(self) -> int:
        return self._pos

    @property
    def branches(self) -> list[Branch]:
        return self._branches

    def schedule(
        self,
        runnable: list[int],
        current_thread: int,
        num_threads: int,
    ) -> Optional[int]:
        """Pick which thread to run at the current scheduling point."""
        if not runnable:
            return None

        if self._pos < len(self._branches):
            # Replaying: follow the recorded path
            chosen = self._branches[self._pos].active_thread
            self._pos += 1
            return chosen

        # New branch: prefer current thread to minimize preemptions
        if current_thread in runnable:
            chosen = current_thread
        else:
            chosen = runnable[0]

        # Compute preemption count
        is_preemption = chosen != current_thread and current_thread in runnable
        prev_preemptions = self._branches[-1].preemptions if self._branches else 0
        preemptions = prev_preemptions + (1 if is_preemption else 0)

        threads = [ThreadStatus.DISABLED] * num_threads
        for tid in runnable:
            threads[tid] = ThreadStatus.ACTIVE if tid == chosen else ThreadStatus.PENDING

        self._branches.append(Branch(
            threads=threads,
            active_thread=chosen,
            preemptions=preemptions,
        ))
        self._pos += 1
        return chosen

    def backtrack(self, path_id: int, thread_id: int) -> None:
        """Mark thread_id for backtracking at branch path_id."""
        if path_id >= len(self._branches):
            return

        branch = self._branches[path_id]
        if thread_id >= len(branch.threads):
            return

        status = branch.threads[thread_id]
        if status in (ThreadStatus.PENDING, ThreadStatus.YIELD):
            # Check preemption bound
            if self._preemption_bound is not None:
                if (branch.active_thread != thread_id
                        and branch.preemptions >= self._preemption_bound):
                    self._add_conservative_backtrack(path_id, thread_id, self._preemption_bound)
                    return
            branch.threads[thread_id] = ThreadStatus.BACKTRACK

    def step(self) -> bool:
        """Advance to the next unexplored execution path.

        Returns True if there's another path to explore.
        """
        while self._branches:
            branch = self._branches[-1]
            active = branch.active_thread
            if active < len(branch.threads) and branch.threads[active] == ThreadStatus.ACTIVE:
                branch.threads[active] = ThreadStatus.VISITED

            # Look for a thread marked for backtracking
            try:
                next_idx = branch.threads.index(ThreadStatus.BACKTRACK)
            except ValueError:
                next_idx = None

            if next_idx is not None:
                branch.threads[next_idx] = ThreadStatus.ACTIVE
                branch.active_thread = next_idx
                self._pos = 0  # Replay from the beginning
                return True

            # No more alternatives: pop and continue backtracking
            self._branches.pop()

        return False

    def _add_conservative_backtrack(
        self, path_id: int, thread_id: int, bound: int
    ) -> None:
        """Conservative backtrack for preemption bounding."""
        for i in range(path_id - 1, -1, -1):
            branch = self._branches[i]
            if thread_id < len(branch.threads):
                status = branch.threads[thread_id]
                would_preempt = (
                    branch.active_thread != thread_id and status.is_runnable()
                )
                if status in (ThreadStatus.PENDING, ThreadStatus.YIELD):
                    if not would_preempt or branch.preemptions < bound:
                        branch.threads[thread_id] = ThreadStatus.BACKTRACK
                        return


# ---------------------------------------------------------------------------
# DPOR Engine
# ---------------------------------------------------------------------------

@dataclass
class SyncEvent:
    """Base class for synchronization events."""
    pass


@dataclass
class LockAcquire(SyncEvent):
    lock_id: int
    release_vv: Optional[VersionVec] = None


@dataclass
class LockRelease(SyncEvent):
    lock_id: int


@dataclass
class ThreadJoin(SyncEvent):
    joined_thread: int


@dataclass
class ThreadSpawn(SyncEvent):
    child_thread: int


class Execution:
    """Per-execution state. Reset at the start of each execution."""

    def __init__(self, num_threads: int) -> None:
        self.threads = [Thread.new(i, num_threads) for i in range(num_threads)]
        self.objects: dict[int, ObjectState] = {}
        self.active_thread = 0
        self.lock_release_vv: dict[int, VersionVec] = {}
        self.aborted = False
        self.schedule_trace: list[int] = []

    def finish_thread(self, thread_id: int) -> None:
        self.threads[thread_id].finished = True

    def block_thread(self, thread_id: int) -> None:
        self.threads[thread_id].blocked = True

    def unblock_thread(self, thread_id: int) -> None:
        self.threads[thread_id].blocked = False

    def runnable_threads(self) -> list[int]:
        return [t.id for t in self.threads if t.is_runnable()]


class DporEngine:
    """The main DPOR engine."""

    def __init__(
        self,
        num_threads: int,
        preemption_bound: Optional[int] = None,
        max_branches: int = 100_000,
        max_executions: Optional[int] = None,
    ) -> None:
        self._path = Path(preemption_bound)
        self._num_threads = num_threads
        self._max_branches = max_branches
        self._max_executions = max_executions
        self._executions_completed = 0

    @property
    def executions_completed(self) -> int:
        return self._executions_completed

    @property
    def tree_depth(self) -> int:
        return self._path.depth

    def begin_execution(self) -> Execution:
        return Execution(self._num_threads)

    def schedule(self, execution: Execution) -> Optional[int]:
        """Pick which thread to run next. Returns None if deadlock."""
        runnable = execution.runnable_threads()
        if not runnable:
            execution.aborted = True
            return None

        if self._path.current_position >= self._max_branches:
            execution.aborted = True
            return None

        chosen = self._path.schedule(runnable, execution.active_thread, self._num_threads)
        if chosen is None:
            return None

        # Update DPOR vector clock
        execution.threads[chosen].dpor_vv.increment(chosen)
        execution.active_thread = chosen
        execution.schedule_trace.append(chosen)
        return chosen

    def process_access(
        self,
        execution: Execution,
        thread_id: int,
        object_id: int,
        kind: AccessKind,
    ) -> None:
        """Process a shared memory access. Core DPOR operation."""
        current_path_id = max(0, self._path.current_position - 1)
        current_dpor_vv = execution.threads[thread_id].dpor_vv.clone()

        if object_id not in execution.objects:
            execution.objects[object_id] = ObjectState()
        obj_state = execution.objects[object_id]

        prev = obj_state.last_dependent_access(kind)
        if prev is not None:
            if not prev.happens_before(current_dpor_vv):
                # Concurrent dependent accesses: insert backtrack point
                self._path.backtrack(prev.path_id, thread_id)

        access = Access(
            path_id=current_path_id,
            dpor_vv=current_dpor_vv,
            thread_id=thread_id,
        )
        obj_state.record_access(access, kind)

    def process_sync(
        self,
        execution: Execution,
        thread_id: int,
        event: SyncEvent,
    ) -> None:
        """Process a synchronization event (updates happens-before)."""
        if isinstance(event, LockAcquire):
            if event.lock_id in execution.lock_release_vv:
                release_vv = execution.lock_release_vv[event.lock_id]
                execution.threads[thread_id].causality.join(release_vv)
                execution.threads[thread_id].dpor_vv.join(release_vv)
        elif isinstance(event, LockRelease):
            vv = execution.threads[thread_id].causality.clone()
            execution.lock_release_vv[event.lock_id] = vv
        elif isinstance(event, ThreadJoin):
            joined = event.joined_thread
            execution.threads[thread_id].causality.join(
                execution.threads[joined].causality
            )
            execution.threads[thread_id].dpor_vv.join(
                execution.threads[joined].dpor_vv
            )
        elif isinstance(event, ThreadSpawn):
            child = event.child_thread
            execution.threads[child].causality.join(
                execution.threads[thread_id].causality
            )
            execution.threads[child].dpor_vv.join(
                execution.threads[thread_id].dpor_vv
            )

    def next_execution(self) -> bool:
        """Finish current execution and advance to next. Returns False when done."""
        self._executions_completed += 1
        if self._max_executions is not None:
            if self._executions_completed >= self._max_executions:
                return False
        return self._path.step()


# ---------------------------------------------------------------------------
# High-level model-checking API
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single step in a thread's execution."""
    object_id: int
    kind: AccessKind
    apply: Callable[[Any], None]


@dataclass
class ExplorationResult:
    """Result of DPOR exploration."""
    executions_explored: int = 0
    all_passed: bool = True
    failures: list[tuple[int, list[int]]] = field(default_factory=list)


def explore_dpor(
    setup: Callable[[], Any],
    thread_steps: list[list[Step]],
    invariant: Callable[[Any], bool],
    preemption_bound: Optional[int] = None,
    max_executions: Optional[int] = None,
) -> ExplorationResult:
    """
    Explore all distinct interleavings using DPOR.

    Args:
        setup: Creates fresh shared state for each execution.
        thread_steps: List of thread definitions. Each thread is a list of Steps.
        invariant: Predicate over shared state; must be True after all threads complete.
        preemption_bound: Max preemptions per execution (None = unbounded).
        max_executions: Safety limit on total executions.

    Returns:
        ExplorationResult with statistics and any failures found.
    """
    num_threads = len(thread_steps)
    engine = DporEngine(
        num_threads=num_threads,
        preemption_bound=preemption_bound,
        max_executions=max_executions,
    )
    result = ExplorationResult()

    while True:
        execution = engine.begin_execution()
        state = setup()
        thread_pcs = [0] * num_threads

        while True:
            # Mark finished threads
            for i in range(num_threads):
                if thread_pcs[i] >= len(thread_steps[i]):
                    execution.finish_thread(i)

            if not execution.runnable_threads():
                break

            chosen = engine.schedule(execution)
            if chosen is None:
                break

            pc = thread_pcs[chosen]
            if pc >= len(thread_steps[chosen]):
                break

            step = thread_steps[chosen][pc]

            # Report access to DPOR engine
            engine.process_access(execution, chosen, step.object_id, step.kind)

            # Mutate shared state
            step.apply(state)

            thread_pcs[chosen] += 1

        exec_num = engine.executions_completed + 1

        if not invariant(state):
            result.all_passed = False
            result.failures.append((exec_num, list(execution.schedule_trace)))

        if not engine.next_execution():
            break

    result.executions_explored = engine.executions_completed
    return result


# ---------------------------------------------------------------------------
# Cooperative primitives for integration with real Python code
# ---------------------------------------------------------------------------

class SharedVar:
    """A shared variable that reports accesses to the DPOR engine.

    This is the "Approach C" (loom-style cooperative primitives) from the
    DPOR spec. Users wrap shared state in SharedVar and the DPOR engine
    automatically tracks dependencies.

    Usage::

        balance = SharedVar(100, object_id=0, engine=engine, execution=execution)
        val = balance.read(thread_id=0)    # Reports Read access
        balance.write(val + 50, thread_id=0)  # Reports Write access
    """

    def __init__(
        self,
        initial_value: Any,
        object_id: int,
        engine: Optional[DporEngine] = None,
        execution: Optional[Execution] = None,
        thread_id: int = 0,
    ) -> None:
        self._value = initial_value
        self._object_id = object_id
        self._engine = engine
        self._execution = execution
        self._thread_id = thread_id

    def read(self, thread_id: Optional[int] = None) -> Any:
        tid = thread_id if thread_id is not None else self._thread_id
        if self._engine and self._execution:
            self._engine.process_access(
                self._execution, tid, self._object_id, AccessKind.READ
            )
        return copy.deepcopy(self._value)

    def write(self, value: Any, thread_id: Optional[int] = None) -> None:
        tid = thread_id if thread_id is not None else self._thread_id
        if self._engine and self._execution:
            self._engine.process_access(
                self._execution, tid, self._object_id, AccessKind.WRITE
            )
        self._value = value

    @property
    def value(self) -> Any:
        """Direct access (bypasses DPOR tracking). Use for invariant checks."""
        return self._value


class CooperativeLock:
    """A lock that reports acquire/release to the DPOR engine.

    Establishes happens-before edges between release and subsequent acquire.
    """

    def __init__(
        self,
        lock_id: int,
        engine: Optional[DporEngine] = None,
        execution: Optional[Execution] = None,
    ) -> None:
        self._lock_id = lock_id
        self._engine = engine
        self._execution = execution
        self._held_by: Optional[int] = None

    def acquire(self, thread_id: int) -> None:
        if self._engine and self._execution:
            self._engine.process_sync(
                self._execution,
                thread_id,
                LockAcquire(lock_id=self._lock_id),
            )
        self._held_by = thread_id

    def release(self, thread_id: int) -> None:
        if self._engine and self._execution:
            self._engine.process_sync(
                self._execution,
                thread_id,
                LockRelease(lock_id=self._lock_id),
            )
        self._held_by = None
