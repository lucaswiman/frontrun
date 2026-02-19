"""
Bytecode-tracing DPOR (Dynamic Partial Order Reduction) for frontrun.

This module implements systematic interleaving exploration using DPOR,
completely separate from the existing bytecode.py random exploration.

The approach:
1. A Rust DPOR engine (frontrun_dpor) manages the exploration tree,
   vector clocks, and backtrack set computation.
2. Python drives execution: runs threads under sys.settrace opcode
   tracing, uses a shadow stack to detect shared-memory accesses,
   and feeds access/sync events to the Rust engine.
3. Cooperative threading primitives (lock, event, etc.) are monkey-patched
   to yield control back to the DPOR scheduler and report synchronization
   events for happens-before tracking.

Usage::

    from frontrun.dpor import explore_dpor

    class Counter:
        def __init__(self):
            self.value = 0
        def increment(self):
            temp = self.value
            self.value = temp + 1

    result = explore_dpor(
        setup=lambda: Counter(),
        threads=[lambda c: c.increment(), lambda c: c.increment()],
        invariant=lambda c: c.value == 2,
    )
    assert not result.property_holds  # lost-update bug found
"""

from __future__ import annotations

import dis
import os
import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from frontrun_dpor import PyDporEngine, PyExecution

T = TypeVar("T")

_PY_VERSION = sys.version_info[:2]
# sys.monitoring (PEP 669) is available since 3.12 and is required for
# free-threaded builds (3.13t/3.14t) where sys.settrace + f_trace_opcodes
# has a known crash bug (CPython #118415).
_USE_SYS_MONITORING = _PY_VERSION >= (3, 12)
_IS_FREE_THREADED = getattr(sys, "_is_gil_enabled", lambda: True)() is False

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class DporResult:
    """Result of DPOR exploration."""

    property_holds: bool
    executions_explored: int = 0
    counterexample_schedule: list[int] | None = None
    failures: list[tuple[int, list[int]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shadow Stack for shared-access detection
# ---------------------------------------------------------------------------


class ShadowStack:
    """Mirrors CPython's evaluation stack to track object identity.

    When LOAD_ATTR/STORE_ATTR execute, we peek at our shadow stack
    to identify which object is being accessed.
    """

    __slots__ = ("stack",)

    def __init__(self) -> None:
        self.stack: list[Any] = []

    def push(self, val: Any) -> None:
        self.stack.append(val)

    def pop(self) -> Any:
        return self.stack.pop() if self.stack else None

    def peek(self, n: int = 0) -> Any:
        idx = -(n + 1)
        return self.stack[idx] if abs(idx) <= len(self.stack) else None

    def clear(self) -> None:
        self.stack.clear()


# Pre-analyzed instruction cache: code_id -> {offset -> instruction}
_INSTR_CACHE: dict[int, dict[int, dis.Instruction]] = {}
# Lock created with threading.Lock() directly (before any patching happens)
_INSTR_CACHE_LOCK = threading.Lock()


def _get_instructions(code: Any) -> dict[int, dis.Instruction]:
    """Get a mapping from byte offset to Instruction for a code object."""
    code_id = id(code)
    # Fast path: already cached (safe to read without lock on GIL builds;
    # on free-threaded builds dict reads are internally locked)
    cached = _INSTR_CACHE.get(code_id)
    if cached is not None:
        return cached
    with _INSTR_CACHE_LOCK:
        # Double-check after acquiring lock
        if code_id in _INSTR_CACHE:
            return _INSTR_CACHE[code_id]
        mapping = {}
        for instr in dis.get_instructions(code, show_caches=False):
            mapping[instr.offset] = instr
        _INSTR_CACHE[code_id] = mapping
        return mapping


# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------

_SKIP_DIRS: set[str] = set()
for _p in sys.path:
    if "lib/python" in _p or "site-packages" in _p:
        _SKIP_DIRS.add(_p)
_THREADING_FILE = threading.__file__
_THIS_FILE = os.path.abspath(__file__)


def _should_trace_file(filename: str) -> bool:
    if filename == _THREADING_FILE or filename == _THIS_FILE:
        return False
    if filename.startswith("<"):
        return False
    for skip_dir in _SKIP_DIRS:
        if filename.startswith(skip_dir):
            return False
    return True


# ---------------------------------------------------------------------------
# Thread-local state for the DPOR scheduler
# ---------------------------------------------------------------------------

_dpor_tls = threading.local()

# Save real factories before any patching
_real_lock = threading.Lock
_real_rlock = threading.RLock
_real_event = threading.Event
_real_condition = threading.Condition
_real_semaphore = threading.Semaphore
_real_bounded_semaphore = threading.BoundedSemaphore
_real_queue = queue.Queue
_real_lifo_queue = queue.LifoQueue
_real_priority_queue = queue.PriorityQueue


# ---------------------------------------------------------------------------
# DPOR Cooperative Primitives
# ---------------------------------------------------------------------------


class _DporCooperativeLock:
    """Lock that yields to the DPOR scheduler and reports sync events."""

    def __init__(self) -> None:
        self._lock = _real_lock()
        self._object_id = id(self)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not blocking:
            result = self._lock.acquire(blocking=False)
            if result:
                self._report_acquire()
            return result

        if self._lock.acquire(blocking=False):
            self._report_acquire()
            return True

        ctx = _get_dpor_context()
        if ctx is None:
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._report_acquire()
            return result

        scheduler, thread_id = ctx
        while not self._lock.acquire(blocking=False):
            if scheduler._finished or scheduler._error:
                result = self._lock.acquire(blocking=blocking, timeout=1.0)
                if result:
                    self._report_acquire()
                return result
            scheduler.wait_for_turn(thread_id)

        self._report_acquire()
        return True

    def release(self) -> None:
        self._report_release()
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> _DporCooperativeLock:
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def _report_acquire(self) -> None:
        engine = getattr(_dpor_tls, "engine", None)
        execution = getattr(_dpor_tls, "execution", None)
        thread_id = getattr(_dpor_tls, "thread_id", None)
        if engine is not None and execution is not None and thread_id is not None:
            engine.report_sync(execution, thread_id, "lock_acquire", self._object_id)

    def _report_release(self) -> None:
        engine = getattr(_dpor_tls, "engine", None)
        execution = getattr(_dpor_tls, "execution", None)
        thread_id = getattr(_dpor_tls, "thread_id", None)
        if engine is not None and execution is not None and thread_id is not None:
            engine.report_sync(execution, thread_id, "lock_release", self._object_id)


class _DporCooperativeRLock:
    """Reentrant lock that yields to the DPOR scheduler."""

    def __init__(self) -> None:
        self._lock = _real_lock()
        self._owner: int | None = None
        self._count = 0
        self._object_id = id(self)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return True

        if not blocking:
            if self._lock.acquire(blocking=False):
                self._owner = me
                self._count = 1
                self._report_acquire()
                return True
            return False

        if self._lock.acquire(blocking=False):
            self._owner = me
            self._count = 1
            self._report_acquire()
            return True

        ctx = _get_dpor_context()
        if ctx is None:
            result = self._lock.acquire(blocking=blocking, timeout=timeout)
            if result:
                self._owner = me
                self._count = 1
                self._report_acquire()
            return result

        scheduler, thread_id = ctx
        while not self._lock.acquire(blocking=False):
            if scheduler._finished or scheduler._error:
                result = self._lock.acquire(blocking=blocking, timeout=1.0)
                if result:
                    self._owner = me
                    self._count = 1
                    self._report_acquire()
                return result
            scheduler.wait_for_turn(thread_id)

        self._owner = me
        self._count = 1
        self._report_acquire()
        return True

    def release(self) -> None:
        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._report_release()
            self._lock.release()

    def __enter__(self) -> _DporCooperativeRLock:
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def _is_owned(self) -> bool:
        return self._owner == threading.get_ident()

    def _report_acquire(self) -> None:
        engine = getattr(_dpor_tls, "engine", None)
        execution = getattr(_dpor_tls, "execution", None)
        thread_id = getattr(_dpor_tls, "thread_id", None)
        if engine is not None and execution is not None and thread_id is not None:
            engine.report_sync(execution, thread_id, "lock_acquire", self._object_id)

    def _report_release(self) -> None:
        engine = getattr(_dpor_tls, "engine", None)
        execution = getattr(_dpor_tls, "execution", None)
        thread_id = getattr(_dpor_tls, "thread_id", None)
        if engine is not None and execution is not None and thread_id is not None:
            engine.report_sync(execution, thread_id, "lock_release", self._object_id)


class _DporCooperativeEvent:
    """Event that yields to the DPOR scheduler."""

    def __init__(self) -> None:
        self._event = _real_event()

    def wait(self, timeout: float | None = None) -> bool:
        if self._event.is_set():
            return True

        ctx = _get_dpor_context()
        if ctx is None:
            return self._event.wait(timeout=timeout)

        scheduler, thread_id = ctx
        if timeout is not None:
            import time

            deadline = time.monotonic() + timeout
            while not self._event.is_set():
                if scheduler._finished or scheduler._error:
                    return self._event.wait(timeout=1.0)
                if time.monotonic() >= deadline:
                    return self._event.is_set()
                scheduler.wait_for_turn(thread_id)
            return True

        while not self._event.is_set():
            if scheduler._finished or scheduler._error:
                return self._event.wait(timeout=1.0)
            scheduler.wait_for_turn(thread_id)
        return True

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()


class _DporCooperativeCondition:
    """Condition that yields to the DPOR scheduler."""

    def __init__(self, lock: _DporCooperativeLock | None = None) -> None:
        if lock is None:
            lock = _DporCooperativeLock()
        self._lock = lock
        self._real_cond = _real_condition(_real_lock())
        self._waiters = 0

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> _DporCooperativeCondition:
        self._lock.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self._lock.release()

    def wait(self, timeout: float | None = None) -> bool:
        self._waiters += 1
        self._lock.release()
        try:
            ctx = _get_dpor_context()
            if ctx is None:
                with self._real_cond:
                    return self._real_cond.wait(timeout=timeout)

            scheduler, thread_id = ctx
            with self._real_cond:
                notified = self._real_cond.wait(timeout=0)
            while not notified:
                if scheduler._finished or scheduler._error:
                    with self._real_cond:
                        return self._real_cond.wait(timeout=1.0)
                scheduler.wait_for_turn(thread_id)
                with self._real_cond:
                    notified = self._real_cond.wait(timeout=0)
            return True
        finally:
            self._waiters -= 1
            self._lock.acquire()

    def wait_for(self, predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        result = predicate()
        while not result:
            self.wait(timeout=timeout)
            result = predicate()
            if timeout is not None:
                break
        return result

    def notify(self, n: int = 1) -> None:
        with self._real_cond:
            self._real_cond.notify(n)

    def notify_all(self) -> None:
        with self._real_cond:
            self._real_cond.notify_all()


class _DporCooperativeSemaphore:
    """Semaphore that yields to the DPOR scheduler."""

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._value = value
        self._lock = _real_lock()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        self._lock.acquire()
        if self._value > 0:
            self._value -= 1
            self._lock.release()
            return True
        self._lock.release()

        if not blocking:
            return False

        ctx = _get_dpor_context()
        if ctx is None:
            while True:
                self._lock.acquire()
                if self._value > 0:
                    self._value -= 1
                    self._lock.release()
                    return True
                self._lock.release()
                import time

                time.sleep(0.001)

        scheduler, thread_id = ctx
        while True:
            self._lock.acquire()
            if self._value > 0:
                self._value -= 1
                self._lock.release()
                return True
            self._lock.release()
            if scheduler._finished or scheduler._error:
                import time

                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    self._lock.acquire()
                    if self._value > 0:
                        self._value -= 1
                        self._lock.release()
                        return True
                    self._lock.release()
                    time.sleep(0.001)
                return False
            scheduler.wait_for_turn(thread_id)

    def release(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        self._lock.acquire()
        self._value += n
        self._lock.release()

    def __enter__(self) -> _DporCooperativeSemaphore:
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class _DporCooperativeBoundedSemaphore(_DporCooperativeSemaphore):
    """BoundedSemaphore that yields to the DPOR scheduler."""

    def __init__(self, value: int = 1) -> None:
        super().__init__(value)
        self._initial_value = value

    def release(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        self._lock.acquire()
        if self._value + n > self._initial_value:
            self._lock.release()
            raise ValueError("Semaphore released too many times")
        self._value += n
        self._lock.release()


class _DporCooperativeQueue:
    """Queue that yields to the DPOR scheduler."""

    _queue_class = _real_queue

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: Any = self._queue_class(maxsize)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        try:
            return self._queue.get(block=False)
        except queue.Empty:
            if not block:
                raise

        ctx = _get_dpor_context()
        if ctx is None:
            return self._queue.get(block=True, timeout=timeout)

        scheduler, thread_id = ctx
        while True:
            try:
                return self._queue.get(block=False)
            except queue.Empty:
                pass
            if scheduler._finished or scheduler._error:
                return self._queue.get(block=True, timeout=1.0)
            scheduler.wait_for_turn(thread_id)

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        try:
            self._queue.put(item, block=False)
            return
        except queue.Full:
            if not block:
                raise

        ctx = _get_dpor_context()
        if ctx is None:
            self._queue.put(item, block=True, timeout=timeout)
            return

        scheduler, thread_id = ctx
        while True:
            try:
                self._queue.put(item, block=False)
                return
            except queue.Full:
                pass
            if scheduler._finished or scheduler._error:
                self._queue.put(item, block=True, timeout=1.0)
                return
            scheduler.wait_for_turn(thread_id)

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def get_nowait(self) -> Any:
        return self._queue.get(block=False)

    def put_nowait(self, item: Any) -> None:
        self._queue.put(item, block=False)

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()


class _DporCooperativeLifoQueue(_DporCooperativeQueue):
    _queue_class = _real_lifo_queue


class _DporCooperativePriorityQueue(_DporCooperativeQueue):
    _queue_class = _real_priority_queue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dpor_context() -> tuple[DporScheduler, int] | None:
    """Get the DPOR scheduler context from thread-local storage."""
    scheduler = getattr(_dpor_tls, "scheduler", None)
    thread_id = getattr(_dpor_tls, "thread_id", None)
    if scheduler is not None and thread_id is not None:
        return scheduler, thread_id
    return None


# ---------------------------------------------------------------------------
# DPOR Opcode Scheduler
# ---------------------------------------------------------------------------


class DporScheduler:
    """Controls thread execution at opcode granularity, driven by the DPOR engine.

    Unlike the random OpcodeScheduler in bytecode.py, this scheduler gets
    its scheduling decisions from the Rust DPOR engine.
    """

    def __init__(self, engine: PyDporEngine, execution: PyExecution, num_threads: int) -> None:
        self.engine = engine
        self.execution = execution
        self.num_threads = num_threads
        self._lock = _real_lock()
        self._condition = threading.Condition(self._lock)
        self._finished = False
        self._error: Exception | None = None
        self._threads_done: set[int] = set()
        self._current_thread: int | None = None

        # Shadow stacks are per-thread (each thread only accesses its own),
        # stored in thread-local storage. This avoids cross-thread access
        # entirely, which is critical for free-threaded builds.
        # Format: _dpor_tls._shadow_stacks = {frame_id: ShadowStack}

        # Request the first scheduling decision
        self._current_thread = self._schedule_next()

    def _schedule_next(self) -> int | None:
        """Ask the DPOR engine which thread to run next."""
        # Mark finished threads
        for tid in list(self._threads_done):
            self.execution.finish_thread(tid)

        runnable = self.execution.runnable_threads()
        if not runnable:
            return None

        return self.engine.schedule(self.execution)

    def wait_for_turn(self, thread_id: int) -> bool:
        """Block until it's this thread's turn. Returns False when done."""
        with self._condition:
            while True:
                if self._finished or self._error:
                    return False
                if self._current_thread == thread_id:
                    # It's our turn. After executing one opcode, schedule next.
                    next_thread = self._schedule_next()
                    self._current_thread = next_thread
                    if next_thread is None:
                        self._finished = True
                    self._condition.notify_all()
                    return True
                # Wait for our turn
                if not self._condition.wait(timeout=5.0):
                    if self._current_thread in self._threads_done:
                        # Current thread is done, try scheduling again
                        next_thread = self._schedule_next()
                        self._current_thread = next_thread
                        if next_thread is None:
                            self._finished = True
                        self._condition.notify_all()
                        continue
                    self._error = TimeoutError(
                        f"DPOR deadlock: waiting for thread {thread_id}, "
                        f"current is {self._current_thread}"
                    )
                    self._condition.notify_all()
                    return False

    def mark_done(self, thread_id: int) -> None:
        with self._condition:
            self._threads_done.add(thread_id)
            self.execution.finish_thread(thread_id)
            # If the done thread was the current one, schedule next
            if self._current_thread == thread_id:
                next_thread = self._schedule_next()
                self._current_thread = next_thread
                if next_thread is None and len(self._threads_done) >= self.num_threads:
                    self._finished = True
            self._condition.notify_all()

    def report_error(self, error: Exception) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    @staticmethod
    def get_shadow_stack(frame_id: int) -> ShadowStack:
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks is None:
            stacks = {}
            _dpor_tls._shadow_stacks = stacks
        if frame_id not in stacks:
            stacks[frame_id] = ShadowStack()
        return stacks[frame_id]

    @staticmethod
    def remove_shadow_stack(frame_id: int) -> None:
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks is not None:
            stacks.pop(frame_id, None)


# ---------------------------------------------------------------------------
# Opcode trace callback with shadow stack access detection
# ---------------------------------------------------------------------------


def _make_object_key(obj_id: int, name: Any) -> int:
    """Create a non-negative u64 object key for the Rust engine."""
    return hash((obj_id, name)) & 0xFFFFFFFFFFFFFFFF


def _report_read(engine: PyDporEngine, execution: PyExecution, thread_id: int, obj: Any, name: Any) -> None:
    if obj is not None:
        engine.report_access(execution, thread_id, _make_object_key(id(obj), name), "read")


def _report_write(engine: PyDporEngine, execution: PyExecution, thread_id: int, obj: Any, name: Any) -> None:
    if obj is not None:
        engine.report_access(execution, thread_id, _make_object_key(id(obj), name), "write")


def _process_opcode(
    frame: Any,
    scheduler: DporScheduler,
    thread_id: int,
) -> None:
    """Process a single opcode, updating the shadow stack and reporting accesses.

    Handles opcodes across Python 3.10-3.14, including:
    - 3.13: LOAD_FAST_LOAD_FAST, STORE_FAST_STORE_FAST
    - 3.14: LOAD_FAST_BORROW, LOAD_SMALL_INT, BINARY_SUBSCR removal
    """
    code = frame.f_code
    instrs = _get_instructions(code)
    instr = instrs.get(frame.f_lasti)
    if instr is None:
        return

    shadow = scheduler.get_shadow_stack(id(frame))
    op = instr.opname
    engine = scheduler.engine
    execution = scheduler.execution

    # === LOAD instructions: push values onto the shadow stack ===

    if op in ("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"):
        # LOAD_FAST_BORROW is new in 3.14: same semantics as LOAD_FAST
        # but uses a borrowed reference internally.
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    elif op == "LOAD_FAST_LOAD_FAST":
        # New in 3.13: pushes two locals in one instruction.
        # argval is a tuple of two variable names.
        argval = instr.argval
        if isinstance(argval, tuple) and len(argval) == 2:
            shadow.push(frame.f_locals.get(argval[0]))
            shadow.push(frame.f_locals.get(argval[1]))
        else:
            shadow.push(None)
            shadow.push(None)

    elif op == "LOAD_GLOBAL":
        val = frame.f_globals.get(instr.argval)
        shadow.push(val)

    elif op == "LOAD_DEREF":
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    elif op in ("LOAD_CONST", "LOAD_CONST_IMMORTAL", "LOAD_CONST_MORTAL"):
        shadow.push(instr.argval)

    elif op == "LOAD_SMALL_INT":
        # New in 3.14: pushes a small integer (the oparg itself).
        shadow.push(instr.arg)

    # === Stack manipulation ===

    elif op == "COPY":
        n = instr.arg
        if n is not None and len(shadow.stack) >= n:
            shadow.push(shadow.stack[-n])
        else:
            shadow.push(None)

    elif op == "SWAP":
        n = instr.arg
        if n is not None and len(shadow.stack) >= n:
            shadow.stack[-1], shadow.stack[-n] = shadow.stack[-n], shadow.stack[-1]

    # === Attribute access: the instructions we care about most ===

    elif op == "LOAD_ATTR":
        obj = shadow.pop()
        attr = instr.argval
        _report_read(engine, execution, thread_id, obj, attr)
        if obj is not None:
            try:
                shadow.push(getattr(obj, attr))
            except Exception:
                shadow.push(None)
        else:
            shadow.push(None)

    elif op == "STORE_ATTR":
        obj = shadow.pop()   # TOS = object
        _val = shadow.pop()  # TOS1 = value
        _report_write(engine, execution, thread_id, obj, instr.argval)

    elif op == "DELETE_ATTR":
        obj = shadow.pop()
        _report_write(engine, execution, thread_id, obj, instr.argval)

    # === Subscript access (dict/list operations) ===

    elif op == "BINARY_SUBSCR":
        # Present on 3.10-3.13. Removed in 3.14 (replaced by BINARY_OP
        # with subscript oparg).
        key = shadow.pop()
        container = shadow.pop()
        _report_read(engine, execution, thread_id, container, repr(key))
        shadow.push(None)

    elif op == "STORE_SUBSCR":
        key = shadow.pop()
        container = shadow.pop()
        _val = shadow.pop()
        _report_write(engine, execution, thread_id, container, repr(key))

    elif op == "DELETE_SUBSCR":
        key = shadow.pop()
        container = shadow.pop()
        _report_write(engine, execution, thread_id, container, repr(key))

    # === Arithmetic and binary operations ===

    elif op == "BINARY_OP":
        # On 3.14, BINARY_OP also handles subscript operations (replacing
        # the removed BINARY_SUBSCR). Check the argrepr for "[]" / subscript.
        argrepr = instr.argrepr
        if argrepr and ("[" in argrepr or "NB_SUBSCR" in argrepr.upper()):
            key = shadow.pop()
            container = shadow.pop()
            _report_read(engine, execution, thread_id, container, repr(key))
            shadow.push(None)
        else:
            shadow.pop()
            shadow.pop()
            shadow.push(None)

    # === Store instructions ===

    elif op in ("STORE_FAST", "STORE_GLOBAL", "STORE_DEREF"):
        shadow.pop()

    elif op == "STORE_FAST_STORE_FAST":
        # New in 3.13: pops two values.
        shadow.pop()
        shadow.pop()

    # === Return/pop ===

    elif op in ("RETURN_VALUE", "RETURN_CONST"):
        shadow.pop()

    elif op == "POP_TOP":
        shadow.pop()

    else:
        # Fallback: use dis.stack_effect for unknown opcodes.
        # This handles CALL, PUSH_NULL, RESUME, PRECALL, and any
        # version-specific opcodes we don't explicitly handle.
        try:
            effect = dis.stack_effect(instr.opcode, instr.arg or 0)
            for _ in range(max(0, -effect)):
                shadow.pop()
            for _ in range(max(0, effect)):
                shadow.push(None)
        except (ValueError, TypeError):
            shadow.clear()


# ---------------------------------------------------------------------------
# DPOR Bytecode Runner
# ---------------------------------------------------------------------------


class DporBytecodeRunner:
    """Runs threads under DPOR-controlled bytecode-level interleaving.

    Uses sys.monitoring (PEP 669) on Python 3.12+ for thread-safe opcode
    instrumentation. Falls back to sys.settrace on 3.10-3.11.
    """

    # sys.monitoring tool ID for DPOR (use PROFILER to avoid conflict with debuggers)
    _TOOL_ID: int | None = None

    def __init__(self, scheduler: DporScheduler, cooperative_locks: bool = True) -> None:
        self.scheduler = scheduler
        self.cooperative_locks = cooperative_locks
        self.threads: list[threading.Thread] = []
        self.errors: dict[int, Exception] = {}
        self._lock_patched = False
        self._monitoring_active = False

    def _patch_locks(self) -> None:
        if not self.cooperative_locks:
            return
        threading.Lock = _DporCooperativeLock  # type: ignore[assignment]
        threading.RLock = _DporCooperativeRLock  # type: ignore[assignment]
        threading.Semaphore = _DporCooperativeSemaphore  # type: ignore[assignment]
        threading.BoundedSemaphore = _DporCooperativeBoundedSemaphore  # type: ignore[assignment]
        threading.Event = _DporCooperativeEvent  # type: ignore[assignment]
        threading.Condition = _DporCooperativeCondition  # type: ignore[assignment]
        queue.Queue = _DporCooperativeQueue  # type: ignore[assignment]
        queue.LifoQueue = _DporCooperativeLifoQueue  # type: ignore[assignment]
        queue.PriorityQueue = _DporCooperativePriorityQueue  # type: ignore[assignment]
        self._lock_patched = True

    def _unpatch_locks(self) -> None:
        if self._lock_patched:
            threading.Lock = _real_lock  # type: ignore[assignment]
            threading.RLock = _real_rlock  # type: ignore[assignment]
            threading.Semaphore = _real_semaphore  # type: ignore[assignment]
            threading.BoundedSemaphore = _real_bounded_semaphore  # type: ignore[assignment]
            threading.Event = _real_event  # type: ignore[assignment]
            threading.Condition = _real_condition  # type: ignore[assignment]
            queue.Queue = _real_queue  # type: ignore[assignment]
            queue.LifoQueue = _real_lifo_queue  # type: ignore[assignment]
            queue.PriorityQueue = _real_priority_queue  # type: ignore[assignment]
            self._lock_patched = False

    # --- sys.settrace backend (3.10-3.11) ---

    def _make_trace(self, thread_id: int) -> Callable[..., Any]:
        scheduler = self.scheduler

        def trace(frame: Any, event: str, arg: Any) -> Any:
            if scheduler._finished or scheduler._error:
                return None

            if event == "call":
                if _should_trace_file(frame.f_code.co_filename):
                    frame.f_trace_opcodes = True
                    return trace
                return None

            if event == "opcode":
                _process_opcode(frame, scheduler, thread_id)
                scheduler.wait_for_turn(thread_id)
                return trace

            if event == "return":
                scheduler.remove_shadow_stack(id(frame))
                return trace

            return trace

        return trace

    # --- sys.monitoring backend (3.12+) ---

    def _setup_monitoring(self) -> None:
        """Set up sys.monitoring INSTRUCTION events for all code objects."""
        if not _USE_SYS_MONITORING:
            return

        mon = sys.monitoring
        tool_id = mon.PROFILER_ID  # type: ignore[attr-defined]
        DporBytecodeRunner._TOOL_ID = tool_id

        mon.use_tool_id(tool_id, "frontrun-dpor")  # type: ignore[attr-defined]
        mon.set_events(tool_id, mon.events.PY_START | mon.events.PY_RETURN | mon.events.INSTRUCTION)  # type: ignore[attr-defined]

        scheduler = self.scheduler

        def handle_py_start(code: Any, instruction_offset: int) -> Any:
            if scheduler._finished or scheduler._error:
                return mon.DISABLE  # type: ignore[attr-defined]
            if not _should_trace_file(code.co_filename):
                return mon.DISABLE  # type: ignore[attr-defined]
            return None

        def handle_py_return(code: Any, instruction_offset: int, retval: Any) -> Any:
            return None

        def handle_instruction(code: Any, instruction_offset: int) -> Any:
            if scheduler._finished or scheduler._error:
                return None
            if not _should_trace_file(code.co_filename):
                return None

            thread_id = getattr(_dpor_tls, "thread_id", None)
            if thread_id is None:
                return None

            # Build a minimal frame-like object for _process_opcode.
            # sys.monitoring gives us code + offset; we need f_locals/f_globals.
            # Use sys._getframe() to get the actual frame.
            frame = sys._getframe(1)
            _process_opcode(frame, scheduler, thread_id)
            scheduler.wait_for_turn(thread_id)
            return None

        mon.register_callback(tool_id, mon.events.PY_START, handle_py_start)  # type: ignore[attr-defined]
        mon.register_callback(tool_id, mon.events.PY_RETURN, handle_py_return)  # type: ignore[attr-defined]
        mon.register_callback(tool_id, mon.events.INSTRUCTION, handle_instruction)  # type: ignore[attr-defined]
        self._monitoring_active = True

    def _teardown_monitoring(self) -> None:
        if not self._monitoring_active:
            return
        mon = sys.monitoring
        tool_id = DporBytecodeRunner._TOOL_ID
        if tool_id is not None:
            mon.set_events(tool_id, 0)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_START, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_RETURN, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.INSTRUCTION, None)  # type: ignore[attr-defined]
            mon.free_tool_id(tool_id)  # type: ignore[attr-defined]
        self._monitoring_active = False

    # --- Thread entry points ---

    def _run_thread_settrace(
        self,
        thread_id: int,
        func: Callable[..., None],
        args: tuple[Any, ...],
    ) -> None:
        """Thread entry using sys.settrace (3.10-3.11)."""
        try:
            _dpor_tls.scheduler = self.scheduler
            _dpor_tls.thread_id = thread_id
            _dpor_tls.engine = self.scheduler.engine
            _dpor_tls.execution = self.scheduler.execution

            trace_fn = self._make_trace(thread_id)
            sys.settrace(trace_fn)
            func(*args)
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            sys.settrace(None)
            _dpor_tls.scheduler = None
            _dpor_tls.thread_id = None
            _dpor_tls.engine = None
            _dpor_tls.execution = None
            self.scheduler.mark_done(thread_id)

    def _run_thread_monitoring(
        self,
        thread_id: int,
        func: Callable[..., None],
        args: tuple[Any, ...],
    ) -> None:
        """Thread entry using sys.monitoring (3.12+)."""
        try:
            _dpor_tls.scheduler = self.scheduler
            _dpor_tls.thread_id = thread_id
            _dpor_tls.engine = self.scheduler.engine
            _dpor_tls.execution = self.scheduler.execution

            func(*args)
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            _dpor_tls.scheduler = None
            _dpor_tls.thread_id = None
            _dpor_tls.engine = None
            _dpor_tls.execution = None
            self.scheduler.mark_done(thread_id)

    def run(
        self,
        funcs: list[Callable[..., None]],
        args: list[tuple[Any, ...]] | None = None,
        timeout: float = 10.0,
    ) -> None:
        if args is None:
            args = [() for _ in funcs]

        use_monitoring = _USE_SYS_MONITORING
        if use_monitoring:
            self._setup_monitoring()
            run_thread = self._run_thread_monitoring
        else:
            run_thread = self._run_thread_settrace

        try:
            for i, (func, a) in enumerate(zip(funcs, args)):
                t = threading.Thread(
                    target=run_thread,
                    args=(i, func, a),
                    name=f"dpor-{i}",
                    daemon=True,
                )
                self.threads.append(t)

            for t in self.threads:
                t.start()

            for t in self.threads:
                t.join(timeout=timeout)
        finally:
            if use_monitoring:
                self._teardown_monitoring()

        if self.errors:
            first_error = next(iter(self.errors.values()))
            if not isinstance(first_error, TimeoutError):
                raise first_error


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def explore_dpor(
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    timeout_per_run: float = 5.0,
    cooperative_locks: bool = True,
) -> DporResult:
    """Systematically explore interleavings using DPOR.

    This is the DPOR replacement for ``explore_interleavings()``. Instead of
    random sampling, it uses the DPOR algorithm to explore only distinct
    interleavings (modulo independent operation reordering).

    Args:
        setup: Creates fresh shared state for each execution.
        threads: List of callables, each receiving the shared state.
        invariant: Predicate over shared state; must be True after all
            threads complete.
        max_executions: Safety limit on total executions (None = unlimited).
        preemption_bound: Limit on preemptions per execution. 2 catches most
            bugs. None = unbounded (full DPOR).
        max_branches: Maximum scheduling points per execution.
        timeout_per_run: Timeout per execution in seconds.
        cooperative_locks: Replace threading/queue primitives with
            scheduler-aware versions.

    Returns:
        DporResult with exploration statistics and any counterexample found.
    """
    num_threads = len(threads)
    pb = None if preemption_bound is None else preemption_bound
    me = None if max_executions is None else max_executions
    engine = PyDporEngine(
        num_threads=num_threads,
        preemption_bound=pb,
        max_branches=max_branches,
        max_executions=me,
    )

    result = DporResult(property_holds=True)

    while True:
        execution = engine.begin_execution()
        scheduler = DporScheduler(engine, execution, num_threads)
        runner = DporBytecodeRunner(scheduler, cooperative_locks=cooperative_locks)

        runner._patch_locks()
        try:
            state = setup()

            def make_thread_func(thread_func: Callable[[T], None], s: T) -> Callable[[], None]:
                def wrapper() -> None:
                    thread_func(s)
                return wrapper

            funcs = [make_thread_func(t, state) for t in threads]
            try:
                runner.run(funcs, timeout=timeout_per_run)
            except TimeoutError:
                pass
        finally:
            runner._unpatch_locks()

        result.executions_explored += 1

        if not invariant(state):
            result.property_holds = False
            schedule = execution.schedule_trace
            result.failures.append((result.executions_explored, list(schedule)))
            if result.counterexample_schedule is None:
                result.counterexample_schedule = list(schedule)

        # Clear instruction cache between executions to avoid stale code ids
        with _INSTR_CACHE_LOCK:
            _INSTR_CACHE.clear()

        if not engine.next_execution():
            break

    return result
