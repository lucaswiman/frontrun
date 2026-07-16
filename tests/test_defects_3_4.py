"""Tests for defects #3 and #4.

Defect #3: CooperativeQueue missing __class_getitem__ breaks psycopg v3.
Defect #4: DPOR opcode tracer IndexError on shadow stack underflow.
"""

from __future__ import annotations

import builtins
import dis
import threading
from types import SimpleNamespace
from typing import Generic, TypeVar

import frontrun
from frontrun._cooperative import CooperativeQueue  # noqa: I001

# ---------------------------------------------------------------------------
# Defect #3: CooperativeQueue.__class_getitem__
# ---------------------------------------------------------------------------

T = TypeVar("T")


class TypedQueue(CooperativeQueue[int]):  # type: ignore[type-arg]
    """Subclass using generic subscript — this is what psycopg v3 does."""


class GenericQueueUser(Generic[T]):
    """Uses CooperativeQueue[T] as a type annotation."""

    def __init__(self) -> None:
        self.q: CooperativeQueue[T] = CooperativeQueue()  # type: ignore[type-arg]


def test_class_getitem_returns_cls() -> None:
    """CooperativeQueue[int] should return CooperativeQueue itself."""
    result = CooperativeQueue[int]  # type: ignore[type-arg]
    assert result is CooperativeQueue


def test_class_getitem_subscript_subclass() -> None:
    """Subclassing CooperativeQueue[T] should not raise TypeError."""
    q = TypedQueue()
    q.put(42)
    assert q.get() == 42


# ---------------------------------------------------------------------------
# Defect #4: Shadow stack underflow in _process_opcode CALL handling
# ---------------------------------------------------------------------------


class _EngineRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[int, str, str]] = []

    def report_access(self, _execution: object, _thread_id: int, object_key: int, kind: str) -> None:
        self.events.append((object_key, kind, "access"))

    def report_first_access(self, _execution: object, _thread_id: int, object_key: int, kind: str) -> None:
        self.events.append((object_key, kind, "first"))


class _SchedulerStub:
    def __init__(self, engine: _EngineRecorder) -> None:
        from frontrun.dpor import ShadowStack, StableObjectIds

        self._shadow = ShadowStack()
        self.engine = engine
        self.execution = object()
        self._engine_lock = threading.Lock()
        self.trace_recorder = None
        self._stable_ids = StableObjectIds()

    def get_shadow_stack(self, _frame_id: int):  # type: ignore[no-untyped-def]
        return self._shadow


def test_process_opcode_call_replaces_callable_with_result_placeholder() -> None:
    """CALL should leave a result placeholder, not the callable, on the shadow stack."""
    from frontrun.dpor import _process_opcode

    def identity(seq: list[int]) -> list[int]:
        return seq

    def sample(seq: list[int]) -> int:
        return identity(seq)[0]

    engine = _EngineRecorder()
    scheduler = _SchedulerStub(engine)
    frame = SimpleNamespace(
        f_code=sample.__code__,
        f_locals={"seq": [10]},
        f_globals=globals() | {"identity": identity},
        f_builtins=builtins.__dict__,
        f_lasti=0,
    )

    for instr in dis.get_instructions(sample):
        frame.f_lasti = instr.offset
        _process_opcode(frame, scheduler, 1)
        if instr.opname == "CALL":
            assert scheduler.get_shadow_stack(id(frame)).peek() is None


def test_process_opcode_subscript_after_call_does_not_reuse_callable() -> None:
    """Subscript access after CALL must not treat the callable itself as the result."""
    from frontrun.dpor import _make_object_key, _process_opcode

    shared = [10]

    def identity(seq: list[int]) -> list[int]:
        return seq

    def sample(seq: list[int]) -> int:
        return identity(seq)[0]

    engine = _EngineRecorder()
    scheduler = _SchedulerStub(engine)
    frame = SimpleNamespace(
        f_code=sample.__code__,
        f_locals={"seq": shared},
        f_globals=globals() | {"identity": identity},
        f_builtins=builtins.__dict__,
        f_lasti=0,
    )

    for instr in dis.get_instructions(sample):
        frame.f_lasti = instr.offset
        _process_opcode(frame, scheduler, 1)

    assert (_make_object_key(id(identity), repr(0)), "read", "access") not in engine.events
    assert (_make_object_key(id(identity), "__cmethods__"), "read", "access") not in engine.events


class SharedObj:
    """Shared mutable object for defect #4 integration tests."""

    is_active: bool = False
    value: int = 0


def test_dpor_setattr_getattr_no_crash() -> None:
    """setattr/getattr on a shared object should not crash the DPOR tracer.

    Exercises the full DPOR pipeline with setattr/getattr to verify the
    bounds-checking fix works end-to-end.
    """
    result = frontrun.explore(
        setup=SharedObj,
        workers=[
            lambda obj: setattr(obj, "is_active", True),
            lambda obj: getattr(obj, "is_active"),
        ],
        invariant=lambda obj: True,
        max_executions=100,
        preemption_bound=2,
    )

    assert result.property_holds is True


def test_dpor_getattr_with_default_no_crash() -> None:
    """getattr() with default value should not crash the DPOR tracer.

    Reproduces the pattern from django-two-factor-auth where
    getattr(settings, 'OTP_TOTP_SYNC', True) crashed.
    """
    result = frontrun.explore(
        setup=SharedObj,
        workers=[
            lambda obj: setattr(obj, "value", 1),
            lambda obj: getattr(obj, "value", None),
        ],
        invariant=lambda obj: True,
        max_executions=100,
        preemption_bound=2,
    )

    assert result.property_holds is True


def test_dpor_nested_setattr_getattr_no_crash() -> None:
    """Nested setattr/getattr with complex args should not crash.

    Uses keyword-style patterns similar to Django metaclass operations
    that triggered the original shadow stack underflow.
    """

    class Model:
        _meta: dict[str, bool] = {}

        def save(self, update_fields: list[str] | None = None) -> None:
            if update_fields:
                for field in update_fields:
                    setattr(self, field, getattr(self, field, None))
            self._meta["saved"] = True

    def setup() -> Model:
        m = Model()
        m._meta = {}
        return m

    result = frontrun.explore(
        setup=setup,
        workers=[
            lambda m: m.save(update_fields=["_meta"]),
            lambda m: m.save(update_fields=["_meta"]),
        ],
        invariant=lambda m: True,
        max_executions=100,
        preemption_bound=2,
    )

    assert result.property_holds is True
