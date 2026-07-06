"""Tests for frontrun/_opcode_observer.py.

Covers the shadow stack, instruction cache, stable object IDs, object key
generation, shared-opcode classification, call access detection, and
_process_opcode integration with a mock scheduler.
"""

from __future__ import annotations

import builtins
import dis
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from frontrun._opcode_observer import (
    ShadowStack,
    StableObjectIds,
    _call_might_report_access,
    _get_instructions,
    _is_shared_opcode,
    _make_object_key,
    _process_opcode,
    _report_access,
    clear_instr_cache,
    process_opcode_with_coarsening,
)

_PY = sys.version_info[:2]


# ---------------------------------------------------------------------------
# 1. ShadowStack basics
# ---------------------------------------------------------------------------


class TestShadowStack:
    def test_push_and_peek(self) -> None:
        s = ShadowStack()
        s.push(42)
        assert s.peek() == 42

    def test_push_pop(self) -> None:
        s = ShadowStack()
        s.push("a")
        s.push("b")
        assert s.pop() == "b"
        assert s.pop() == "a"

    def test_pop_empty_returns_none(self) -> None:
        s = ShadowStack()
        assert s.pop() is None

    def test_peek_empty_returns_none(self) -> None:
        s = ShadowStack()
        assert s.peek() is None

    def test_peek_with_offset(self) -> None:
        s = ShadowStack()
        s.push("a")
        s.push("b")
        s.push("c")
        assert s.peek(0) == "c"
        assert s.peek(1) == "b"
        assert s.peek(2) == "a"

    def test_peek_offset_beyond_stack_returns_none(self) -> None:
        s = ShadowStack()
        s.push("x")
        assert s.peek(1) is None
        assert s.peek(99) is None

    def test_clear(self) -> None:
        s = ShadowStack()
        s.push(1)
        s.push(2)
        s.clear()
        assert s.pop() is None
        assert len(s.stack) == 0


# ---------------------------------------------------------------------------
# 2. _get_instructions / clear_instr_cache
# ---------------------------------------------------------------------------


def _sample_fn() -> int:
    x = 1
    return x


class TestInstructionCache:
    def test_returns_dict_of_instructions(self) -> None:
        mapping = _get_instructions(_sample_fn.__code__)
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        for offset, instr in mapping.items():
            assert isinstance(offset, int)
            assert isinstance(instr, dis.Instruction)

    def test_caching_returns_same_dict(self) -> None:
        clear_instr_cache()
        m1 = _get_instructions(_sample_fn.__code__)
        m2 = _get_instructions(_sample_fn.__code__)
        assert m1 is m2

    def test_clear_invalidates_cache(self) -> None:
        clear_instr_cache()
        m1 = _get_instructions(_sample_fn.__code__)
        clear_instr_cache()
        m2 = _get_instructions(_sample_fn.__code__)
        # After clearing, a new dict is built (not the same object).
        assert m1 is not m2
        # But the contents should be equivalent.
        assert set(m1.keys()) == set(m2.keys())


# ---------------------------------------------------------------------------
# 3. StableObjectIds
# ---------------------------------------------------------------------------


class TestStableObjectIds:
    def test_assigns_incrementing_ids(self) -> None:
        sids = StableObjectIds()
        a, b, c = object(), object(), object()
        assert sids.get(a) == 0
        assert sids.get(b) == 1
        assert sids.get(c) == 2

    def test_same_object_same_id(self) -> None:
        sids = StableObjectIds()
        obj = object()
        id1 = sids.get(obj)
        id2 = sids.get(obj)
        assert id1 == id2

    def test_reset_for_execution_clears(self) -> None:
        sids = StableObjectIds()
        obj = object()
        sids.get(obj)
        sids.reset_for_execution()
        # After reset, the next object gets ID 0 again.
        new_obj = object()
        assert sids.get(new_obj) == 0


# ---------------------------------------------------------------------------
# 4. _make_object_key
# ---------------------------------------------------------------------------


class TestMakeObjectKey:
    def test_non_negative(self) -> None:
        for i in range(100):
            key = _make_object_key(i, f"attr_{i}")
            assert key >= 0
            assert key <= 0xFFFFFFFFFFFFFFFF

    def test_deterministic(self) -> None:
        k1 = _make_object_key(12345, "value")
        k2 = _make_object_key(12345, "value")
        assert k1 == k2

    def test_different_inputs_different_keys(self) -> None:
        k1 = _make_object_key(1, "x")
        k2 = _make_object_key(2, "x")
        k3 = _make_object_key(1, "y")
        # Not strictly guaranteed by hash, but overwhelmingly likely.
        assert k1 != k2
        assert k1 != k3


# ---------------------------------------------------------------------------
# 5. _is_shared_opcode
# ---------------------------------------------------------------------------


class TestIsSharedOpcode:
    """Verify classification of opcodes as shared vs thread-local."""

    def _find_opcode(self, code: Any, opname: str) -> int | None:
        """Return the offset of the first instruction with the given opname, or None."""
        for instr in dis.get_instructions(code):
            if instr.opname == opname:
                return instr.offset
        return None

    def test_load_attr_is_shared(self) -> None:
        def fn(obj: Any) -> Any:
            return obj.x  # LOAD_ATTR

        off = self._find_opcode(fn.__code__, "LOAD_ATTR")
        assert off is not None
        assert _is_shared_opcode(fn.__code__, off) is True

    def test_store_attr_is_shared(self) -> None:
        def fn(obj: Any) -> None:
            obj.x = 1  # STORE_ATTR

        off = self._find_opcode(fn.__code__, "STORE_ATTR")
        assert off is not None
        assert _is_shared_opcode(fn.__code__, off) is True

    def test_load_global_is_shared(self) -> None:
        def fn() -> Any:
            return len  # LOAD_GLOBAL

        off = self._find_opcode(fn.__code__, "LOAD_GLOBAL")
        assert off is not None
        assert _is_shared_opcode(fn.__code__, off) is True

    def test_load_fast_is_not_shared(self) -> None:
        def fn() -> int:
            x = 1
            return x  # LOAD_FAST

        off = self._find_opcode(fn.__code__, "LOAD_FAST") or self._find_opcode(fn.__code__, "LOAD_FAST_BORROW")
        assert off is not None
        opname = _get_instructions(fn.__code__)[off].opname
        assert _is_shared_opcode(fn.__code__, off) is False, f"Expected {opname} to be non-shared"

    def test_store_fast_is_not_shared(self) -> None:
        def fn() -> None:
            x = 1  # noqa: F841  # STORE_FAST
            return

        off = self._find_opcode(fn.__code__, "STORE_FAST")
        assert off is not None
        assert _is_shared_opcode(fn.__code__, off) is False

    def test_call_opcodes_return_false(self) -> None:
        """CALL opcodes are handled separately via _call_might_report_access."""

        def fn() -> int:
            return len([])  # CALL

        off = self._find_opcode(fn.__code__, "CALL")
        if off is not None:
            assert _is_shared_opcode(fn.__code__, off) is False

    @pytest.mark.skipif(_PY < (3, 12), reason="BINARY_SUBSCR still exists before 3.12 as the primary subscript op")
    def test_binary_subscr_is_shared(self) -> None:
        def fn(d: dict[str, int]) -> int:
            return d["key"]

        off = self._find_opcode(fn.__code__, "BINARY_SUBSCR")
        if off is not None:
            assert _is_shared_opcode(fn.__code__, off) is True

    def test_invalid_offset_returns_false(self) -> None:
        assert _is_shared_opcode(_sample_fn.__code__, 999999) is False


# ---------------------------------------------------------------------------
# 6. _call_might_report_access
# ---------------------------------------------------------------------------


class TestCallMightReportAccess:
    def test_builtin_method_on_mutable_returns_true(self) -> None:
        """A bound C method like list.append should be detected."""
        s = ShadowStack()
        my_list: list[int] = [1, 2, 3]
        s.push(my_list.append)  # builtin_function_or_method on mutable list
        s.push(42)  # argument
        assert _call_might_report_access(s, argc=1) is True

    def test_all_none_returns_false(self) -> None:
        s = ShadowStack()
        s.push(None)
        s.push(None)
        assert _call_might_report_access(s, argc=1) is False

    def test_passthrough_builtin_returns_true(self) -> None:
        """len() is a passthrough builtin that reads its argument."""
        s = ShadowStack()
        s.push(len)
        s.push([1, 2])  # argument
        assert _call_might_report_access(s, argc=1) is True

    def test_setattr_returns_true(self) -> None:
        s = ShadowStack()
        s.push(builtins.setattr)
        s.push(object())
        s.push("attr")
        s.push(42)
        assert _call_might_report_access(s, argc=3) is True

    def test_immutable_method_returns_false(self) -> None:
        """Methods on immutable types (str.upper) should not trigger."""
        s = ShadowStack()
        s.push("hello".upper)
        assert _call_might_report_access(s, argc=0) is False

    def test_container_constructor_returns_true(self) -> None:
        """list(iterable) is a container constructor."""
        s = ShadowStack()
        s.push(list)
        s.push([1, 2, 3])
        assert _call_might_report_access(s, argc=1) is True

    def test_empty_stack_returns_false(self) -> None:
        s = ShadowStack()
        assert _call_might_report_access(s, argc=0) is False


# ---------------------------------------------------------------------------
# 7. _process_opcode with a mock scheduler
# ---------------------------------------------------------------------------


class _EngineRecorder:
    """Records access events from _process_opcode."""

    def __init__(self) -> None:
        self.events: list[tuple[int, str, str]] = []

    def report_access(self, _execution: object, _thread_id: int, object_key: int, kind: str) -> None:
        self.events.append((object_key, kind, "access"))

    def report_first_access(self, _execution: object, _thread_id: int, object_key: int, kind: str) -> None:
        self.events.append((object_key, kind, "first"))


class _SchedulerStub:
    """Minimal mock satisfying _process_opcode's interface."""

    def __init__(self, engine: _EngineRecorder) -> None:
        self._shadow = ShadowStack()
        self.engine = engine
        self.execution = object()
        self._engine_lock = threading.Lock()
        self.trace_recorder = None
        self._stable_ids = StableObjectIds()

    def get_shadow_stack(self, _frame_id: int) -> ShadowStack:
        return self._shadow


class _CoarseningSchedulerStub(_SchedulerStub):
    """Scheduler stub that records yielded opcodes."""

    def __init__(self, engine: _EngineRecorder) -> None:
        super().__init__(engine)
        self.yielded_opnames: list[str] = []

    def report_and_wait(self, frame: Any, _thread_id: int) -> bool:
        instr = _get_instructions(frame.f_code).get(frame.f_lasti)
        if instr is not None:
            self.yielded_opnames.append(instr.opname)
        _process_opcode(frame, self, _thread_id)
        return True


def _walk_with_coarsening(fn: Any, f_locals: dict[str, Any] | None = None) -> _CoarseningSchedulerStub:
    engine = _EngineRecorder()
    scheduler = _CoarseningSchedulerStub(engine)
    frame = SimpleNamespace(
        f_code=fn.__code__,
        f_locals=f_locals or {},
        f_globals=fn.__globals__,
        f_builtins=builtins.__dict__,
        f_lasti=0,
    )
    for instr in dis.get_instructions(fn):
        frame.f_lasti = instr.offset
        process_opcode_with_coarsening(fn.__code__, instr.offset, frame, scheduler, 1, detect_io=False)
    return scheduler


class TestIteratorCoarsening:
    def test_range_loop_iter_opcodes_do_not_yield(self) -> None:
        """Iteration over range() is thread-local and should not create DPOR turns."""

        def target() -> None:
            for _ in range(3):
                pass

        scheduler = _walk_with_coarsening(target)

        assert "GET_ITER" not in scheduler.yielded_opnames
        assert "FOR_ITER" not in scheduler.yielded_opnames

    def test_list_loop_iter_opcodes_still_yield(self) -> None:
        """Iteration over a mutable container still reports reads and yields."""

        def target(items: list[int]) -> None:
            for _ in items:
                pass

        scheduler = _walk_with_coarsening(target, {"items": [1, 2]})

        assert "GET_ITER" in scheduler.yielded_opnames
        assert "FOR_ITER" in scheduler.yielded_opnames


class TestProcessOpcode:
    """Test _process_opcode by walking bytecode of simple functions."""

    def _walk_function(self, fn: Any) -> tuple[_EngineRecorder, _SchedulerStub]:
        """Walk all opcodes of `fn` through _process_opcode and return (engine, scheduler)."""
        engine = _EngineRecorder()
        scheduler = _SchedulerStub(engine)
        frame = SimpleNamespace(
            f_code=fn.__code__,
            f_locals={},
            f_globals=fn.__globals__,
            f_builtins=builtins.__dict__,
            f_lasti=0,
        )
        for instr in dis.get_instructions(fn):
            frame.f_lasti = instr.offset
            _process_opcode(frame, scheduler, 1)
        return engine, scheduler

    def test_load_const_store_fast_pushes_and_pops(self) -> None:
        """LOAD_CONST pushes to shadow stack, STORE_FAST pops."""

        def target() -> int:
            x = 42
            return x

        engine, scheduler = self._walk_function(target)
        # Shadow stack should be roughly empty after balanced push/pop.
        # (RETURN_VALUE pops the last value, leaving the stack empty or near-empty.)
        assert len(scheduler._shadow.stack) <= 1

    def test_load_global_reports_read(self) -> None:
        """LOAD_GLOBAL should report a first_read on the globals dict."""

        def target() -> type:
            return len  # type: ignore[return-value]  # LOAD_GLOBAL

        engine, _sched = self._walk_function(target)
        # Should have at least one "first" event for the global read of "len".
        first_events = [(k, kind) for k, kind, tag in engine.events if tag == "first"]
        assert len(first_events) > 0, "Expected at least one first_read event from LOAD_GLOBAL"

    def test_store_attr_reports_write(self) -> None:
        """STORE_ATTR should report a write on the object."""

        class Box:
            value: int = 0

        shared = Box()

        def target() -> None:
            shared.value = 99

        engine = _EngineRecorder()
        scheduler = _SchedulerStub(engine)
        # `shared` is captured as a closure variable (LOAD_DEREF), so it
        # must be in f_locals for the mock frame to resolve it.
        frame = SimpleNamespace(
            f_code=target.__code__,
            f_locals={"shared": shared},
            f_globals=target.__globals__,
            f_builtins=builtins.__dict__,
            f_lasti=0,
        )
        for instr in dis.get_instructions(target):
            frame.f_lasti = instr.offset
            _process_opcode(frame, scheduler, 1)

        write_events = [(k, kind) for k, kind, tag in engine.events if kind == "write"]
        assert len(write_events) > 0, "Expected write events from STORE_ATTR"

    def test_call_replaces_callable_with_none(self) -> None:
        """After CALL, the shadow stack TOS should be None (result placeholder)."""

        def identity(x: int) -> int:
            return x

        def target() -> int:
            return identity(1)

        engine = _EngineRecorder()
        scheduler = _SchedulerStub(engine)
        frame = SimpleNamespace(
            f_code=target.__code__,
            f_locals={},
            f_globals=target.__globals__ | {"identity": identity},
            f_builtins=builtins.__dict__,
            f_lasti=0,
        )
        for instr in dis.get_instructions(target):
            frame.f_lasti = instr.offset
            _process_opcode(frame, scheduler, 1)
            if instr.opname == "CALL":
                assert scheduler._shadow.peek() is None, "CALL should leave None (result placeholder) on TOS"

    def test_load_attr_reports_read(self) -> None:
        """LOAD_ATTR should report a read on the object's attribute."""

        class Box:
            value: int = 42

        shared = Box()

        def target() -> int:
            return shared.value

        engine = _EngineRecorder()
        scheduler = _SchedulerStub(engine)
        # `shared` is captured as a closure variable (LOAD_DEREF), so it
        # must be in f_locals for the mock frame to resolve it.
        frame = SimpleNamespace(
            f_code=target.__code__,
            f_locals={"shared": shared},
            f_globals=target.__globals__,
            f_builtins=builtins.__dict__,
            f_lasti=0,
        )
        for instr in dis.get_instructions(target):
            frame.f_lasti = instr.offset
            _process_opcode(frame, scheduler, 1)

        read_events = [(k, kind) for k, kind, tag in engine.events if kind == "read"]
        assert len(read_events) > 0, "Expected read events from LOAD_ATTR"


# ---------------------------------------------------------------------------
# 8. Version-specific behavior
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_PY >= (3, 11), reason="DUP_TOP/ROT_TWO only exist on 3.10")
class TestPython310StackOps:
    def test_dup_top(self) -> None:
        """DUP_TOP should duplicate TOS on the shadow stack."""
        s = ShadowStack()
        s.push("val")
        # Simulate DUP_TOP manually.
        s.push(s.peek())
        assert s.peek(0) == "val"
        assert s.peek(1) == "val"

    def test_rot_two(self) -> None:
        """ROT_TWO should swap the top two elements."""
        s = ShadowStack()
        s.push("a")
        s.push("b")
        s.stack[-1], s.stack[-2] = s.stack[-2], s.stack[-1]
        assert s.peek(0) == "a"
        assert s.peek(1) == "b"


@pytest.mark.skipif(_PY < (3, 11), reason="COPY/SWAP opcodes added in 3.11")
class TestPython311PlusCopySwap:
    def test_copy_opcode_via_process(self) -> None:
        """COPY should duplicate the nth element onto TOS."""
        # We test the ShadowStack logic that _process_opcode uses for COPY.
        s = ShadowStack()
        s.push("a")
        s.push("b")
        s.push("c")
        # COPY 2 means copy stack[-2] to TOS.
        n = 2
        s.push(s.stack[-n])
        assert s.peek(0) == "b"
        assert len(s.stack) == 4

    def test_swap_opcode_via_process(self) -> None:
        """SWAP should exchange TOS with the nth element."""
        s = ShadowStack()
        s.push("a")
        s.push("b")
        s.push("c")
        n = 3  # SWAP 3
        s.stack[-1], s.stack[-n] = s.stack[-n], s.stack[-1]
        assert s.peek(0) == "a"
        assert s.stack[-3] == "c"


@pytest.mark.skipif(_PY < (3, 13), reason="LOAD_FAST_LOAD_FAST added in 3.13")
class TestPython313LoadFastLoadFast:
    def test_load_fast_load_fast_pushes_two(self) -> None:
        """LOAD_FAST_LOAD_FAST should push two values onto the shadow stack."""

        # Find a function that uses LOAD_FAST_LOAD_FAST.
        def fn(a: int, b: int) -> int:
            return a + b

        found = False
        for instr in dis.get_instructions(fn):
            if instr.opname in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
                found = True
                break

        if not found:
            pytest.skip("Compiler did not emit LOAD_FAST_LOAD_FAST for this function")

        engine = _EngineRecorder()
        scheduler = _SchedulerStub(engine)
        frame = SimpleNamespace(
            f_code=fn.__code__,
            f_locals={"a": 10, "b": 20},
            f_globals=fn.__globals__,
            f_builtins=builtins.__dict__,
            f_lasti=instr.offset,
        )
        _process_opcode(frame, scheduler, 1)
        # Should have pushed two values.
        assert len(scheduler._shadow.stack) == 2
        assert scheduler._shadow.stack[0] == 10
        assert scheduler._shadow.stack[1] == 20


@pytest.mark.skipif(_PY < (3, 14), reason="LOAD_SMALL_INT added in 3.14")
class TestPython314LoadSmallInt:
    def test_load_small_int(self) -> None:
        """LOAD_SMALL_INT pushes the oparg (a small integer) onto the shadow stack."""

        def fn() -> int:
            return 1

        found_instr = None
        for instr in dis.get_instructions(fn):
            if instr.opname == "LOAD_SMALL_INT":
                found_instr = instr
                break

        if found_instr is None:
            pytest.skip("Compiler did not emit LOAD_SMALL_INT")

        engine = _EngineRecorder()
        scheduler = _SchedulerStub(engine)
        frame = SimpleNamespace(
            f_code=fn.__code__,
            f_locals={},
            f_globals=fn.__globals__,
            f_builtins=builtins.__dict__,
            f_lasti=found_instr.offset,
        )
        _process_opcode(frame, scheduler, 1)
        assert scheduler._shadow.peek() == found_instr.arg


@pytest.mark.skipif(_PY < (3, 14), reason="BINARY_SUBSCR removed in 3.14")
class TestPython314NoBinarySubscr:
    def test_no_binary_subscr_emitted(self) -> None:
        """On 3.14, BINARY_SUBSCR should not appear in disassembly."""

        def fn(d: dict[str, int]) -> int:
            return d["key"]

        opcodes = {instr.opname for instr in dis.get_instructions(fn)}
        assert "BINARY_SUBSCR" not in opcodes, "BINARY_SUBSCR should be removed in 3.14"

    def test_binary_op_subscript_is_shared(self) -> None:
        """On 3.14, subscript access uses BINARY_OP with a subscript argrepr."""

        def fn(d: dict[str, int]) -> int:
            return d["key"]

        for instr in dis.get_instructions(fn):
            if instr.opname == "BINARY_OP" and instr.argrepr and "[" in instr.argrepr:
                assert _is_shared_opcode(fn.__code__, instr.offset) is True
                return
        pytest.skip("Compiler did not emit BINARY_OP with subscript argrepr")


@pytest.mark.skipif(_PY >= (3, 14), reason="BINARY_SUBSCR exists on <3.14")
class TestPrePython314BinarySubscr:
    def test_binary_subscr_exists(self) -> None:
        """Before 3.14, BINARY_SUBSCR should appear for subscript access."""

        def fn(d: dict[str, int]) -> int:
            return d["key"]

        opcodes = {instr.opname for instr in dis.get_instructions(fn)}
        assert "BINARY_SUBSCR" in opcodes, "Expected BINARY_SUBSCR before 3.14"


# ---------------------------------------------------------------------------
# 9. Access-reporting dispatch: pin which engine method each helper calls
#    and which kind string it forwards.
# ---------------------------------------------------------------------------


class _DispatchRecorder:
    """Records which engine method was invoked and the kind string passed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def report_access(self, _execution: object, _thread_id: int, object_key: int, kind: str) -> None:
        self.calls.append(("access", object_key, kind))

    def report_first_access(self, _execution: object, _thread_id: int, object_key: int, kind: str) -> None:
        self.calls.append(("first_access", object_key, kind))


class TestReportHelperDispatch:
    """Pin behavior: each of the 5 access-reporting kinds dispatches to a
    specific (engine method, kind string) pair.  This test locks in the
    contract of the consolidated ``_report_access`` helper — which used to
    exist as five parallel helpers (``_report_read``, ``_report_first_read``,
    ``_report_write``, ``_report_weak_read``, ``_report_weak_write``).
    """

    @pytest.mark.parametrize(
        ("kind", "expected_method", "expected_kind"),
        [
            ("read", "access", "read"),
            ("first_read", "first_access", "read"),
            ("write", "access", "write"),
            ("weak_read", "access", "weak_read"),
            ("weak_write", "access", "weak_write"),
        ],
    )
    def test_dispatch(self, kind: Any, expected_method: str, expected_kind: str) -> None:
        engine = _DispatchRecorder()
        execution = object()
        lock = threading.Lock()
        sids = StableObjectIds()
        obj = object()

        _report_access(engine, execution, 7, obj, "attr_name", lock, sids, kind)

        assert len(engine.calls) == 1
        method, _key, recorded_kind = engine.calls[0]
        assert method == expected_method
        assert recorded_kind == expected_kind

    @pytest.mark.parametrize(
        "kind",
        ["read", "first_read", "write", "weak_read", "weak_write"],
    )
    def test_none_object_is_noop(self, kind: Any) -> None:
        """Every kind short-circuits when obj is None."""
        engine = _DispatchRecorder()
        execution = object()
        lock = threading.Lock()
        sids = StableObjectIds()

        _report_access(engine, execution, 0, None, "name", lock, sids, kind)

        assert engine.calls == []

    @pytest.mark.parametrize(
        "kind",
        ["read", "first_read", "write", "weak_read", "weak_write"],
    )
    def test_object_key_uses_stable_id_and_name(self, kind: Any) -> None:
        """All kinds compute the key via _make_object_key(stable_id, name)."""
        engine = _DispatchRecorder()
        execution = object()
        lock = threading.Lock()
        sids = StableObjectIds()
        obj = object()

        _report_access(engine, execution, 0, obj, "some_attr", lock, sids, kind)

        expected_key = _make_object_key(sids.get(obj), "some_attr")
        assert engine.calls[0][1] == expected_key
