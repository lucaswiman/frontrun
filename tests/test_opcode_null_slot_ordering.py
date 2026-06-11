"""Finding 6: shadow-stack NULL-slot ordering for LOAD_GLOBAL / LOAD_ATTR.

The value/NULL convention for the extra slot pushed by LOAD_GLOBAL (NULL flag)
and LOAD_ATTR (method flag) flipped in CPython 3.13 for BOTH opcodes.  The
shadow-stack emulation gated LOAD_GLOBAL at >= (3, 14) and LOAD_ATTR at
>= (3, 12), so 3.13 LOAD_GLOBAL and 3.12 LOAD_ATTR ended up with the wrong
ordering.

These tests derive the *expected* ordering from the running interpreter's
``dis`` argrepr (e.g. "len + NULL" means value below / NULL on top) and assert
the shadow stack matches — so they stay correct on whichever interpreter runs
them, and catch a mis-gated version.
"""

from __future__ import annotations

import builtins
import dis
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from frontrun import _opcode_observer as obs
from frontrun._opcode_observer import (
    ShadowStack,
    StableObjectIds,
    _process_opcode,
)


class _Engine:
    def report_access(self, *a: Any, **k: Any) -> None:
        pass

    def report_first_access(self, *a: Any, **k: Any) -> None:
        pass


class _Sched:
    def __init__(self) -> None:
        self._shadow = ShadowStack()
        self.engine = _Engine()
        self.execution = object()
        self._engine_lock = threading.Lock()
        self.trace_recorder = None
        self._stable_ids = StableObjectIds()

    def get_shadow_stack(self, _fid: int) -> ShadowStack:
        return self._shadow


def _null_on_top_from_argrepr(argrepr: str) -> bool:
    """Parse a dis argrepr like 'len + NULL' / 'NULL + len' / 'm + NULL|self'.

    The argrepr lists stack slots bottom-to-top separated by ' + '.  Returns
    True if the NULL slot is the topmost (pushed last), False if the NULL slot
    is below the value.
    """
    parts = [p.strip() for p in argrepr.split("+")]
    assert len(parts) == 2, f"unexpected argrepr {argrepr!r}"
    # The slot containing 'NULL' or 'self' is the synthetic one.
    top = parts[1]
    return "NULL" in top or "self" in top


def _walk_until(fn: Any, opname: str, f_locals: dict[str, Any]) -> tuple[_Sched, Any]:
    sched = _Sched()
    frame = SimpleNamespace(
        f_code=fn.__code__,
        f_locals=f_locals,
        f_globals=fn.__globals__,
        f_builtins=builtins.__dict__,
        f_lasti=0,
    )
    target_instr = None
    for instr in dis.get_instructions(fn):
        frame.f_lasti = instr.offset
        _process_opcode(frame, sched, 1)
        if instr.opname == opname and instr.arg is not None and instr.arg & 1:
            target_instr = instr
            break
    return sched, target_instr


def test_load_global_null_slot_ordering_matches_dis():
    def target() -> int:
        return len([1])  # LOAD_GLOBAL len (+ NULL)

    sched, instr = _walk_until(target, "LOAD_GLOBAL", {})
    if instr is None:
        pytest.skip("interpreter does not emit LOAD_GLOBAL with NULL flag here")

    null_on_top = _null_on_top_from_argrepr(instr.argrepr)
    top, below = sched._shadow.stack[-1], sched._shadow.stack[-2]
    if null_on_top:
        assert top is None and below is not None, (
            f"LOAD_GLOBAL: dis says NULL on top ({instr.argrepr!r}) but shadow stack top={top!r} below={below!r}"
        )
    else:
        assert below is None and top is not None, (
            f"LOAD_GLOBAL: dis says NULL below ({instr.argrepr!r}) but shadow stack top={top!r} below={below!r}"
        )


def test_load_attr_method_null_slot_ordering_matches_dis():
    class C:
        def m(self) -> int:
            return 1

    obj = C()

    def target(c: C) -> int:
        return c.m()  # LOAD_ATTR m (method flag) + NULL|self

    sched, instr = _walk_until(target, "LOAD_ATTR", {"c": obj})
    if instr is None:
        pytest.skip("interpreter does not emit LOAD_ATTR with method flag here")

    null_on_top = _null_on_top_from_argrepr(instr.argrepr)
    top, below = sched._shadow.stack[-1], sched._shadow.stack[-2]
    # The bound-method value is the non-None slot; the synthetic slot is None.
    if null_on_top:
        assert top is None and below is not None, (
            f"LOAD_ATTR: dis says NULL/self on top ({instr.argrepr!r}) but shadow stack top={top!r} below={below!r}"
        )
    else:
        assert below is None and top is not None, (
            f"LOAD_ATTR: dis says NULL/self below ({instr.argrepr!r}) but shadow stack top={top!r} below={below!r}"
        )


# ---------------------------------------------------------------------------
# Version-gate tests: drive the gate logic for 3.12 and 3.13 explicitly by
# monkeypatching _PY_VERSION, so the mis-gated layouts are caught even when the
# test runs on a single interpreter (3.14).  The correct per-version layout:
#   3.11-3.12: [NULL, value]   (NULL below, value on TOS)
#   3.13+:     [value, NULL]   (value below, NULL on TOS)
# ---------------------------------------------------------------------------


def _walk_until_flagged(fn: Any, opname: str, f_locals: dict[str, Any]) -> _Sched:
    sched = _Sched()
    frame = SimpleNamespace(
        f_code=fn.__code__,
        f_locals=f_locals,
        f_globals=fn.__globals__,
        f_builtins=builtins.__dict__,
        f_lasti=0,
    )
    for instr in dis.get_instructions(fn):
        frame.f_lasti = instr.offset
        _process_opcode(frame, sched, 1)
        if instr.opname == opname and instr.arg is not None and instr.arg & 1:
            break
    return sched


def test_load_global_gate_3_13_is_value_then_null(monkeypatch: Any) -> None:
    def target() -> int:
        return len([1])

    if not any(i.opname == "LOAD_GLOBAL" and i.arg is not None and i.arg & 1 for i in dis.get_instructions(target)):
        pytest.skip("no flagged LOAD_GLOBAL on this interpreter")

    monkeypatch.setattr(obs, "_PY_VERSION", (3, 13))
    sched = _walk_until_flagged(target, "LOAD_GLOBAL", {})
    top, below = sched._shadow.stack[-1], sched._shadow.stack[-2]
    assert top is None and below is not None, (
        f"3.13 LOAD_GLOBAL must be [value, NULL] (NULL on top); got top={top!r} below={below!r}"
    )


def test_load_attr_gate_3_12_is_null_then_value(monkeypatch: Any) -> None:
    class C:
        def m(self) -> int:
            return 1

    obj = C()

    def target(c: C) -> int:
        return c.m()

    if not any(i.opname == "LOAD_ATTR" and i.arg is not None and i.arg & 1 for i in dis.get_instructions(target)):
        pytest.skip("no flagged LOAD_ATTR on this interpreter")

    monkeypatch.setattr(obs, "_PY_VERSION", (3, 12))
    sched = _walk_until_flagged(target, "LOAD_ATTR", {"c": obj})
    top, below = sched._shadow.stack[-1], sched._shadow.stack[-2]
    assert below is None and top is not None, (
        f"3.12 LOAD_ATTR must be [NULL, value] (NULL below); got top={top!r} below={below!r}"
    )
