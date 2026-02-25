"""
Tests for tricky race conditions that stress-test DPOR's ability to detect
races through various Python mechanisms that bypass normal bytecode tracking.

Each test category exploits a specific gap in the DPOR shadow-stack /
opcode-tracing approach.  The races are real (verified by barrier-based
proofs where feasible) but DPOR currently cannot detect them due to
tracking limitations.

These tests assert that DPOR finds the race (``not result.property_holds``).
A **failing** test means DPOR missed a real race — i.e., there is a gap
in the implementation.

Categories of DPOR gaps tested:
1. Closure cell variables (LOAD_DEREF / STORE_DEREF not reported to engine)
2. setattr() / getattr() builtin calls (__self__ is builtins module → skipped)
3. object.__setattr__() (wrapper_descriptor, not builtin_function_or_method)
4. Mixed __dict__ / attribute access (different object keys for same data)
5. exec() / eval() (filename starts with '<' → tracing disabled)
6. globals() dict subscript vs STORE_GLOBAL (repr(key) ≠ argval)
7. operator module functions (operator.setitem, etc. — builtins module __self__)
8. Closure cell + list mutation (compound gap: invisible check + invisible state)
9. type() + setattr compound race (both calls invisible)
10. vars() aliasing race (read via vars(), write via attribute)
"""

from __future__ import annotations

import operator
import threading

import pytest

from frontrun.dpor import explore_dpor

# ---------------------------------------------------------------------------
# 1. Closure cell variable races
#
# LOAD_DEREF and STORE_DEREF access closure "cell" variables, but the
# shadow stack handler for these opcodes does NOT report any read/write
# to the DPOR engine.  Two threads sharing a closure-captured variable
# can race invisibly.
# ---------------------------------------------------------------------------


def _make_closure_counter() -> tuple[object, object]:
    """Create a counter using closure cells (nonlocal variables).

    Returns (increment_fn, get_fn).
    """
    count = 0

    def increment() -> None:
        nonlocal count
        temp = count  # LOAD_DEREF — not reported
        count = temp + 1  # STORE_DEREF — not reported

    def get() -> int:
        return count

    return increment, get


class _ClosureCounterState:
    """Wrapper so explore_dpor can use setup= / threads= / invariant=."""

    def __init__(self) -> None:
        inc, get = _make_closure_counter()
        self.increment = inc
        self.get = get


class TestClosureCellRace:
    """DPOR should detect lost-update races on nonlocal (cell) variables.

    Gap: LOAD_DEREF / STORE_DEREF opcodes are handled in _process_opcode
    but only push/pop the shadow stack — no read/write is reported to the
    DPOR engine.  The engine never sees the conflict.
    """

    def test_dpor_detects_closure_cell_race(self) -> None:
        result = explore_dpor(
            setup=_ClosureCounterState,
            threads=[
                lambda s: s.increment(),
                lambda s: s.increment(),
            ],
            invariant=lambda s: s.get() == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the closure-cell lost-update race (LOAD_DEREF/STORE_DEREF tracking gap)"
        )

    def test_barrier_proves_closure_cell_race_is_real(self) -> None:
        """Barrier-forced interleaving proves the closure-cell race is real."""
        barrier = threading.Barrier(2)

        for _ in range(20):
            count_cell: list[int] = [0]

            def handler() -> None:
                tmp = count_cell[0]
                barrier.wait()
                count_cell[0] = tmp + 1

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if count_cell[0] != 2:
                return
        pytest.fail("Barrier-forced closure-cell race never triggered in 20 attempts")


# ---------------------------------------------------------------------------
# 2. setattr() / getattr() builtin function races
#
# setattr.__self__ is the builtins module (types.ModuleType), which is in
# _IMMUTABLE_TYPES.  The CALL handler skips builtins whose __self__ is
# immutable, so setattr(obj, 'x', val) / getattr(obj, 'x') are invisible.
# ---------------------------------------------------------------------------


class _SetattrCounterState:
    def __init__(self) -> None:
        self.value = 0


def _setattr_increment(state: _SetattrCounterState) -> None:
    temp = getattr(state, "value")  # CALL getattr — invisible
    setattr(state, "value", temp + 1)  # CALL setattr — invisible


def _setattr_invariant(state: _SetattrCounterState) -> bool:
    return state.value == 2


class TestSetattrRace:
    """DPOR should detect races through setattr()/getattr() builtin calls.

    Gap: setattr/getattr are ``builtin_function_or_method`` but their
    ``__self__`` is the ``builtins`` module (``types.ModuleType``), which
    is in ``_IMMUTABLE_TYPES``.  The CALL handler skips them entirely.
    """

    def test_dpor_detects_setattr_race(self) -> None:
        result = explore_dpor(
            setup=_SetattrCounterState,
            threads=[_setattr_increment, _setattr_increment],
            invariant=_setattr_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the setattr()/getattr() lost-update race "
            "(builtins module __self__ excluded from tracking)"
        )

    def test_barrier_proves_setattr_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _SetattrCounterState()

            def handler() -> None:
                tmp = getattr(state, "value")
                barrier.wait()
                setattr(state, "value", tmp + 1)

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.value != 2:
                return
        pytest.fail("Barrier-forced setattr race never triggered in 20 attempts")


# ---------------------------------------------------------------------------
# 3. object.__setattr__() / object.__getattribute__() races
#
# These are wrapper_descriptor types (not builtin_function_or_method),
# so the CALL handler's type(item) is _BUILTIN_METHOD_TYPE check fails.
# The access goes completely undetected.
# ---------------------------------------------------------------------------


class _DunderSetattrState:
    def __init__(self) -> None:
        self.value = 0


def _dunder_setattr_increment(state: _DunderSetattrState) -> None:
    temp = object.__getattribute__(state, "value")  # wrapper_descriptor call
    object.__setattr__(state, "value", temp + 1)  # wrapper_descriptor call


def _dunder_setattr_invariant(state: _DunderSetattrState) -> bool:
    return state.value == 2


class TestObjectDunderSetattrRace:
    """DPOR should detect races through object.__setattr__/__getattribute__.

    Gap: ``object.__setattr__`` and ``object.__getattribute__`` are
    ``wrapper_descriptor`` types, not ``builtin_function_or_method``.
    The CALL handler only checks for the latter type, so these calls
    are completely invisible to DPOR.
    """

    def test_dpor_detects_dunder_setattr_race(self) -> None:
        result = explore_dpor(
            setup=_DunderSetattrState,
            threads=[_dunder_setattr_increment, _dunder_setattr_increment],
            invariant=_dunder_setattr_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the object.__setattr__() lost-update race "
            "(wrapper_descriptor not tracked by CALL handler)"
        )

    def test_barrier_proves_dunder_setattr_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _DunderSetattrState()

            def handler() -> None:
                tmp = object.__getattribute__(state, "value")
                barrier.wait()
                object.__setattr__(state, "value", tmp + 1)

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.value != 2:
                return
        pytest.fail("Barrier-forced object.__setattr__ race never triggered in 20 attempts")


# ---------------------------------------------------------------------------
# 4. Mixed __dict__ / attribute access races
#
# Accessing obj.__dict__['key'] uses STORE_SUBSCR on the __dict__ object
# with key repr('key') = "'key'", while obj.key uses STORE_ATTR with
# argval 'key'.  The DPOR object keys are different:
#   STORE_ATTR:   hash((id(obj), 'key'))
#   STORE_SUBSCR: hash((id(obj.__dict__), "'key'"))
# So cross-path conflicts (one thread using obj.x, another using
# obj.__dict__['x']) are invisible.
# ---------------------------------------------------------------------------


class _DictAccessState:
    def __init__(self) -> None:
        self.value = 0


def _dict_read_attr_write(state: _DictAccessState) -> None:
    """Read via __dict__, write via __dict__ — bypasses STORE_ATTR."""
    d = state.__dict__
    temp = d["value"]  # BINARY_SUBSCR on d
    d["value"] = temp + 1  # STORE_SUBSCR on d


def _dict_access_invariant(state: _DictAccessState) -> bool:
    return state.value == 2


class TestDictDirectAccessRace:
    """DPOR should detect races when one thread uses __dict__ and another uses attributes.

    Gap: STORE_ATTR reports ``hash((id(obj), attr_name))`` while
    STORE_SUBSCR on ``__dict__`` reports ``hash((id(obj.__dict__), repr(key)))``.
    Since ``id(obj) != id(obj.__dict__)`` and ``attr_name != repr(key)``,
    the keys are completely different and no conflict is detected.
    """

    def test_dpor_detects_mixed_attr_dict_race(self) -> None:
        """One thread uses normal attribute access, the other uses __dict__."""

        def attr_increment(state: _DictAccessState) -> None:
            temp = state.value  # LOAD_ATTR
            state.value = temp + 1  # STORE_ATTR

        result = explore_dpor(
            setup=_DictAccessState,
            threads=[attr_increment, _dict_read_attr_write],
            invariant=_dict_access_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the mixed attr/__dict__ lost-update race "
            "(different object keys for same underlying data)"
        )

    def test_barrier_proves_dict_direct_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _DictAccessState()

            def handler_attr() -> None:
                tmp = state.value
                barrier.wait()
                state.value = tmp + 1

            def handler_dict() -> None:
                d = state.__dict__
                tmp = d["value"]
                barrier.wait()
                d["value"] = tmp + 1

            t1 = threading.Thread(target=handler_attr)
            t2 = threading.Thread(target=handler_dict)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.value != 2:
                return
        pytest.fail("Barrier-forced mixed attr/__dict__ race never triggered in 20 attempts")


# ---------------------------------------------------------------------------
# 5. exec() / eval() races
#
# Code executed by exec() has filename '<string>', which starts with '<'.
# The _should_trace_file() function returns False for such filenames,
# so all bytecode inside exec() is invisible to the DPOR tracer.
# ---------------------------------------------------------------------------


class _ExecCounterState:
    def __init__(self) -> None:
        self.value = 0


def _exec_increment(state: _ExecCounterState) -> None:
    # The read-modify-write happens inside exec(), whose code object
    # has filename '<string>' → not traced.
    exec("state.value = state.value + 1", {"state": state})  # noqa: S102


def _exec_invariant(state: _ExecCounterState) -> bool:
    return state.value == 2


class TestExecEvalRace:
    """DPOR should detect races inside exec()/eval() code.

    Gap: ``_should_trace_file()`` returns ``False`` for filenames starting
    with ``'<'``.  ``exec()``/``eval()`` code objects have filename
    ``'<string>'``, so their bytecode is never instrumented.  All
    attribute accesses inside exec/eval are completely invisible.
    """

    def test_dpor_detects_exec_race(self) -> None:
        result = explore_dpor(
            setup=_ExecCounterState,
            threads=[_exec_increment, _exec_increment],
            invariant=_exec_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the exec() lost-update race (filename '<string>' excluded from tracing)"
        )

    def test_dpor_detects_eval_read_exec_write_race(self) -> None:
        """Use eval() for reading and exec() for writing — both invisible."""

        def eval_exec_increment(state: _ExecCounterState) -> None:
            temp = eval("state.value", {"state": state})  # noqa: S307
            exec("state.value = temp + 1", {"state": state, "temp": temp})  # noqa: S102

        result = explore_dpor(
            setup=_ExecCounterState,
            threads=[eval_exec_increment, eval_exec_increment],
            invariant=_exec_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the eval()/exec() lost-update race"

    def test_barrier_proves_exec_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _ExecCounterState()

            def handler() -> None:
                tmp = state.value
                barrier.wait()
                exec("state.value = tmp + 1", {"state": state, "tmp": tmp})  # noqa: S102

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.value != 2:
                return
        pytest.fail("Barrier-forced exec() race never triggered in 20 attempts")


# ---------------------------------------------------------------------------
# 6. globals() dict subscript vs STORE_GLOBAL mismatch
#
# STORE_GLOBAL reports: _make_object_key(id(f_globals), argval)
#   where argval is the variable name, e.g. 'x'
# STORE_SUBSCR on globals()['x'] reports: _make_object_key(id(globals_dict), repr(key))
#   where repr(key) is "'x'" (with quotes!)
# These produce different object keys, so cross-path conflicts are invisible.
# ---------------------------------------------------------------------------


_global_via_subscript: int = 0


class _GlobalSubscriptState:
    def __init__(self) -> None:
        global _global_via_subscript
        _global_via_subscript = 0


def _global_store_global_increment(_state: _GlobalSubscriptState) -> None:
    """Increment using normal STORE_GLOBAL."""
    global _global_via_subscript
    tmp = _global_via_subscript
    _global_via_subscript = tmp + 1


def _global_subscript_increment(_state: _GlobalSubscriptState) -> None:
    """Increment using globals()['name'] — different DPOR object key."""
    g = globals()
    tmp = g["_global_via_subscript"]
    g["_global_via_subscript"] = tmp + 1


def _global_subscript_invariant(_state: _GlobalSubscriptState) -> bool:
    return _global_via_subscript == 2


class TestGlobalSubscriptRace:
    """DPOR should detect races between STORE_GLOBAL and globals()['x'] = v.

    Gap: STORE_GLOBAL reports key ``hash((id(f_globals), 'name'))``.
    STORE_SUBSCR reports key ``hash((id(globals_dict), repr('name')))``.
    Since ``repr('name') == "'name'"`` (with quotes) but
    ``argval == 'name'`` (without quotes), the keys differ and DPOR
    sees no conflict.
    """

    def test_dpor_detects_mixed_global_access_race(self) -> None:
        result = explore_dpor(
            setup=_GlobalSubscriptState,
            threads=[_global_store_global_increment, _global_subscript_increment],
            invariant=_global_subscript_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the STORE_GLOBAL vs globals()['x'] race (argval vs repr(key) object key mismatch)"
        )

    def test_barrier_proves_mixed_global_race_is_real(self) -> None:
        global _global_via_subscript
        barrier = threading.Barrier(2)

        for _ in range(20):
            _global_via_subscript = 0

            def handler_store_global() -> None:
                global _global_via_subscript
                tmp = _global_via_subscript
                barrier.wait()
                _global_via_subscript = tmp + 1

            def handler_subscript() -> None:
                g = globals()
                tmp = g["_global_via_subscript"]
                barrier.wait()
                g["_global_via_subscript"] = tmp + 1

            t1 = threading.Thread(target=handler_store_global)
            t2 = threading.Thread(target=handler_subscript)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if _global_via_subscript != 2:
                return
        pytest.fail("Barrier-forced mixed global access race never triggered")


# ---------------------------------------------------------------------------
# 7. operator module function races
#
# operator.setitem(d, k, v) is equivalent to d[k] = v but goes through
# a C function call.  operator.setitem is a builtin_function_or_method
# whose __self__ is the builtins module, so the CALL handler skips it.
# ---------------------------------------------------------------------------


class _OperatorModuleState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"count": 0}


def _operator_increment(state: _OperatorModuleState) -> None:
    temp = operator.getitem(state.data, "count")
    operator.setitem(state.data, "count", temp + 1)


def _operator_invariant(state: _OperatorModuleState) -> bool:
    return state.data["count"] == 2


class TestOperatorModuleRace:
    """DPOR should detect races through operator module functions.

    Gap: ``operator.getitem``/``operator.setitem`` are
    ``builtin_function_or_method`` with ``__self__`` = builtins module.
    The CALL handler skips them because ``types.ModuleType`` is in
    ``_IMMUTABLE_TYPES``.
    """

    def test_dpor_detects_operator_setitem_race(self) -> None:
        result = explore_dpor(
            setup=_OperatorModuleState,
            threads=[_operator_increment, _operator_increment],
            invariant=_operator_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the operator.setitem() lost-update race "
            "(builtins module __self__ excluded from tracking)"
        )

    def test_barrier_proves_operator_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _OperatorModuleState()

            def handler() -> None:
                tmp = operator.getitem(state.data, "count")
                barrier.wait()
                operator.setitem(state.data, "count", tmp + 1)

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.data["count"] != 2:
                return
        pytest.fail("Barrier-forced operator module race never triggered")


# ---------------------------------------------------------------------------
# 8. Closure cell + list mutation (compound gap)
#
# Combines two gaps: closure cell (LOAD_DEREF invisible) and len()
# (builtins __self__ invisible).  The entire check-then-act pattern is
# invisible to DPOR.
# ---------------------------------------------------------------------------


def _make_closure_list_checker() -> tuple[object, object, object]:
    """Create a list guarded by a closure with a check-then-act pattern."""
    items: list[str] = ["only-item"]
    pop_count = 0

    def try_pop() -> None:
        nonlocal pop_count
        # LOAD_DEREF for items — not reported
        # CALL len — builtins, not reported
        # Everything is invisible to DPOR
        if len(items) > 0:
            try:
                items.pop()
            except IndexError:
                pass
            pop_count += 1  # STORE_DEREF — not reported

    def get_pop_count() -> int:
        return pop_count

    def get_items() -> list[str]:
        return items

    return try_pop, get_pop_count, get_items


class _ClosureListState:
    def __init__(self) -> None:
        try_pop, get_pop_count, get_items = _make_closure_list_checker()
        self.try_pop = try_pop
        self.get_pop_count = get_pop_count
        self.get_items = get_items


class TestClosureListCompoundRace:
    """DPOR should detect races in closure-captured lists.

    Compound gap: LOAD_DEREF (closure cell) + len() (builtins) + STORE_DEREF
    (closure cell) are all invisible.  The entire check-then-act pattern
    is untracked.
    """

    def test_dpor_detects_closure_list_race(self) -> None:
        result = explore_dpor(
            setup=_ClosureListState,
            threads=[
                lambda s: s.try_pop(),
                lambda s: s.try_pop(),
            ],
            invariant=lambda s: s.get_pop_count() <= 1,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the closure-list compound race (LOAD_DEREF + len() + STORE_DEREF all invisible)"
        )


# ---------------------------------------------------------------------------
# 9. type() + setattr() compound race
#
# Both type() and setattr() are builtins with __self__ = builtins module.
# A pattern like ``setattr(type(obj), 'class_counter', val)`` is doubly
# invisible: neither the type() call (reading the class) nor the setattr()
# call (writing the class attribute) is reported.
# ---------------------------------------------------------------------------


class _TypeSetattrState:
    class_counter: int = 0

    def __init__(self) -> None:
        type(self).class_counter = 0


def _type_setattr_increment(state: _TypeSetattrState) -> None:
    # type() call: builtins module __self__ → invisible
    # getattr() call: builtins module __self__ → invisible
    # setattr() call: builtins module __self__ → invisible
    cls = type(state)
    temp = getattr(cls, "class_counter")
    setattr(cls, "class_counter", temp + 1)


def _type_setattr_invariant(state: _TypeSetattrState) -> bool:
    return type(state).class_counter == 2


class TestTypeSetattrCompoundRace:
    """DPOR should detect races through type() + setattr() combinations.

    Compound gap: ``type(obj)`` returns the class object but the call is
    invisible.  ``getattr(cls, 'x')`` and ``setattr(cls, 'x', val)`` are
    also invisible.  The entire read-modify-write on a class attribute
    via builtins is untracked.
    """

    def test_dpor_detects_type_setattr_race(self) -> None:
        result = explore_dpor(
            setup=_TypeSetattrState,
            threads=[_type_setattr_increment, _type_setattr_increment],
            invariant=_type_setattr_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the type()+setattr() class-attribute race "
            "(all calls go through builtins with invisible __self__)"
        )

    def test_barrier_proves_type_setattr_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            _TypeSetattrState.class_counter = 0

            def handler() -> None:
                cls = type(_TypeSetattrState())
                tmp = getattr(cls, "class_counter")
                barrier.wait()
                setattr(cls, "class_counter", tmp + 1)

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if _TypeSetattrState.class_counter != 2:
                return
        pytest.fail("Barrier-forced type()+setattr() race never triggered")


# ---------------------------------------------------------------------------
# 10. vars() aliasing race
#
# vars(obj) returns obj.__dict__.  Writing through vars() uses
# STORE_SUBSCR on the __dict__ object with repr(key), while reading
# through normal attribute access uses LOAD_ATTR with argval.
# The object keys are different.
# ---------------------------------------------------------------------------


class _VarsAliasState:
    def __init__(self) -> None:
        self.value = 0


def _vars_write(state: _VarsAliasState) -> None:
    """Write via vars() — tracked as STORE_SUBSCR on __dict__."""
    d = vars(state)
    temp = d["value"]
    d["value"] = temp + 1


def _attr_write(state: _VarsAliasState) -> None:
    """Write via normal attribute — tracked as STORE_ATTR on obj."""
    temp = state.value
    state.value = temp + 1


def _vars_alias_invariant(state: _VarsAliasState) -> bool:
    return state.value == 2


class TestVarsAliasRace:
    """DPOR should detect races between vars(obj)['x'] and obj.x access.

    Gap: Same as the mixed __dict__/attribute gap (#4), but using the
    ``vars()`` builtin instead of ``__dict__``.  ``vars()`` is also a
    builtin with ``__self__`` = builtins module, adding another layer
    of invisibility (DPOR can't even see that vars() was called).
    """

    def test_dpor_detects_vars_alias_race(self) -> None:
        result = explore_dpor(
            setup=_VarsAliasState,
            threads=[_attr_write, _vars_write],
            invariant=_vars_alias_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the vars()/attribute aliasing race"

    def test_barrier_proves_vars_alias_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _VarsAliasState()

            def handler_attr() -> None:
                tmp = state.value
                barrier.wait()
                state.value = tmp + 1

            def handler_vars() -> None:
                d = vars(state)
                tmp = d["value"]
                barrier.wait()
                d["value"] = tmp + 1

            t1 = threading.Thread(target=handler_attr)
            t2 = threading.Thread(target=handler_vars)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.value != 2:
                return
        pytest.fail("Barrier-forced vars()/attribute race never triggered")


# ---------------------------------------------------------------------------
# 11. exec() with compile() using custom filename
#
# Even when a user compiles code with compile() and a real-looking
# filename, exec() still uses the compiled code object's co_filename.
# If the filename starts with '<', tracing is disabled.  This test
# verifies that the common compile(source, '<eval>', 'exec') pattern
# (used by many templating engines and ORMs) is invisible.
# ---------------------------------------------------------------------------


class _CompileExecState:
    def __init__(self) -> None:
        self.value = 0


def _compile_exec_increment(state: _CompileExecState) -> None:
    """Use compile() + exec() — common in ORMs and templating engines."""
    code = compile("state.value = state.value + 1", "<generated>", "exec")
    exec(code, {"state": state})  # noqa: S102


def _compile_exec_invariant(state: _CompileExecState) -> bool:
    return state.value == 2


class TestCompileExecRace:
    """DPOR should detect races in compile()+exec() code.

    Gap: ``compile(source, '<generated>', 'exec')`` creates a code object
    with ``co_filename='<generated>'``.  Since it starts with ``'<'``,
    ``_should_trace_file()`` returns ``False`` and the code is untraced.
    This pattern is very common in ORMs (SQLAlchemy), template engines
    (Jinja2, Mako), and serialization libraries.
    """

    def test_dpor_detects_compile_exec_race(self) -> None:
        result = explore_dpor(
            setup=_CompileExecState,
            threads=[_compile_exec_increment, _compile_exec_increment],
            invariant=_compile_exec_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the compile()+exec() race (co_filename '<generated>' excluded from tracing)"
        )


# ---------------------------------------------------------------------------
# 12. dict.update() race through operator or builtins
#
# dict.update() is a C-level method.  When the dict is loaded via
# LOAD_FAST (local variable, invisible) and updated via operator or
# builtins, the mutation is completely untracked.
# ---------------------------------------------------------------------------


class _DictUpdateState:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {"a": 0}


def _dict_operator_iadd(state: _DictUpdateState) -> None:
    """Read-modify-write on a dict value using operator.getitem/setitem."""
    d = state.counts
    temp = operator.getitem(d, "a")  # invisible (builtins __self__)
    operator.setitem(d, "a", temp + 1)  # invisible (builtins __self__)


def _dict_update_invariant(state: _DictUpdateState) -> bool:
    return state.counts["a"] == 2


class TestDictOperatorRace:
    """DPOR should detect races on dicts accessed through operator module.

    Gap: ``operator.getitem``/``operator.setitem`` have ``__self__`` =
    builtins module.  The dict is loaded via LOAD_ATTR (which IS tracked
    on the container ``state.counts``), but the actual read/write through
    operator functions is invisible.  Even though DPOR sees the dict load,
    it can't see the mutation, so it doesn't know the two threads conflict.
    """

    def test_dpor_detects_dict_operator_race(self) -> None:
        result = explore_dpor(
            setup=_DictUpdateState,
            threads=[_dict_operator_iadd, _dict_operator_iadd],
            invariant=_dict_update_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the dict operator.setitem() race"

    def test_barrier_proves_dict_operator_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _DictUpdateState()

            def handler() -> None:
                d = state.counts
                tmp = operator.getitem(d, "a")
                barrier.wait()
                operator.setitem(d, "a", tmp + 1)

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.counts["a"] != 2:
                return
        pytest.fail("Barrier-forced dict operator race never triggered")


# ---------------------------------------------------------------------------
# 13. Mixed LOAD_GLOBAL / STORE_GLOBAL with exec() modifying same global
#
# Combines two gaps: one thread uses STORE_GLOBAL (tracked) while
# the other uses exec() (untraced).  The exec'd code accesses the
# same global but is invisible, creating an asymmetric race.
# ---------------------------------------------------------------------------


_exec_global_counter: int = 0


class _ExecGlobalState:
    def __init__(self) -> None:
        global _exec_global_counter
        _exec_global_counter = 0


def _normal_global_increment(_state: _ExecGlobalState) -> None:
    global _exec_global_counter
    tmp = _exec_global_counter  # LOAD_GLOBAL — tracked
    _exec_global_counter = tmp + 1  # STORE_GLOBAL — tracked


def _exec_global_increment(_state: _ExecGlobalState) -> None:
    # exec() with access to the module's globals
    exec(  # noqa: S102
        "global _exec_global_counter\ntmp = _exec_global_counter\n_exec_global_counter = tmp + 1",
        globals(),
    )


def _exec_global_invariant(_state: _ExecGlobalState) -> bool:
    return _exec_global_counter == 2


class TestExecGlobalMixedRace:
    """DPOR should detect races between normal globals and exec'd globals.

    Compound gap: Thread 1 uses STORE_GLOBAL (tracked).  Thread 2 uses
    exec() which has filename '<string>' (untraced), so its
    LOAD_GLOBAL/STORE_GLOBAL are invisible.  DPOR only sees Thread 1's
    access and has no conflict to trigger interleaving exploration.
    """

    def test_dpor_detects_exec_global_mixed_race(self) -> None:
        result = explore_dpor(
            setup=_ExecGlobalState,
            threads=[_normal_global_increment, _exec_global_increment],
            invariant=_exec_global_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the normal-global vs exec()-global race"

    def test_barrier_proves_exec_global_mixed_race_is_real(self) -> None:
        global _exec_global_counter
        barrier = threading.Barrier(2)

        for _ in range(20):
            _exec_global_counter = 0

            def handler_normal() -> None:
                global _exec_global_counter
                tmp = _exec_global_counter
                barrier.wait()
                _exec_global_counter = tmp + 1

            def handler_exec() -> None:
                g = globals()
                tmp = g["_exec_global_counter"]
                barrier.wait()
                g["_exec_global_counter"] = tmp + 1

            t1 = threading.Thread(target=handler_normal)
            t2 = threading.Thread(target=handler_exec)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if _exec_global_counter != 2:
                return
        pytest.fail("Barrier-forced exec-global mixed race never triggered")


# ---------------------------------------------------------------------------
# 14. Closure cell with augmented assignment (nonlocal +=)
#
# The ``nonlocal x; x += 1`` pattern compiles to:
#   LOAD_DEREF x → LOAD_SMALL_INT 1 → BINARY_OP += → STORE_DEREF x
# None of these report accesses.  Even though the standard attribute
# ``self.value += 1`` is detected (via LOAD_ATTR+STORE_ATTR), the
# closure cell equivalent is completely invisible.
# ---------------------------------------------------------------------------


def _make_augmented_closure() -> tuple[object, object]:
    """Create a counter using ``nonlocal +=`` (augmented assignment on cell)."""
    count = 0

    def increment() -> None:
        nonlocal count
        count += 1  # LOAD_DEREF + BINARY_OP + STORE_DEREF — all invisible

    def get() -> int:
        return count

    return increment, get


class _AugmentedClosureState:
    def __init__(self) -> None:
        inc, get = _make_augmented_closure()
        self.increment = inc
        self.get = get


class TestAugmentedClosureRace:
    """DPOR should detect ``nonlocal x; x += 1`` races.

    Gap: Same as #1 (LOAD_DEREF/STORE_DEREF) but using augmented
    assignment.  The ``+=`` compiles to LOAD_DEREF + BINARY_OP +
    STORE_DEREF, none of which report accesses.  This is the closure
    analog of the ``self.value += 1`` pattern that IS detected via
    LOAD_ATTR + STORE_ATTR.
    """

    def test_dpor_detects_augmented_closure_race(self) -> None:
        result = explore_dpor(
            setup=_AugmentedClosureState,
            threads=[
                lambda s: s.increment(),
                lambda s: s.increment(),
            ],
            invariant=lambda s: s.get() == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, (
            "DPOR should detect the nonlocal += lost-update race (LOAD_DEREF + BINARY_OP + STORE_DEREF all invisible)"
        )


# ---------------------------------------------------------------------------
# 15. Wrapper descriptor races (dict.__setitem__, list.__setitem__, etc.)
#
# Calling unbound C methods like dict.__setitem__(d, k, v) uses
# wrapper_descriptor, not builtin_function_or_method.  The CALL handler
# only checks for the latter type.
# ---------------------------------------------------------------------------


class _WrapperDescriptorState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"count": 0}


def _wrapper_descriptor_increment(state: _WrapperDescriptorState) -> None:
    """Use unbound dict methods — wrapper_descriptor, not detected."""
    temp = dict.__getitem__(state.data, "count")  # wrapper_descriptor
    dict.__setitem__(state.data, "count", temp + 1)  # wrapper_descriptor


def _wrapper_descriptor_invariant(state: _WrapperDescriptorState) -> bool:
    return state.data["count"] == 2


class TestWrapperDescriptorRace:
    """DPOR should detect races through unbound C type methods.

    Gap: ``dict.__setitem__``, ``list.__setitem__``, etc. are
    ``wrapper_descriptor`` objects.  The CALL handler only checks
    for ``builtin_function_or_method`` (``type(len)``), so unbound
    C type methods are completely invisible.  This pattern is common
    in code that uses ``super().__setattr__()`` or calls dunder methods
    directly.
    """

    def test_dpor_detects_wrapper_descriptor_race(self) -> None:
        result = explore_dpor(
            setup=_WrapperDescriptorState,
            threads=[_wrapper_descriptor_increment, _wrapper_descriptor_increment],
            invariant=_wrapper_descriptor_invariant,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds, "DPOR should detect the wrapper_descriptor (dict.__setitem__) race"

    def test_barrier_proves_wrapper_descriptor_race_is_real(self) -> None:
        barrier = threading.Barrier(2)

        for _ in range(20):
            state = _WrapperDescriptorState()

            def handler() -> None:
                tmp = dict.__getitem__(state.data, "count")
                barrier.wait()
                dict.__setitem__(state.data, "count", tmp + 1)

            t1 = threading.Thread(target=handler)
            t2 = threading.Thread(target=handler)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if state.data["count"] != 2:
                return
        pytest.fail("Barrier-forced wrapper_descriptor race never triggered")
