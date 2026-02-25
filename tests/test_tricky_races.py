"""Tests for DPOR detection of races through non-obvious Python mechanisms."""

from __future__ import annotations

import operator

from frontrun.dpor import explore_dpor

# -- Helpers ------------------------------------------------------------------


def _make_closure_counter() -> tuple[object, object]:
    count = 0

    def increment() -> None:
        nonlocal count
        temp = count
        count = temp + 1

    def get() -> int:
        return count

    return increment, get


class _ClosureCounterState:
    def __init__(self) -> None:
        inc, get = _make_closure_counter()
        self.increment = inc
        self.get = get


class _SetattrCounterState:
    def __init__(self) -> None:
        self.value = 0


class _DunderSetattrState:
    def __init__(self) -> None:
        self.value = 0


class _DictAccessState:
    def __init__(self) -> None:
        self.value = 0


class _ExecCounterState:
    def __init__(self) -> None:
        self.value = 0


_global_via_subscript: int = 0


class _GlobalSubscriptState:
    def __init__(self) -> None:
        global _global_via_subscript
        _global_via_subscript = 0


class _OperatorModuleState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"count": 0}


def _make_closure_list_checker() -> tuple[object, object, object]:
    items: list[str] = ["only-item"]
    pop_count = 0

    def try_pop() -> None:
        nonlocal pop_count
        if len(items) > 0:
            try:
                items.pop()
            except IndexError:
                pass
            pop_count += 1

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


class _TypeSetattrState:
    class_counter: int = 0

    def __init__(self) -> None:
        type(self).class_counter = 0


class _VarsAliasState:
    def __init__(self) -> None:
        self.value = 0


class _CompileExecState:
    def __init__(self) -> None:
        self.value = 0


class _DictUpdateState:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {"a": 0}


_exec_global_counter: int = 0


class _ExecGlobalState:
    def __init__(self) -> None:
        global _exec_global_counter
        _exec_global_counter = 0


def _make_augmented_closure() -> tuple[object, object]:
    count = 0

    def increment() -> None:
        nonlocal count
        count += 1

    def get() -> int:
        return count

    return increment, get


class _AugmentedClosureState:
    def __init__(self) -> None:
        inc, get = _make_augmented_closure()
        self.increment = inc
        self.get = get


class _WrapperDescriptorState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"count": 0}


# -- Tests --------------------------------------------------------------------


class TestClosureCellRace:
    """LOAD_DEREF / STORE_DEREF on nonlocal (cell) variables."""

    def test_dpor_detects_closure_cell_race(self) -> None:
        result = explore_dpor(
            setup=_ClosureCounterState,
            threads=[lambda s: s.increment(), lambda s: s.increment()],
            invariant=lambda s: s.get() == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestSetattrRace:
    """setattr() / getattr() builtin calls."""

    def test_dpor_detects_setattr_race(self) -> None:
        def inc(state: _SetattrCounterState) -> None:
            temp = getattr(state, "value")
            setattr(state, "value", temp + 1)

        result = explore_dpor(
            setup=_SetattrCounterState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestObjectDunderSetattrRace:
    """object.__setattr__ / object.__getattribute__ (wrapper_descriptor)."""

    def test_dpor_detects_dunder_setattr_race(self) -> None:
        def inc(state: _DunderSetattrState) -> None:
            temp = object.__getattribute__(state, "value")
            object.__setattr__(state, "value", temp + 1)

        result = explore_dpor(
            setup=_DunderSetattrState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestDictDirectAccessRace:
    """Mixed __dict__ subscript vs normal attribute access."""

    def test_dpor_detects_mixed_attr_dict_race(self) -> None:
        def attr_inc(state: _DictAccessState) -> None:
            temp = state.value
            state.value = temp + 1

        def dict_inc(state: _DictAccessState) -> None:
            d = state.__dict__
            temp = d["value"]
            d["value"] = temp + 1

        result = explore_dpor(
            setup=_DictAccessState,
            threads=[attr_inc, dict_inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestExecEvalRace:
    """exec() / eval() code (dynamically compiled code objects)."""

    def test_dpor_detects_exec_race(self) -> None:
        def inc(state: _ExecCounterState) -> None:
            exec("state.value = state.value + 1", {"state": state})  # noqa: S102

        result = explore_dpor(
            setup=_ExecCounterState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds

    def test_dpor_detects_eval_read_exec_write_race(self) -> None:
        def inc(state: _ExecCounterState) -> None:
            temp = eval("state.value", {"state": state})  # noqa: S307
            exec("state.value = temp + 1", {"state": state, "temp": temp})  # noqa: S102

        result = explore_dpor(
            setup=_ExecCounterState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestGlobalSubscriptRace:
    """STORE_GLOBAL vs globals()['x'] subscript access."""

    def test_dpor_detects_mixed_global_access_race(self) -> None:
        def store_global_inc(_state: _GlobalSubscriptState) -> None:
            global _global_via_subscript
            tmp = _global_via_subscript
            _global_via_subscript = tmp + 1

        def subscript_inc(_state: _GlobalSubscriptState) -> None:
            g = globals()
            tmp = g["_global_via_subscript"]
            g["_global_via_subscript"] = tmp + 1

        result = explore_dpor(
            setup=_GlobalSubscriptState,
            threads=[store_global_inc, subscript_inc],
            invariant=lambda _s: _global_via_subscript == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestOperatorModuleRace:
    """operator.getitem / operator.setitem calls."""

    def test_dpor_detects_operator_setitem_race(self) -> None:
        def inc(state: _OperatorModuleState) -> None:
            temp = operator.getitem(state.data, "count")
            operator.setitem(state.data, "count", temp + 1)

        result = explore_dpor(
            setup=_OperatorModuleState,
            threads=[inc, inc],
            invariant=lambda s: s.data["count"] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestClosureListCompoundRace:
    """Closure cell + len() + list mutation (compound)."""

    def test_dpor_detects_closure_list_race(self) -> None:
        result = explore_dpor(
            setup=_ClosureListState,
            threads=[lambda s: s.try_pop(), lambda s: s.try_pop()],
            invariant=lambda s: s.get_pop_count() <= 1,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestTypeSetattrCompoundRace:
    """type() + getattr() + setattr() on class attributes."""

    def test_dpor_detects_type_setattr_race(self) -> None:
        def inc(state: _TypeSetattrState) -> None:
            cls = type(state)
            temp = getattr(cls, "class_counter")
            setattr(cls, "class_counter", temp + 1)

        result = explore_dpor(
            setup=_TypeSetattrState,
            threads=[inc, inc],
            invariant=lambda s: type(s).class_counter == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestVarsAliasRace:
    """vars(obj)['x'] vs obj.x access paths."""

    def test_dpor_detects_vars_alias_race(self) -> None:
        def attr_inc(state: _VarsAliasState) -> None:
            temp = state.value
            state.value = temp + 1

        def vars_inc(state: _VarsAliasState) -> None:
            d = vars(state)
            temp = d["value"]
            d["value"] = temp + 1

        result = explore_dpor(
            setup=_VarsAliasState,
            threads=[attr_inc, vars_inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestCompileExecRace:
    """compile() + exec() with synthetic filenames."""

    def test_dpor_detects_compile_exec_race(self) -> None:
        def inc(state: _CompileExecState) -> None:
            code = compile("state.value = state.value + 1", "<generated>", "exec")
            exec(code, {"state": state})  # noqa: S102

        result = explore_dpor(
            setup=_CompileExecState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestDictOperatorRace:
    """Dict mutation through operator.getitem / operator.setitem."""

    def test_dpor_detects_dict_operator_race(self) -> None:
        def inc(state: _DictUpdateState) -> None:
            d = state.counts
            temp = operator.getitem(d, "a")
            operator.setitem(d, "a", temp + 1)

        result = explore_dpor(
            setup=_DictUpdateState,
            threads=[inc, inc],
            invariant=lambda s: s.counts["a"] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestExecGlobalMixedRace:
    """Normal STORE_GLOBAL vs exec() modifying the same global."""

    def test_dpor_detects_exec_global_mixed_race(self) -> None:
        def normal_inc(_state: _ExecGlobalState) -> None:
            global _exec_global_counter
            tmp = _exec_global_counter
            _exec_global_counter = tmp + 1

        def exec_inc(_state: _ExecGlobalState) -> None:
            exec(  # noqa: S102
                "global _exec_global_counter\ntmp = _exec_global_counter\n_exec_global_counter = tmp + 1",
                globals(),
            )

        result = explore_dpor(
            setup=_ExecGlobalState,
            threads=[normal_inc, exec_inc],
            invariant=lambda _s: _exec_global_counter == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestAugmentedClosureRace:
    """nonlocal x; x += 1 (augmented assignment on closure cell)."""

    def test_dpor_detects_augmented_closure_race(self) -> None:
        result = explore_dpor(
            setup=_AugmentedClosureState,
            threads=[lambda s: s.increment(), lambda s: s.increment()],
            invariant=lambda s: s.get() == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestWrapperDescriptorRace:
    """Unbound C type methods (dict.__setitem__, dict.__getitem__)."""

    def test_dpor_detects_wrapper_descriptor_race(self) -> None:
        def inc(state: _WrapperDescriptorState) -> None:
            temp = dict.__getitem__(state.data, "count")
            dict.__setitem__(state.data, "count", temp + 1)

        result = explore_dpor(
            setup=_WrapperDescriptorState,
            threads=[inc, inc],
            invariant=lambda s: s.data["count"] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds
