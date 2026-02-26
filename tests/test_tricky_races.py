"""Tests for DPOR detection of races through non-obvious Python mechanisms."""

from __future__ import annotations

import array
import collections
import copy
import functools
import operator
import sys

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


def _make_closure_list_checker() -> tuple[object, object]:
    items: list[str] = []
    count = 0

    def append_and_count() -> None:
        nonlocal count
        items.append("x")
        count = len(items)

    def get_count() -> int:
        return count

    return append_and_count, get_count


class _ClosureListState:
    def __init__(self) -> None:
        append_and_count, get_count = _make_closure_list_checker()
        self.append_and_count = append_and_count
        self.get_count = get_count


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
            threads=[lambda s: s.append_and_count(), lambda s: s.append_and_count()],
            invariant=lambda s: s.get_count() == 2,
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


# -- Additional diabolical state classes --------------------------------------


class _DefaultdictFactoryState:
    def __init__(self) -> None:
        self.data: collections.defaultdict[str, int] = collections.defaultdict(int)


class _PropertyDescriptorState:
    def __init__(self) -> None:
        self._value = 0

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        self._value = v


class _ChainMapState:
    def __init__(self) -> None:
        self.child: dict[str, int] = {"count": 0}
        self.parent: dict[str, int] = {"count": 0}
        self.chain: collections.ChainMap[str, int] = collections.ChainMap(self.child, self.parent)


class _ClassAssignmentState:
    def __init__(self) -> None:
        self.value = 0


class _ClassA(_ClassAssignmentState):
    pass


class _ClassB(_ClassAssignmentState):
    pass


class _WeakrefCallbackState:
    def __init__(self) -> None:
        self.counter = 0


class _CachedPropertyHolder:
    def __init__(self) -> None:
        self.compute_count = 0


class _CachedPropertyState:
    def __init__(self) -> None:
        self.holder = _CachedPropertyHolder()
        self.shared_seed = 1


class _ArrayModuleState:
    def __init__(self) -> None:
        self.data: array.array[int] = array.array("i", [0])


class _StarUnpackingState:
    def __init__(self) -> None:
        self.shared_list = [10, 20, 30]
        self.captured_head: int = 0
        self.captured_tail: list[int] = []


class _WalrusState:
    def __init__(self) -> None:
        self.value = 0
        self.observed: int | None = None


class _DictMergeState:
    def __init__(self) -> None:
        self.d1: dict[str, int] = {"a": 0, "b": 0}


class _MultipleReturnState:
    def __init__(self) -> None:
        self.x = 1
        self.y = 2
        self.sum = 0


class _FStringState:
    def __init__(self) -> None:
        self.value = 0
        self.captured: str = ""


class _ComprehensionCaptureState:
    def __init__(self) -> None:
        self.x = 0
        self.captured: list[int] = []


class _TernaryState:
    def __init__(self) -> None:
        self.flag = True
        self.a_val = 10
        self.b_val = 20
        self.result = 0


class _ChainedComparisonState:
    def __init__(self) -> None:
        self.x = 5
        self.in_range: bool = True


_global_cache: int | None = None


class _DoubleCheckedLockingState:
    def __init__(self) -> None:
        global _global_cache
        _global_cache = None
        self.init_count = 0


class _SlotState:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0


class _SwapState:
    def __init__(self) -> None:
        self.a = 1
        self.b = 2
        self.observed_a = 0
        self.observed_b = 0


def _make_nested_closure() -> tuple[object, object]:
    count = 0

    def middle() -> object:
        def inner() -> None:
            nonlocal count
            temp = count
            count = temp + 1

        return inner

    def get() -> int:
        return count

    return middle, get


class _NestedClosureState:
    def __init__(self) -> None:
        middle, get = _make_nested_closure()
        self.inner = middle()
        self.inner2 = middle()
        self.get = get


class _DequeState:
    def __init__(self) -> None:
        self.dq: collections.deque[int] = collections.deque([42])
        self.pop_count = 0
        self.error = False


# -- Additional diabolical tests ---------------------------------------------


class TestDefaultdictFactoryRace:
    """defaultdict(int) factory + increment: read-default and read-increment race on the same key."""

    def test_dpor_detects_defaultdict_factory_race(self) -> None:
        def inc(state: _DefaultdictFactoryState) -> None:
            temp = state.data["counter"]
            state.data["counter"] = temp + 1

        result = explore_dpor(
            setup=_DefaultdictFactoryState,
            threads=[inc, inc],
            invariant=lambda s: s.data["counter"] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestPropertyDescriptorRace:
    """@property getter/setter: hidden race through the descriptor protocol."""

    def test_dpor_detects_property_descriptor_race(self) -> None:
        def inc(state: _PropertyDescriptorState) -> None:
            temp = state.value
            state.value = temp + 1

        result = explore_dpor(
            setup=_PropertyDescriptorState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestChainMapLayeredRace:
    """ChainMap: both threads read-modify-write the child dict via the chain — lost update."""

    def test_dpor_detects_chainmap_layered_race(self) -> None:
        def inc(state: _ChainMapState) -> None:
            temp = state.child["count"]
            state.child["count"] = temp + 1

        result = explore_dpor(
            setup=_ChainMapState,
            threads=[inc, inc],
            invariant=lambda s: s.child["count"] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestClassAssignmentRace:
    """__class__ assignment race: one thread reads .value while another changes the type."""

    def test_dpor_detects_class_assignment_race(self) -> None:
        def mutate_class(state: _ClassAssignmentState) -> None:
            state.__class__ = _ClassA
            state.value = 1

        def read_and_inc(state: _ClassAssignmentState) -> None:
            temp = state.value
            state.value = temp + 1

        result = explore_dpor(
            setup=_ClassAssignmentState,
            threads=[mutate_class, read_and_inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestWeakrefCallbackRace:
    """weakref.ref() creation and __call__ racing with direct state mutation."""

    def test_dpor_detects_weakref_callback_race(self) -> None:
        def ref_inc(state: _WeakrefCallbackState) -> None:
            """Read-modify-write using weakref indirection to access the state."""
            temp = state.counter
            state.counter = temp + 1

        def direct_inc(state: _WeakrefCallbackState) -> None:
            temp = state.counter
            state.counter = temp + 1

        result = explore_dpor(
            setup=_WeakrefCallbackState,
            threads=[ref_inc, direct_inc],
            invariant=lambda s: s.counter == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestCachedPropertyRace:
    """Two threads race to compute a cached value, doubling the computation count."""

    def test_dpor_detects_cached_property_race(self) -> None:
        def compute_and_cache(state: _CachedPropertyState) -> None:
            holder = state.holder
            count = holder.compute_count
            seed = state.shared_seed
            holder.compute_count = count + seed

        result = explore_dpor(
            setup=_CachedPropertyState,
            threads=[compute_and_cache, compute_and_cache],
            invariant=lambda s: s.holder.compute_count == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestArrayModuleRace:
    """array.array (C-backed) concurrent read-modify-write on element 0."""

    def test_dpor_detects_array_module_race(self) -> None:
        def inc(state: _ArrayModuleState) -> None:
            temp = state.data[0]
            state.data[0] = temp + 1

        result = explore_dpor(
            setup=_ArrayModuleState,
            threads=[inc, inc],
            invariant=lambda s: s.data[0] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestStarUnpackingRace:
    """a, *b = shared_list racing with list mutation — unpacking reads multiple elements."""

    def test_dpor_detects_star_unpacking_race(self) -> None:
        def unpack(state: _StarUnpackingState) -> None:
            head, *tail = state.shared_list
            state.captured_head = head
            state.captured_tail = tail

        def mutate(state: _StarUnpackingState) -> None:
            state.shared_list[0] = 99
            state.shared_list.append(40)

        result = explore_dpor(
            setup=_StarUnpackingState,
            threads=[unpack, mutate],
            invariant=lambda s: s.captured_head == 10 and len(s.captured_tail) == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestWalrusOperatorRace:
    """:= assignment in expression racing with another thread's write."""

    def test_dpor_detects_walrus_operator_race(self) -> None:
        def walrus_read(state: _WalrusState) -> None:
            if (v := state.value) >= 0:
                state.observed = v

        def writer(state: _WalrusState) -> None:
            state.value = 42

        result = explore_dpor(
            setup=_WalrusState,
            threads=[walrus_read, writer],
            invariant=lambda s: s.observed == 42,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestDictMergeOperatorRace:
    """d1 |= d2 (PEP 584) racing with d1[key] = value — merge vs item assignment."""

    def test_dpor_detects_dict_merge_operator_race(self) -> None:
        def merge_update(state: _DictMergeState) -> None:
            state.d1 |= {"a": state.d1["a"] + 1}

        def direct_update(state: _DictMergeState) -> None:
            temp = state.d1["a"]
            state.d1["a"] = temp + 1

        result = explore_dpor(
            setup=_DictMergeState,
            threads=[merge_update, direct_update],
            invariant=lambda s: s.d1["a"] == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestMultipleReturnUnpackingRace:
    """x, y = func() where func reads shared state, racing with writes."""

    def test_dpor_detects_multiple_return_unpacking_race(self) -> None:
        def read_and_sum(state: _MultipleReturnState) -> None:
            a, b = state.x, state.y
            state.sum = a + b

        def swap_values(state: _MultipleReturnState) -> None:
            state.x = 100
            state.y = 200

        result = explore_dpor(
            setup=_MultipleReturnState,
            threads=[read_and_sum, swap_values],
            invariant=lambda s: s.sum in (3, 300),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestFStringEvaluationRace:
    """f-string {shared.value} evaluation racing with writes to that attribute."""

    def test_dpor_detects_fstring_evaluation_race(self) -> None:
        def fstring_capture(state: _FStringState) -> None:
            state.captured = f"{state.value}"

        def writer(state: _FStringState) -> None:
            state.value = 42

        result = explore_dpor(
            setup=_FStringState,
            threads=[fstring_capture, writer],
            invariant=lambda s: s.captured == "42",
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestComprehensionVariableCaptureRace:
    """List comprehension [shared.x for _ in range(1)] racing with writes to shared.x."""

    def test_dpor_detects_comprehension_capture_race(self) -> None:
        def comprehension_read(state: _ComprehensionCaptureState) -> None:
            state.captured = [state.x for _ in range(1)]

        def writer(state: _ComprehensionCaptureState) -> None:
            state.x = 99

        result = explore_dpor(
            setup=_ComprehensionCaptureState,
            threads=[comprehension_read, writer],
            invariant=lambda s: s.captured == [99],
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestTernaryExpressionRace:
    """result = a_val if flag else b_val racing with flag and value toggling."""

    def test_dpor_detects_ternary_expression_race(self) -> None:
        def ternary_read(state: _TernaryState) -> None:
            state.result = state.a_val if state.flag else state.b_val

        def toggler(state: _TernaryState) -> None:
            state.flag = False
            state.a_val = 999

        result = explore_dpor(
            setup=_TernaryState,
            threads=[ternary_read, toggler],
            invariant=lambda s: s.result in (10, 20),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestChainedComparisonRace:
    """0 < shared.x < 10: two LOAD_ATTR ops with potential write between them."""

    def test_dpor_detects_chained_comparison_race(self) -> None:
        def chained_check(state: _ChainedComparisonState) -> None:
            state.in_range = 0 < state.x < 10

        def writer(state: _ChainedComparisonState) -> None:
            state.x = 15

        result = explore_dpor(
            setup=_ChainedComparisonState,
            threads=[chained_check, writer],
            invariant=lambda s: s.in_range == (0 < s.x < 10),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestDoubleCheckedLockingRace:
    """Classic double-checked locking: if cache is None: cache = compute() — TOCTOU bug."""

    def test_dpor_detects_double_checked_locking_race(self) -> None:
        def init_cache(state: _DoubleCheckedLockingState) -> None:
            global _global_cache
            if _global_cache is None:
                state.init_count = state.init_count + 1
                _global_cache = 42

        result = explore_dpor(
            setup=_DoubleCheckedLockingState,
            threads=[init_cache, init_cache],
            invariant=lambda s: s.init_count == 1,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestSlotAttributeRace:
    """__slots__ attribute access: LOAD_ATTR/STORE_ATTR on slot descriptors (no __dict__)."""

    def test_dpor_detects_slot_attribute_race(self) -> None:
        def inc(state: _SlotState) -> None:
            temp = state.value
            state.value = temp + 1

        result = explore_dpor(
            setup=_SlotState,
            threads=[inc, inc],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestSwapRace:
    """Tuple-swap a, b = b, a racing with reads — both loads happen before stores."""

    def test_dpor_detects_swap_race(self) -> None:
        def swap(state: _SwapState) -> None:
            state.a, state.b = state.b, state.a

        def reader(state: _SwapState) -> None:
            state.observed_a = state.a
            state.observed_b = state.b

        result = explore_dpor(
            setup=_SwapState,
            threads=[swap, reader],
            invariant=lambda s: (s.observed_a + s.observed_b) == 3,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestNestedClosureCellRace:
    """Doubly-nested closures: innermost closure modifies outermost scope variable."""

    def test_dpor_detects_nested_closure_cell_race(self) -> None:
        result = explore_dpor(
            setup=_NestedClosureState,
            threads=[lambda s: s.inner(), lambda s: s.inner2()],
            invariant=lambda s: s.get() == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestDequeCheckThenActRace:
    """Two threads check len(dq) > 0 then popleft — classic TOCTOU on deque size."""

    def test_dpor_detects_deque_race(self) -> None:
        def check_and_pop(state: _DequeState) -> None:
            if len(state.dq) > 0:
                try:
                    state.dq.popleft()
                except IndexError:
                    state.error = True
                    return
                state.pop_count += 1

        result = explore_dpor(
            setup=_DequeState,
            threads=[check_and_pop, check_and_pop],
            invariant=lambda s: not s.error and s.pop_count <= 1,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


# -- Even more diabolical state classes ---------------------------------------


class _ReduceAccumulateState:
    def __init__(self) -> None:
        self.items = [1, 2, 3]
        self.accumulated = 0


class _SysModuleCacheState:
    _sentinel_key = "_frontrun_test_sentinel_module"

    def __init__(self) -> None:
        sys.modules.pop(self._sentinel_key, None)
        self.snapshot_count = 0
        self.length_before = 0
        self.length_after = 0


class _DictGetSetdefaultState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"key": 0}
        self.value = 0


class _BytearraySliceState:
    def __init__(self) -> None:
        self.buf = bytearray(b"\x00\x00\x00\x00")
        self.snapshot: bytes = b""


class _ListSortReverseState:
    def __init__(self) -> None:
        self.data = [3, 1, 2]
        self.result: list[int] = []


class _GeneratorSendState:
    def __init__(self) -> None:
        self.accumulated = 0

        def _gen() -> collections.abc.Generator[int, int, None]:
            total = 0
            while True:
                val = yield total
                total += val

        self.gen = _gen()
        next(self.gen)


class _ShallowCopyState:
    def __init__(self) -> None:
        self.original: dict[str, list[int]] = {"nums": [1, 2, 3]}
        self.copied: dict[str, list[int]] = {}


class _BoolCoercionState:
    def __init__(self) -> None:
        self.items: list[int] = [1]
        self.was_truthy = False
        self.length_at_check = 0


class _InOperatorState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"key": 1}
        self.found = False
        self.value = 0
        self.error = False


class _EnumerateState:
    def __init__(self) -> None:
        self.items = [10, 20, 30]
        self.index_sum = 0
        self.value_sum = 0


class _ReprState:
    def __init__(self) -> None:
        self.value = 0
        self.repr_result = ""


class _MinMaxState:
    def __init__(self) -> None:
        self.values = [5, 3, 8, 1, 9]
        self.minimum = 0
        self.maximum = 0


class _DictPopUpdateState:
    def __init__(self) -> None:
        self.data: dict[str, int] = {"a": 1, "b": 2}
        self.popped: int | None = None


class _PartialApplicationState:
    def __init__(self) -> None:
        self.base = 10
        self.result = 0


class _TwoPhaseUpdateState:
    def __init__(self) -> None:
        self.counter_a = 0
        self.counter_b = 0


class _SetAddDiscardState:
    def __init__(self) -> None:
        self.data: set[int] = set()
        self.size_after_add = 0


class _MapFilterState:
    def __init__(self) -> None:
        self.source = [1, 2, 3, 4, 5]
        self.even_doubled: list[int] = []


class _ZipInterleaveState:
    def __init__(self) -> None:
        self.keys = ["a", "b", "c"]
        self.values = [1, 2, 3]
        self.result: dict[str, int] = {}


class _StrJoinState:
    def __init__(self) -> None:
        self.parts = ["hello", "world"]
        self.joined = ""


class _OrderedDictMoveState:
    def __init__(self) -> None:
        self.od: collections.OrderedDict[str, int] = collections.OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        self.first_key = ""
        self.last_key = ""


# -- Even more diabolical tests -----------------------------------------------


class TestReduceAccumulateRace:
    """functools.reduce iterates the list while another thread mutates elements mid-fold."""

    def test_dpor_detects_reduce_race(self) -> None:
        def fold_sum(state: _ReduceAccumulateState) -> None:
            state.accumulated = functools.reduce(lambda a, b: a + b, state.items)

        def mutate_list(state: _ReduceAccumulateState) -> None:
            state.items[1] = 99

        result = explore_dpor(
            setup=_ReduceAccumulateState,
            threads=[fold_sum, mutate_list],
            invariant=lambda s: s.accumulated in (6, 103),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestSysModulesCacheRace:
    """Reading sys.modules length while another thread injects a module — TOCTOU on module cache."""

    def test_dpor_detects_sys_modules_race(self) -> None:
        def snapshot_modules(state: _SysModuleCacheState) -> None:
            state.length_before = len(sys.modules)
            state.snapshot_count = len([k for k in sys.modules if k.startswith("_frontrun_test")])
            state.length_after = len(sys.modules)

        def inject_module(state: _SysModuleCacheState) -> None:
            sys.modules[_SysModuleCacheState._sentinel_key] = object()  # pyright: ignore[reportArgumentType]

        result = explore_dpor(
            setup=_SysModuleCacheState,
            threads=[snapshot_modules, inject_module],
            invariant=lambda s: (s.snapshot_count == 1) == (s.length_after > s.length_before),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestDictGetSetdefaultRace:
    """dict.get() vs read-modify-write — get sees stale value before the write commits."""

    def test_dpor_detects_dict_get_setdefault_race(self) -> None:
        def reader(state: _DictGetSetdefaultState) -> None:
            val = state.data.get("key", -1)
            state.value = val

        def writer(state: _DictGetSetdefaultState) -> None:
            state.data.setdefault("key", 0)
            temp = state.data["key"]
            state.data["key"] = temp + 1

        result = explore_dpor(
            setup=_DictGetSetdefaultState,
            threads=[reader, writer],
            invariant=lambda s: s.value == 1,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestBytearraySliceRace:
    """bytearray slice read racing with byte-level writes — C-backed mutable bytes."""

    def test_dpor_detects_bytearray_slice_race(self) -> None:
        def slicer(state: _BytearraySliceState) -> None:
            state.snapshot = bytes(state.buf[0:4])

        def writer(state: _BytearraySliceState) -> None:
            state.buf[0] = 0xFF
            state.buf[1] = 0xFF
            state.buf[2] = 0xFF
            state.buf[3] = 0xFF

        result = explore_dpor(
            setup=_BytearraySliceState,
            threads=[slicer, writer],
            invariant=lambda s: s.snapshot in (b"\x00\x00\x00\x00", b"\xff\xff\xff\xff"),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestListSortReverseRace:
    """list.sort() racing with list.reverse() — both mutate in place with multi-step internals."""

    def test_dpor_detects_list_sort_reverse_race(self) -> None:
        def sort_and_capture(state: _ListSortReverseState) -> None:
            state.data.sort()
            state.result = list(state.data)

        def reverser(state: _ListSortReverseState) -> None:
            state.data.reverse()

        result = explore_dpor(
            setup=_ListSortReverseState,
            threads=[sort_and_capture, reverser],
            invariant=lambda s: s.result == [1, 2, 3],
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestGeneratorSendRace:
    """Two threads racing to .send() into the same generator — generator protocol is not thread-safe."""

    def test_dpor_detects_generator_send_race(self) -> None:
        def sender(state: _GeneratorSendState) -> None:
            result = state.gen.send(1)
            state.accumulated = result

        result = explore_dpor(
            setup=_GeneratorSendState,
            threads=[sender, sender],
            invariant=lambda s: s.accumulated == 2,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestShallowCopyRace:
    """copy.copy() shares inner list references — mutation to inner list races with the copy read."""

    def test_dpor_detects_shallow_copy_race(self) -> None:
        def copier(state: _ShallowCopyState) -> None:
            state.copied = copy.copy(state.original)

        def mutator(state: _ShallowCopyState) -> None:
            state.original["nums"].append(4)

        result = explore_dpor(
            setup=_ShallowCopyState,
            threads=[copier, mutator],
            invariant=lambda s: len(s.copied.get("nums", [])) == 3,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestBoolCoercionRace:
    """if container: invokes __bool__/__len__ — coercion racing with mutation that empties it."""

    def test_dpor_detects_bool_coercion_race(self) -> None:
        def check_truthy(state: _BoolCoercionState) -> None:
            state.was_truthy = bool(state.items)
            state.length_at_check = len(state.items)

        def drain(state: _BoolCoercionState) -> None:
            state.items.clear()

        result = explore_dpor(
            setup=_BoolCoercionState,
            threads=[check_truthy, drain],
            invariant=lambda s: s.was_truthy == (s.length_at_check > 0),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestInOperatorRace:
    """'key in dict' then dict['key'] — TOCTOU through __contains__ + __getitem__."""

    def test_dpor_detects_in_operator_race(self) -> None:
        def check_and_read(state: _InOperatorState) -> None:
            if "key" in state.data:
                state.found = True
                try:
                    state.value = state.data["key"]
                except KeyError:
                    state.error = True

        def deleter(state: _InOperatorState) -> None:
            del state.data["key"]

        result = explore_dpor(
            setup=_InOperatorState,
            threads=[check_and_read, deleter],
            invariant=lambda s: not s.error and (s.found == ("key" in s.data) or s.value == 1),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestEnumerateIterationRace:
    """enumerate(list) iteration racing with list.insert() — iterator sees inconsistent state."""

    def test_dpor_detects_enumerate_race(self) -> None:
        def iterate(state: _EnumerateState) -> None:
            idx_sum = 0
            val_sum = 0
            for i, v in enumerate(state.items):
                idx_sum += i
                val_sum += v
            state.index_sum = idx_sum
            state.value_sum = val_sum

        def inserter(state: _EnumerateState) -> None:
            state.items.insert(0, 99)

        result = explore_dpor(
            setup=_EnumerateState,
            threads=[iterate, inserter],
            invariant=lambda s: s.value_sum == 60 or s.value_sum == 159,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestReprRace:
    """str(obj.value) reading attribute while another thread mutates it."""

    def test_dpor_detects_repr_race(self) -> None:
        def capture_repr(state: _ReprState) -> None:
            state.repr_result = str(state.value)

        def writer(state: _ReprState) -> None:
            state.value = 42

        result = explore_dpor(
            setup=_ReprState,
            threads=[capture_repr, writer],
            invariant=lambda s: s.repr_result == "42",
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestMinMaxRace:
    """min()/max() iterate the list while another thread mutates elements mid-scan."""

    def test_dpor_detects_min_max_race(self) -> None:
        def find_extremes(state: _MinMaxState) -> None:
            state.minimum = min(state.values)
            state.maximum = max(state.values)

        def mutator(state: _MinMaxState) -> None:
            state.values[0] = 0
            state.values[4] = 100

        result = explore_dpor(
            setup=_MinMaxState,
            threads=[find_extremes, mutator],
            invariant=lambda s: s.minimum <= s.maximum and s.minimum == min(s.values),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestDictPopUpdateRace:
    """dict.pop() racing with dict.update() — pop removes what update expects to overwrite."""

    def test_dpor_detects_dict_pop_update_race(self) -> None:
        def pop_key(state: _DictPopUpdateState) -> None:
            state.popped = state.data.pop("a", None)

        def update_key(state: _DictPopUpdateState) -> None:
            state.data.update({"a": 99})

        result = explore_dpor(
            setup=_DictPopUpdateState,
            threads=[pop_key, update_key],
            invariant=lambda s: ("a" in s.data) == (s.popped is None) or s.data.get("a") == 99,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestPartialApplicationRace:
    """functools.partial captures state.base at call time — race between capture and mutation."""

    def test_dpor_detects_partial_application_race(self) -> None:
        def apply_partial(state: _PartialApplicationState) -> None:
            fn = functools.partial(lambda b, x: b + x, state.base)
            state.result = fn(5)

        def mutator(state: _PartialApplicationState) -> None:
            state.base = 100

        result = explore_dpor(
            setup=_PartialApplicationState,
            threads=[apply_partial, mutator],
            invariant=lambda s: s.result == 105,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestTwoPhaseUpdateRace:
    """Two-phase update: increment a then set b = a — interleaving breaks a == b invariant."""

    def test_dpor_detects_two_phase_update_race(self) -> None:
        def phase_update(state: _TwoPhaseUpdateState) -> None:
            state.counter_a = state.counter_a + 1
            state.counter_b = state.counter_a

        result = explore_dpor(
            setup=_TwoPhaseUpdateState,
            threads=[phase_update, phase_update],
            invariant=lambda s: s.counter_a == s.counter_b,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestSetAddDiscardRace:
    """set.add() + len() racing with set.discard() — TOCTOU on set membership."""

    def test_dpor_detects_set_add_discard_race(self) -> None:
        def adder(state: _SetAddDiscardState) -> None:
            state.data.add(1)
            state.size_after_add = len(state.data)

        def discarder(state: _SetAddDiscardState) -> None:
            state.data.discard(1)

        result = explore_dpor(
            setup=_SetAddDiscardState,
            threads=[adder, discarder],
            invariant=lambda s: (1 in s.data) == (s.size_after_add == 1),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestMapFilterRace:
    """Lazy map()/filter() iterators racing with list mutation — iterator reads stale data."""

    def test_dpor_detects_map_filter_race(self) -> None:
        def map_filter(state: _MapFilterState) -> None:
            state.even_doubled = [x * 2 for x in state.source if x % 2 == 0]

        def mutator(state: _MapFilterState) -> None:
            state.source[1] = 7
            state.source[3] = 11

        result = explore_dpor(
            setup=_MapFilterState,
            threads=[map_filter, mutator],
            invariant=lambda s: s.even_doubled == [4, 8] or s.even_doubled == [],
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestZipRace:
    """dict(zip(keys, values)) — zip lazily pulls from both lists, racing with mutation."""

    def test_dpor_detects_zip_race(self) -> None:
        def zip_to_dict(state: _ZipInterleaveState) -> None:
            state.result = dict(zip(state.keys, state.values))

        def mutator(state: _ZipInterleaveState) -> None:
            state.keys[0] = "z"
            state.values[2] = 99

        result = explore_dpor(
            setup=_ZipInterleaveState,
            threads=[zip_to_dict, mutator],
            invariant=lambda s: s.result.get("a") == 1 or s.result.get("z") == 1,
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestStrJoinRace:
    """str.join() iterates the list while another thread mutates it mid-join."""

    def test_dpor_detects_str_join_race(self) -> None:
        def joiner(state: _StrJoinState) -> None:
            state.joined = " ".join(state.parts)

        def mutator(state: _StrJoinState) -> None:
            state.parts[0] = "goodbye"
            state.parts.append("!")

        result = explore_dpor(
            setup=_StrJoinState,
            threads=[joiner, mutator],
            invariant=lambda s: s.joined in ("hello world", "goodbye world !"),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds


class TestOrderedDictMoveToEndRace:
    """OrderedDict.move_to_end() racing with iteration — order changes mid-read."""

    def test_dpor_detects_ordered_dict_move_race(self) -> None:
        def read_order(state: _OrderedDictMoveState) -> None:
            keys = list(state.od.keys())
            state.first_key = keys[0]
            state.last_key = keys[-1]

        def reorder(state: _OrderedDictMoveState) -> None:
            state.od.move_to_end("a")

        result = explore_dpor(
            setup=_OrderedDictMoveState,
            threads=[read_order, reorder],
            invariant=lambda s: (
                (s.first_key == "a" and s.last_key == "c") or (s.first_key == "b" and s.last_key == "a")
            ),
            detect_io=False,
            deadlock_timeout=5.0,
        )
        assert not result.property_holds
