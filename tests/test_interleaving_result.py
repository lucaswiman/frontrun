"""Public result assertions and representations, including marker schedules."""

import re

import pytest

from frontrun.common import InterleavingResult, Schedule, Step


@pytest.mark.parametrize("prefix", ["", "counter_test: "])
def test_assert_holds_returns_none_on_success(prefix):
    assert InterleavingResult(property_holds=True, num_explored=5).assert_holds(msg_prefix=prefix) is None


@pytest.mark.parametrize("explanation", [None, "Race on counter."])
@pytest.mark.parametrize("prefix", ["", "counter_test: "])
def test_assert_holds_failure_message(explanation, prefix):
    result = InterleavingResult(property_holds=False, explanation=explanation)
    with pytest.raises(AssertionError, match=f"^{re.escape(prefix + (explanation or ''))}$"):
        result.assert_holds(msg_prefix=prefix)


def test_assert_holds_default_prefix():
    with pytest.raises(AssertionError, match="^Race on counter\\.$"):
        InterleavingResult(property_holds=False, explanation="Race on counter.").assert_holds()


@pytest.mark.parametrize(
    ("counterexample", "expected"),
    [
        pytest.param(None, "None", id="none"),
        pytest.param([0, 1, 0, 1, 0], "[0, 1, 0, 1, 0]", id="short-list"),
        pytest.param(list(range(20)), "20 steps", id="long-list"),
        pytest.param(Schedule([Step("t1", "a"), Step("t2", "b")]), "Schedule(", id="short-marker-schedule"),
        pytest.param(Schedule([Step(f"t{i % 2}", f"m{i}") for i in range(20)]), "Schedule(", id="long-marker-schedule"),
    ],
)
def test_repr_supports_each_counterexample_shape(counterexample, expected):
    holds = counterexample is None
    result = InterleavingResult(property_holds=holds, counterexample=counterexample, num_explored=5)
    rendered = repr(result)
    assert f"property_holds={holds}" in rendered
    assert "num_explored=5" in rendered
    assert expected in rendered
