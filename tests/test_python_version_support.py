"""Guard the supported-Python floor (3.11+).

Python 3.10 reached end of life, so frontrun requires 3.11+.  These tests
pin the packaging metadata to that floor and make sure the opcode shadow
stack doesn't keep dead handlers for opcodes that CPython removed in 3.11
(they can never fire on a supported interpreter, and keeping them obscures
which stack layouts the observer actually has to model).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import frontrun._opcode_observer as opcode_observer

_ROOT = Path(__file__).resolve().parent.parent

# Opcodes that existed on 3.10 and were removed in 3.11 (bpo-45292 / the 3.11
# specializing interpreter work).  None of these exist on any supported version.
_REMOVED_IN_3_11 = frozenset(
    {
        "DUP_TOP",
        "DUP_TOP_TWO",
        "ROT_TWO",
        "ROT_THREE",
        "ROT_FOUR",
        "ROT_N",
        "CALL_FUNCTION",
        "CALL_FUNCTION_KW",
        "CALL_METHOD",
        "GEN_START",
        "SETUP_WITH",
        "JUMP_ABSOLUTE",
        "BINARY_ADD",
        "INPLACE_ADD",
    }
)


def _pyproject() -> dict[str, object]:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_requires_python_floor_is_3_11() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    assert project["requires-python"] == ">=3.11"
    classifiers = project["classifiers"]
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers


def test_ruff_targets_3_11() -> None:
    tool = _pyproject()["tool"]
    assert isinstance(tool, dict)
    assert tool["ruff"]["target-version"] == "py311"


def test_opcode_observer_has_no_3_10_only_opcode_handlers() -> None:
    source = Path(opcode_observer.__file__).read_text()
    quoted = set(re.findall(r'"([A-Z][A-Z0-9_]+)"', source))
    assert not (quoted & _REMOVED_IN_3_11)
    # The 3.10 catch-all for INPLACE_*/BINARY_* arithmetic opcodes is dead too:
    # every BINARY_* opcode on 3.11+ has an explicit handler.
    assert '"INPLACE_"' not in source
