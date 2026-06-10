"""SQL parameter resolution for all five PEP 249 paramstyles.

Substitutes placeholder markers (``?``, ``:name``, ``%s``, etc.) with SQL
literal representations of the corresponding Python values.  The resolved
SQL is used only for AST analysis (predicate extraction) — it is never
sent to a database.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------


def _python_to_sql_literal(value: Any) -> str:
    """Convert a Python value to a SQL literal for AST analysis.

    The result must be syntactically valid SQL that sqlglot can parse.
    It does NOT need to be safe for execution.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"X'{bytes(value).hex()}'"
    # Default: string literal (double internal quotes)
    s = str(value).replace("'", "''")
    return f"'{s}'"


# ---------------------------------------------------------------------------
# Precompiled placeholder patterns
# ---------------------------------------------------------------------------

# Negative lookbehinds prevent matching escaped sequences
# (:: in PG casts, %% in psycopg2 literal-percent escapes).
_RE_QMARK = re.compile(r"\?")
_RE_NUMERIC = re.compile(r"(?<!:):(\d+)")
_RE_NAMED = re.compile(r"(?<!:):([A-Za-z_]\w*)")
_RE_FORMAT = re.compile(r"(?<!%)%s")
_RE_PYFORMAT = re.compile(r"(?<!%)%\(([^)]+)\)s")
_RE_DOLLAR = re.compile(r"\$(\d+)")


# ---------------------------------------------------------------------------
# Literal-aware substitution
# ---------------------------------------------------------------------------


def _split_literal_segments(sql: str) -> list[tuple[str, bool]]:
    """Split *sql* into ``(text, is_literal)`` segments.

    Single-quoted string literals (with ``''`` escape) are tagged
    ``is_literal=True`` so that placeholder substitution can skip them: a
    ``?`` or ``%s`` inside a string literal is data, not a placeholder
    (finding 7).  Everything outside a literal is tagged ``is_literal=False``.
    """
    segments: list[tuple[str, bool]] = []
    i = 0
    n = len(sql)
    start = 0
    in_literal = False
    while i < n:
        ch = sql[i]
        if not in_literal:
            if ch == "'":
                # Flush the non-literal run, then start a literal at the quote.
                if i > start:
                    segments.append((sql[start:i], False))
                start = i
                in_literal = True
            i += 1
        else:
            if ch == "'":
                # ``''`` is an escaped quote inside the literal, not a close.
                if i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                # Closing quote: emit the literal (including both quotes).
                i += 1
                segments.append((sql[start:i], True))
                start = i
                in_literal = False
            else:
                i += 1
    if start < n:
        # Trailing run; an unterminated literal is still treated as a literal
        # so we never substitute inside a half-open string.
        segments.append((sql[start:n], in_literal))
    return segments


def _sub_outside_literals(pattern: re.Pattern[str], replacer: Any, sql: str) -> str:
    """Apply ``pattern.sub(replacer, ...)`` only outside single-quoted literals.

    *replacer* is invoked in left-to-right order across the non-literal
    segments, so stateful positional replacers (which advance an index per
    match) see placeholders in source order while ignoring those embedded in
    string literals.
    """
    out: list[str] = []
    for text, is_literal in _split_literal_segments(sql):
        if is_literal:
            out.append(text)
        else:
            out.append(pattern.sub(replacer, text))
    return "".join(out)


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------


def resolve_parameters(sql: str, parameters: Any, paramstyle: str) -> str:
    """Substitute parameter placeholders with SQL literal values.

    Supports all five PEP 249 paramstyles plus '$1' style.
    Returns the original SQL unchanged if resolution fails.
    """
    if parameters is None:
        return sql
    try:
        if paramstyle == "qmark":
            return _resolve_positional(sql, parameters, _RE_QMARK)
        if paramstyle == "numeric":
            return _resolve_numeric(sql, parameters, _RE_NUMERIC)
        if paramstyle == "dollar":
            return _resolve_numeric(sql, parameters, _RE_DOLLAR)
        if paramstyle == "named":
            return _resolve_named(sql, parameters)
        if paramstyle in ("format", "pyformat"):
            # Detect actual style from parameter type:
            # dict → %(name)s, sequence → %s
            if isinstance(parameters, dict):
                return _resolve_pyformat(sql, parameters)
            return _resolve_positional(sql, parameters, _RE_FORMAT)
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    return sql


def _resolve_positional(sql: str, parameters: Any, pattern: re.Pattern[str]) -> str:
    """Replace positional placeholders (? or %s) in left-to-right order."""
    params = tuple(parameters)
    idx = 0

    def replacer(m: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(params):
            return m.group(0)  # more placeholders than params → leave as-is
        val = _python_to_sql_literal(params[idx])
        idx += 1
        return val

    return _sub_outside_literals(pattern, replacer, sql)


def _resolve_numeric(sql: str, parameters: Any, pattern: re.Pattern[str]) -> str:
    """Replace :N or $N placeholders (1-based index)."""
    params = tuple(parameters)

    def replacer(m: re.Match[str]) -> str:
        idx = int(m.group(1)) - 1
        if idx < 0:
            raise IndexError("placeholder index must be >= 1")
        return _python_to_sql_literal(params[idx])

    return _sub_outside_literals(pattern, replacer, sql)


def _resolve_named(sql: str, parameters: Any) -> str:
    """Replace :name placeholders with named parameters."""
    params = dict(parameters)

    def replacer(m: re.Match[str]) -> str:
        return _python_to_sql_literal(params[m.group(1)])

    return _sub_outside_literals(_RE_NAMED, replacer, sql)


def _resolve_pyformat(sql: str, parameters: Any) -> str:
    """Replace %(name)s placeholders with named parameters."""
    params = dict(parameters)

    def replacer(m: re.Match[str]) -> str:
        return _python_to_sql_literal(params[m.group(1)])

    return _sub_outside_literals(_RE_PYFORMAT, replacer, sql)
