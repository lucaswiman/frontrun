"""Shared file-filtering logic for bytecode tracing.

Both ``bytecode.py`` (random exploration) and ``dpor.py`` (systematic DPOR)
need to distinguish user code from library/stdlib code.  This module
provides a single implementation so the filtering rules stay in sync.

The default behaviour skips stdlib, site-packages, and frontrun internals.
Users can widen the filter to include specific installed packages by
passing ``trace_packages`` patterns to the exploration APIs.  Patterns
use :func:`fnmatch.fnmatch` syntax (e.g. ``"django_*"``) and are matched
against the **module name** derived from the file path.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
import threading
from collections.abc import Sequence

# Directories to never trace into (stdlib, site-packages)
_SKIP_DIRS: frozenset[str] = frozenset(p for p in sys.path if "lib/python" in p or "site-packages" in p)

_THREADING_FILE = threading.__file__

# Skip the entire frontrun package directory
_FRONTRUN_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

# Regex matching CPython ABI tag suffixes on C extension filenames,
# e.g. ".cpython-314-x86_64-linux-gnu" before the final ".so"/".pyd".
_CPYTHON_ABI_RE = re.compile(r"\.cpython-\d+[^.]*")


def _filename_to_module(filename: str) -> str | None:
    """Convert a filename to a dotted module name, or None if not determinable.

    Strips the site-packages prefix and ``.py`` suffix, then converts
    path separators to dots.  Returns ``None`` for non-``.py`` files or
    files outside site-packages.
    """
    # Longest prefix wins: on stock layouts a stdlib dir and its site-packages
    # child are both in _SKIP_DIRS, so a file can match two prefixes.  Sorting
    # by length (descending) makes the mapping deterministic and picks the most
    # specific package name — iterating the frozenset directly would depend on
    # PYTHONHASHSEED and silently break trace_packages filtering run-to-run.
    for skip_dir in sorted(_SKIP_DIRS, key=len, reverse=True):
        if filename.startswith(skip_dir):
            rel = filename[len(skip_dir) :]
            if rel.startswith(os.sep):
                rel = rel[1:]
            if rel.endswith(os.sep + "__init__.py"):
                rel = rel[: -(len(os.sep) + len("__init__.py"))]
            elif rel.endswith(".py"):
                rel = rel[:-3]
            else:
                # C extension: strip ABI tag + final extension
                # e.g. "_sqlite3.cpython-314-x86_64-linux-gnu.so" -> "_sqlite3"
                rel = _CPYTHON_ABI_RE.sub("", rel)
                dot = rel.rfind(".")
                if dot != -1:
                    rel = rel[:dot]
            return rel.replace(os.sep, ".")
    return None


class TraceFilter:
    """Configurable filter deciding which files should be traced.

    The default filter (``trace_packages=None``) traces only user code:
    files outside stdlib, site-packages, and frontrun internals.

    When ``trace_packages`` is provided, files in site-packages whose
    module names match any of the given patterns are **also** traced.
    Patterns use :func:`fnmatch.fnmatch` syntax (e.g. ``"django_*"``,
    ``"mylib.*"``).  Note that ``*`` in fnmatch matches any characters
    including dots, so ``"django_*"`` matches both ``django_filters``
    and ``django_filters.views``.

    Example::

        # Trace user code + any django_* package + myapp.utils
        filt = TraceFilter(trace_packages=["django_*", "myapp.*"])
        filt.should_trace_file("/path/to/site-packages/django_filters/views.py")
        # -> True
    """

    def __init__(self, trace_packages: Sequence[str] | None = None) -> None:
        if trace_packages is not None and len(trace_packages) > 0:
            # Compile fnmatch patterns to regexes for speed
            combined = "|".join(fnmatch.translate(p) for p in trace_packages)
            self._pattern: re.Pattern[str] | None = re.compile(combined)
        else:
            self._pattern = None

    def should_trace_file(self, filename: str) -> bool:
        """Check whether *filename* should be traced."""
        if filename == _THREADING_FILE:
            return False
        if filename.startswith("<frozen"):
            return False
        if filename.startswith(_FRONTRUN_DIR):
            return False

        # Check against skip dirs (stdlib / site-packages)
        in_skip_dir = False
        for skip_dir in _SKIP_DIRS:
            if filename.startswith(skip_dir):
                in_skip_dir = True
                break

        if not in_skip_dir:
            return True  # user code — always trace

        # File is in a skip dir.  Allow it only if it matches a trace_packages pattern.
        if self._pattern is not None:
            module_name = _filename_to_module(filename)
            if module_name is not None and self._pattern.match(module_name):
                return True

        return False


_DEFAULT_FILTER = TraceFilter()

# ---- active filter (process-wide) ----
#
# The filter is a plain module-level global, NOT a threading.local.
# Worker threads spawned by frontrun.explore call
# should_trace_file() from their sys.settrace/sys.monitoring callbacks
# and must see the filter installed by the main thread.  The main thread
# always sets and clears the filter outside the worker-thread window, so
# there is no concurrent mutation.

_active_filter_lock = threading.Lock()
_active_filter: TraceFilter | None = None


def set_active_trace_filter(filt: TraceFilter | None) -> None:
    """Set the active trace filter for the entire process.

    Pass ``None`` to reset to the default filter.
    """
    global _active_filter  # noqa: PLW0603
    with _active_filter_lock:
        _active_filter = filt


def get_active_trace_filter() -> TraceFilter:
    """Return the currently active trace filter."""
    return _active_filter or _DEFAULT_FILTER


def should_trace_file(filename: str) -> bool:
    """Check whether a file is user code that should be traced.

    Delegates to the currently active :class:`TraceFilter`.
    """
    return get_active_trace_filter().should_trace_file(filename)


def is_dynamic_code(filename: str) -> bool:
    """Check whether a filename indicates dynamically generated code.

    Returns True for filenames like ``<string>``, ``<generated>``, ``<stdin>``
    that are produced by ``exec()``, ``compile()``, or interactive mode.
    Does NOT match ``<frozen ...>`` (already excluded by ``should_trace_file``).
    """
    return filename.startswith("<")


def _is_cmdline_mode() -> bool:
    """Return True if the program came from ``python -c`` or stdin.

    ``python -c`` leaves ``__main__`` without ``__file__``; a program piped
    via stdin (``python -`` / ``python < script.py``) sets ``__file__`` to
    the pseudo-filename ``"<stdin>"``.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return False
    return not hasattr(main, "__file__") or getattr(main, "__file__", None) == "<stdin>"


def is_cmdline_user_code(filename: str, f_globals: dict[str, object]) -> bool:
    """Check whether *filename* is user code from ``python -c`` or stdin.

    When Python is started with ``-c`` (or the program is piped via stdin /
    typed at the REPL), user-defined functions have ``co_filename ==
    "<string>"`` (``"<stdin>"`` for stdin) and their ``f_globals`` (or
    ``__globals__``) is ``__main__.__dict__``.  This function returns
    ``True`` for exactly that combination, allowing the trace callbacks
    to treat such code as user code rather than library-exec'd code.

    Returns ``False`` for other filenames, for ``<string>``/``<stdin>``
    code whose globals don't belong to ``__main__``, or when the process
    was started as a normal script (``__main__`` has a ``__file__``).
    """
    if filename not in ("<string>", "<stdin>"):
        return False
    if not _is_cmdline_mode():
        return False
    main = sys.modules.get("__main__")
    if main is None:
        return False
    return f_globals is main.__dict__
