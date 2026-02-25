"""Shared file-filtering logic for bytecode tracing.

Both ``bytecode.py`` (random exploration) and ``dpor.py`` (systematic DPOR)
need to distinguish user code from library/stdlib code.  This module
provides a single implementation so the filtering rules stay in sync.
"""

import os
import sys
import threading

# Directories to never trace into (stdlib, site-packages)
_SKIP_DIRS: frozenset[str] = frozenset(p for p in sys.path if "lib/python" in p or "site-packages" in p)

_THREADING_FILE = threading.__file__

# Skip the entire frontrun package directory
_FRONTRUN_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def should_trace_file(filename: str) -> bool:
    """Check whether a file is user code that should be traced."""
    if filename == _THREADING_FILE:
        return False
    # Skip frozen stdlib modules ("<frozen importlib._bootstrap>" etc.) but
    # allow exec/eval/compile code ("<string>", "<generated>", etc.).
    if filename.startswith("<frozen"):
        return False
    if filename.startswith(_FRONTRUN_DIR):
        return False
    for skip_dir in _SKIP_DIRS:
        if filename.startswith(skip_dir):
            return False
    return True
