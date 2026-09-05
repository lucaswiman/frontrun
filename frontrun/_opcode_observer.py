"""Opcode-level shadow stack and shared-memory access detection.

Mirrors CPython's evaluation stack at bytecode granularity to identify
which Python objects are being read or written by each opcode.  Reports
detected accesses to the Rust DPOR engine for systematic interleaving
exploration.

This module is used by both ``dpor.py`` (sync threads) and
``async_dpor.py`` (async tasks).  It is deliberately decoupled from the
scheduler implementations — the only requirement is that the caller
provides an object satisfying the narrow interface used by
``_process_opcode`` (engine, execution, lock, shadow stacks, stable IDs).
"""

from __future__ import annotations

import _io as _io_module
import builtins as _builtins_mod
import dis
import functools as _functools_mod
import operator as _operator_mod
import sys
import threading
import types
from collections import deque
from collections.abc import Callable
from typing import Any, Literal

from frontrun._cooperative import CooperativeLock as _CooperativeLock
from frontrun._cooperative import real_lock
from frontrun._tracing import is_cmdline_user_code as _is_cmdline_user_code
from frontrun._tracing import is_dynamic_code as _is_dynamic_code
from frontrun._tracing import should_trace_file as _should_trace_file

_PY_VERSION = sys.version_info[:2]

# sys.monitoring (PEP 669) is available since 3.12 and is required for
# free-threaded builds (3.13t/3.14t) where sys.settrace + f_trace_opcodes
# has a known crash bug (CPython #118415).
_USE_SYS_MONITORING = _PY_VERSION >= (3, 12)

__all__ = [
    "OpcodeTraceHandle",
    "ShadowStack",
    "StableObjectIds",
    "_CALL_OPCODES",
    "_SHARED_ACCESS_OPCODES",
    "_USE_SYS_MONITORING",
    "_call_might_report_access",
    "_get_instructions",
    "_is_shared_opcode",
    "_make_object_key",
    "_process_opcode",
    "clear_instr_cache",
    "get_object_key_reverse_map",
    "install_thread_line_trace",
    "install_thread_opcode_trace",
    "make_monitoring_callbacks",
    "make_settrace_callback",
    "process_opcode_with_coarsening",
    "set_object_key_reverse_map",
    "setup_opcode_monitoring",
    "start_opcode_trace",
    "stop_opcode_trace",
    "teardown_opcode_monitoring",
    "uninstall_thread_line_trace",
    "uninstall_thread_opcode_trace",
]


# ---------------------------------------------------------------------------
# Shadow Stack for shared-access detection
# ---------------------------------------------------------------------------


class ShadowStack:
    """Mirrors CPython's evaluation stack to track object identity.

    When LOAD_ATTR/STORE_ATTR execute, we peek at our shadow stack
    to identify which object is being accessed.
    """

    __slots__ = ("stack",)

    def __init__(self) -> None:
        self.stack: list[Any] = []

    def push(self, val: Any) -> None:
        self.stack.append(val)

    def pop(self) -> Any:
        return self.stack.pop() if self.stack else None

    def peek(self, n: int = 0) -> Any:
        idx = -(n + 1)
        return self.stack[idx] if abs(idx) <= len(self.stack) else None

    def clear(self) -> None:
        self.stack.clear()


# ---------------------------------------------------------------------------
# Instruction cache
# ---------------------------------------------------------------------------

# Pre-analyzed instruction cache: code object -> {offset -> instruction}.
#
# Keyed by the code object itself (not ``id(code)``).  Using the code
# object as the dict key keeps a strong reference, which prevents the
# object from being garbage-collected while cached.  This eliminates the
# stale-cache bug where a GC'd code object's id was reused by a new one
# within a single DPOR execution.
_INSTR_CACHE: dict[Any, dict[int, dis.Instruction]] = {}
# Must use real_lock() — not threading.Lock() — to avoid cooperative re-entry
# when the pytest plugin patches threading.Lock globally.
_INSTR_CACHE_LOCK = real_lock()


def _get_instructions(code: Any) -> dict[int, dis.Instruction]:
    """Get a mapping from byte offset to Instruction for a code object."""
    # Fast path: already cached (safe to read without lock on GIL builds;
    # on free-threaded builds dict reads are internally locked)
    cached = _INSTR_CACHE.get(code)
    if cached is not None:
        return cached
    with _INSTR_CACHE_LOCK:
        # Double-check after acquiring lock
        if code in _INSTR_CACHE:
            return _INSTR_CACHE[code]
        mapping = {}
        for instr in dis.get_instructions(code):
            mapping[instr.offset] = instr
        _INSTR_CACHE[code] = mapping
        return mapping


def clear_instr_cache() -> None:
    """Clear the instruction cache (called between DPOR executions)."""
    with _INSTR_CACHE_LOCK:
        _INSTR_CACHE.clear()


# ---------------------------------------------------------------------------
# Type classification for opcode processing
# ---------------------------------------------------------------------------

# C-level method access classification.
#
# Python bytecode tracing can't see inside C method calls — list.append(),
# set.add(), dict.update(), etc. all execute opaquely.  We detect these in the
# CALL handler by checking if the callable is a builtin_function_or_method
# (i.e. bound to a C-implemented object) and classifying the call as a read or
# write based on the method name.
#
# Design: immutable types are excluded entirely (calling str.upper() can't
# cause a data race).  For mutable types, known read-only methods report READ;
# everything else defaults to WRITE.
_BUILTIN_METHOD_TYPE = type(len)  # builtin_function_or_method
_WRAPPER_DESCRIPTOR_TYPE = type(object.__setattr__)  # wrapper_descriptor
_METHOD_WRAPPER_TYPE = type("".__str__)  # method-wrapper

_IMMUTABLE_TYPES = (str, bytes, int, float, bool, complex, tuple, frozenset, range, type(None), types.ModuleType)

# File-backed I/O types whose conflicts are tracked by the I/O detection layer
# (LD_PRELOAD + _traced_open) keyed by file path.  DPOR skips Python-level
# tracking on these because: (1) each thread typically creates its own handle,
# (2) id() reuse can cause stable_id collisions between short-lived instances,
# and (3) the real conflicts are captured at the file-path level.
#
# NOTE: StringIO and BytesIO are intentionally EXCLUDED — they are in-memory
# buffers with no file path, so LD_PRELOAD never sees them.  When shared
# between threads they need normal Python-level conflict tracking.
_IO_WRAPPER_TYPES: tuple[type, ...] = (
    _io_module.TextIOWrapper,
    _io_module.BufferedWriter,
    _io_module.BufferedReader,
    _io_module.BufferedRandom,
    _io_module.BufferedRWPair,
    _io_module.FileIO,
)

# Types whose real I/O conflicts are tracked by a higher-level reporter
# (Redis key-level, SQL table-level).  Python-level attribute/method
# tracking on these objects creates false races due to id() reuse on
# short-lived per-thread instances (same issue as _IO_WRAPPER_TYPES).
# Also suppress cooperative lock wrapper — its synchronization semantics
# are fully captured by report_sync(); Python-level attribute writes
# (_owner_thread_id) are redundant and create false wakeups.
_io_client_types: list[type] = [
    # socket objects: per-connection, tracked by I/O detection layer
    __import__("socket").socket,
    # Cooperative lock: tracked by report_sync, not Python attributes
    _CooperativeLock,
]
try:
    import redis as _redis_module  # type: ignore[import-untyped]
    import redis.asyncio as _aioredis_module  # type: ignore[import-untyped]
    import redis.connection as _redis_conn_module  # type: ignore[import-untyped]

    _io_client_types.extend(
        [
            _redis_module.Redis,
            _redis_module.StrictRedis,
            _aioredis_module.Redis,
            _redis_conn_module.ConnectionPool,
            _redis_conn_module.Connection,
        ]
    )
except ImportError:
    pass
_db_cursor_types: list[type] = [__import__("sqlite3").Cursor]
try:
    import psycopg2.extensions as _psycopg2_ext  # type: ignore[import-untyped]

    _io_client_types.extend([_psycopg2_ext.connection, _psycopg2_ext.cursor])
    _db_cursor_types.append(_psycopg2_ext.cursor)
except ImportError:
    pass
try:
    import sqlalchemy.engine as _sa_engine  # type: ignore[import-untyped]

    _io_client_types.append(_sa_engine.Connection)
except ImportError:
    pass
_IO_CLIENT_TYPES: tuple[type, ...] = tuple(_io_client_types)
del _io_client_types

# DB cursor types: tracked at Python object level even though they are also
# I/O client types.  Two threads sharing a single cursor is a real race
# (the cursor's internal result buffer is clobbered), but the SQL-level
# reporter only tracks table/row granularity and misses it.  Per-thread
# cursors are different objects, so tracking them doesn't false-conflict.
_DB_CURSOR_TYPES: tuple[type, ...] = tuple(_db_cursor_types)
del _db_cursor_types

# C-level methods that are read-only (don't mutate the object).
_C_METHOD_READ_ONLY = frozenset(
    {
        # Lookup / iteration (common to multiple container types)
        "__contains__",
        "__getitem__",
        "__getattribute__",
        "__len__",
        "__iter__",
        "__reversed__",
        # list / tuple
        "count",
        "index",
        # dict
        "get",
        "keys",
        "values",
        "items",
        # set
        "issubset",
        "issuperset",
        "isdisjoint",
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        # copy
        "copy",
        "__copy__",
    }
)

# C-level methods on immutable types that iterate their FIRST ARGUMENT.
# The method's __self__ is immutable (e.g. str for str.join), so the standard
# C-method handler skips them.  We detect these by name and report a READ on
# the first argument instead.
_IMMUTABLE_SELF_ARG_READERS = frozenset({"join"})

# Type constructors that iterate their first argument (read it).
# These are `type` objects (list, dict, bytes, etc.) called as constructors.
# The CALL handler needs to report a READ on the first argument when one of
# these types is called.
_CONTAINER_CONSTRUCTORS: frozenset[type] = frozenset(
    {list, dict, set, frozenset, tuple, bytes, bytearray, enumerate, zip, map, filter, reversed}
)


# ---------------------------------------------------------------------------
# Passthrough builtins: functions that operate on their ARGUMENTS rather than
# __self__.  Keyed by id(function) for O(1) lookup.
# Format: {id(fn): (access_kind, obj_arg_index, name_arg_index_or_None)}
# ---------------------------------------------------------------------------

_PASSTHROUGH_BUILTINS: dict[int, tuple[str, int, int | None]] = {}


def _register_passthrough(fn: Any, kind: str, obj_idx: int, name_idx: int | None) -> None:
    _PASSTHROUGH_BUILTINS[id(fn)] = (kind, obj_idx, name_idx)


# Attribute writers: setattr(obj, name, val), delattr(obj, name)
_register_passthrough(_builtins_mod.setattr, "write", 0, 1)
_register_passthrough(_builtins_mod.getattr, "read", 0, 1)
_register_passthrough(_builtins_mod.delattr, "write", 0, 1)
_register_passthrough(_builtins_mod.hasattr, "read", 0, 1)
# operator module item access: operator.setitem(d, k, v), etc.
_register_passthrough(_operator_mod.setitem, "write", 0, 1)
_register_passthrough(_operator_mod.getitem, "read", 0, 1)
_register_passthrough(_operator_mod.delitem, "write", 0, 1)
# len() reads the container (needed to detect check-then-act races)
_register_passthrough(_builtins_mod.len, "read", 0, None)
# Container-iterating builtins: these read their first argument by iterating it.
# Without explicit registration, DPOR doesn't see the read because __self__ is a
# module (builtins / _functools) which is immutable.
_register_passthrough(_builtins_mod.sorted, "read", 0, None)
_register_passthrough(_builtins_mod.min, "read", 0, None)
_register_passthrough(_builtins_mod.max, "read", 0, None)
_register_passthrough(_builtins_mod.sum, "read", 0, None)
_register_passthrough(_builtins_mod.any, "read", 0, None)
_register_passthrough(_builtins_mod.all, "read", 0, None)
_register_passthrough(_builtins_mod.next, "read", 0, None)
_register_passthrough(_functools_mod.reduce, "read", 1, None)


# ---------------------------------------------------------------------------
# Utilities for object key management and access reporting
# ---------------------------------------------------------------------------


def _safe_getattr(obj: Any, attr: str) -> Any:
    """Get an attribute without triggering property descriptors.

    Uses the instance ``__dict__`` to bypass the descriptor protocol,
    preventing property getters from firing.  This is critical because
    ``_process_opcode`` runs inside ``DporScheduler._report_and_wait``'s
    locked section — a property getter that accesses the DB would call
    back into ``report_and_wait``, causing a recursive lock deadlock.

    For class-level attributes (methods, class variables), falls back to
    ``getattr`` only if the attribute is NOT a data descriptor (property,
    etc.).
    """
    # 1. Try instance __dict__ first — bypasses all descriptors
    try:
        inst_dict = object.__getattribute__(obj, "__dict__")
        if attr in inst_dict:
            return inst_dict[attr]
    except (AttributeError, TypeError):
        pass

    # 2. For class-level attributes, check if it's a data descriptor
    #    (property, cached_property, etc.) and skip to avoid side effects.
    for cls in type(obj).__mro__:
        cls_dict = cls.__dict__
        if attr in cls_dict:
            candidate = cls_dict[attr]
            # Data descriptors have __set__ or __delete__ in addition to __get__.
            # These include property, cached_property, and ORM descriptors.
            # Skip them to avoid triggering side-effectful getters.
            if hasattr(candidate, "__set__") or hasattr(candidate, "__delete__"):
                return None
            # Non-data descriptors (regular methods, staticmethod, classmethod)
            # are safe to invoke via getattr.
            return getattr(obj, attr)

    # 3. Fallback for dynamic attributes (__getattr__ etc.)
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _make_object_key(obj_id: int, name: Any) -> int:
    """Create a non-negative u64 object key for the Rust engine."""
    return hash((obj_id, name)) & 0xFFFFFFFFFFFFFFFF


# Module-level reverse map: object_key -> human-readable description.
# Set to a dict when report collection is active, None otherwise.
_object_key_reverse_map: dict[int, str] | None = None


def get_object_key_reverse_map() -> dict[int, str] | None:
    """Return the current reverse map (or None if not collecting)."""
    return _object_key_reverse_map


def set_object_key_reverse_map(rmap: dict[int, str] | None) -> None:
    """Set or clear the reverse map for object key descriptions."""
    global _object_key_reverse_map
    _object_key_reverse_map = rmap


class StableObjectIds:
    """Assign stable per-instance IDs to Python objects, consistent across executions.

    Replaces ``id(obj)`` in object key generation.  Since ``frontrun.explore``
    creates fresh ``state = setup()`` each execution, ``id(obj)`` changes
    between executions for the same logical object, so a raw counter assigned
    in *first-touch order* is unsound: DPOR's whole purpose is to change the
    schedule after a backtrack, which permutes touch order, so the same
    logical object would get a different ID than last execution.  The Rust
    ``Path`` carries object-ID-keyed state (sleep sets, trace caches) across
    executions; permuted IDs make its independence check compare the wrong
    objects and silently prune genuinely distinct interleavings.

    The fix is :meth:`pre_register`: immediately after ``setup()`` (and before
    any worker runs), the per-execution object graph reachable from the fresh
    ``state`` is walked in a **deterministic, schedule-independent order** and
    each object is assigned its counter ID then.  Because ``setup()``
    deterministically rebuilds the same logical graph every execution, this
    produces the **same ID for the same logical object** regardless of the
    runtime schedule.  Objects created *during* worker execution (not
    reachable from ``state`` at setup time) fall back to first-touch order; if
    their touch order diverges that is the inherent residual limitation, but
    the common case (state objects, locks, attributes) is now stable.

    Collisions are conservative-safe: if two distinct objects ever shared an
    ID, that only adds spurious dependencies (over-exploration), never missed
    interleavings.  The dangerous direction — same object, different IDs — is
    what pre-registration eliminates.

    The mapping is maintained per ``frontrun.explore`` call and reset at the start
    of each execution via :meth:`reset_for_execution`.
    """

    # Bound the pre-registration walk so a pathological state graph can't
    # stall exploration.  These are generous; typical state graphs are tiny.
    _MAX_PREREGISTER_OBJECTS = 100_000

    __slots__ = ("_map", "_next_id", "_preregistered")

    def __init__(self) -> None:
        self._map: dict[int, int] = {}
        self._next_id = 0
        # Stable IDs assigned by pre_register (setup-reachable objects).
        # These IDs are deterministic across executions *and* across the
        # exploration/replay boundary, so access anchors may embed them to
        # distinguish instances of the same class (defect #22).
        self._preregistered: set[int] = set()

    def get(self, obj: object) -> int:
        """Return the stable ID for *obj*, assigning one on first access."""
        py_id = id(obj)
        stable_id = self._map.get(py_id)
        if stable_id is None:
            stable_id = self._next_id
            self._map[py_id] = stable_id
            self._next_id += 1
        return stable_id

    def pre_register(self, root: object) -> None:
        """Assign stable IDs to *root*'s object graph in deterministic order.

        Called once per execution, right after ``setup()`` and before any
        worker runs.  Performs a deterministic breadth-first traversal of the
        graph reachable from *root* and calls :meth:`get` on each object so
        the counter IDs are assigned in a **schedule-independent** order that
        is identical across executions (since ``setup()`` rebuilds the same
        logical graph each time).

        Traversal order is fully determined by structure, never by runtime
        scheduling:

        * ``__dict__`` attributes — sorted by attribute name.
        * ``list`` / ``tuple`` — index order.
        * ``dict`` — value insertion order (part of Python's language-level
          mapping contract and therefore stable for deterministic ``setup``).
        * ``set`` / ``frozenset`` — **skipped** (iteration order is not
          deterministic across processes), so their elements register lazily.

        Cycles are handled via a visited set; primitive leaf values (``str``,
        ``int``, ``float``, ``bool``, ``bytes``, ``None``) are not registered
        because they are never used as DPOR access objects.
        """
        seen: set[int] = set()
        queue: deque[object] = deque([root])
        count = 0
        while queue and count < self._MAX_PREREGISTER_OBJECTS:
            obj = queue.popleft()
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            # Skip primitive/immutable leaves: never DPOR access objects and
            # potentially interned/shared (registering them is harmless but
            # pointless and could explode the walk).
            if obj is None or isinstance(obj, (str, bytes, int, float, bool, complex)):
                continue
            # Assign this object's stable ID now, in deterministic order,
            # and record it as pre-registered (run-stable identity).
            self._preregistered.add(self.get(obj))
            count += 1
            self._enqueue_children(obj, queue)

    @staticmethod
    def _enqueue_children(obj: object, queue: deque[object]) -> None:
        """Append *obj*'s children to *queue* in a deterministic order."""
        # Instance attributes, sorted by name (always str — no repr needed).
        obj_dict = getattr(obj, "__dict__", None)
        if isinstance(obj_dict, dict):
            queue.extend(obj_dict[key] for key in sorted(obj_dict))
        if isinstance(obj, (list, tuple)):
            queue.extend(obj)
        elif isinstance(obj, dict):
            # Never sort by key repr: the default object repr embeds a memory
            # address, so two otherwise-identical setup graphs can assign
            # different instance suffixes to replay anchors.  Dict insertion
            # order is language-defined and is the deterministic setup()
            # contract already required by exploration/replay.
            queue.extend(obj.values())
        # set/frozenset deliberately omitted: iteration order is not stable
        # across processes, so registering their elements here would itself
        # be schedule-independent but process-dependent; leave them lazy.

    def is_preregistered(self, obj: object) -> bool:
        """Whether *obj* got its stable ID from :meth:`pre_register`.

        Only pre-registered IDs are deterministic across the exploration and
        replay runs (first-touch IDs depend on the runtime schedule), so only
        they are safe to embed in replay anchor labels.
        """
        stable_id = self._map.get(id(obj))
        return stable_id is not None and stable_id in self._preregistered

    def reset_for_execution(self) -> None:
        """Clear the mapping at the start of each execution.

        Since ``frontrun.explore`` creates fresh state objects each execution,
        old ``id(obj)`` values are stale.  The mapping is rebuilt by
        :meth:`pre_register` (and, for objects created mid-run, lazily during
        replay).
        """
        self._map.clear()
        self._next_id = 0
        self._preregistered.clear()


def _register_object_key(key: int, obj: Any, name: Any) -> None:
    """Register a human-readable description for an object key in the reverse map."""
    rmap = _object_key_reverse_map
    if rmap is not None and key not in rmap:
        type_name = type(obj).__name__
        name_str = str(name) if name is not None else ""
        rmap[key] = f"{type_name}.{name_str}" if name_str else type_name


# ---------------------------------------------------------------------------
# Per-thread access sink (access-anchored replay support)
# ---------------------------------------------------------------------------
#
# When set, every shared-memory access reported by ``_report_access`` is also
# forwarded to the sink as ``(thread_id, obj, name, kind_str, key)``.  The
# DPOR exploration scheduler uses this to record an access trace of the
# failing execution; the replay scheduler uses it to gate accesses to racing
# objects on the recorded order (access-anchored replay), which is robust to
# the positional schedule drift caused by run-varying opcode counts (e.g. a
# real subprocess between a racing write and read).

_access_sink_tls = threading.local()

AccessSink = Callable[[int, Any, Any, str, int], None]


def set_access_sink(sink: AccessSink | None) -> None:
    """Install (or clear) the per-thread access sink."""
    _access_sink_tls.sink = sink


def _get_access_sink() -> AccessSink | None:
    return getattr(_access_sink_tls, "sink", None)


# Builtin container types whose accesses are too generic to serve as replay
# anchors (labels like ``dict.__cmethods__`` match unrelated objects).
_ANCHOR_GENERIC_TYPES = frozenset(
    {"dict", "list", "set", "frozenset", "tuple", "deque", "defaultdict", "OrderedDict", "Counter"}
)


def anchor_label(obj: Any, name: Any, stable_ids: StableObjectIds | None = None) -> str | None:
    """A run-stable label for an access, or ``None`` if too generic to anchor.

    For objects pre-registered in *stable_ids* (the ``setup()`` graph, walked
    in deterministic order by both exploration and replay), the label embeds
    the stable ID — ``TypeName.attr#3`` — so two instances of the same class
    never collide (defect #22).

    Objects created mid-run fall back to plain ``TypeName.attr``: their
    first-touch stable IDs depend on the runtime schedule and therefore
    differ between the exploration run and each replay run, so the type-level
    label is the strongest identity that is still comparable across runs.
    """
    type_name = type(obj).__name__
    if type_name in _ANCHOR_GENERIC_TYPES:
        return None
    name_str = str(name) if name is not None else ""
    if not name_str.isidentifier() or name_str.startswith("__"):
        return None
    if stable_ids is not None and stable_ids.is_preregistered(obj):
        return f"{type_name}.{name_str}#{stable_ids.get(obj)}"
    return f"{type_name}.{name_str}"


# Access-reporting kinds.
#
# - ``read``/``write``: ordinary accesses routed to ``engine.report_access``.
# - ``first_read``: historical name for a read routed to
#   ``engine.report_first_access``.  That API now preserves the earliest and
#   latest positions, both of which are required for sound middle interleavings.
# - ``weak_read``: does not conflict with weak writes or other weak reads.
#   Used for LOAD_ATTR on mutable values so loading a container to subscript
#   it doesn't create a spurious conflict with ``STORE_SUBSCR``'s weak write
#   on disjoint keys, while still conflicting with C-method writes.
# - ``weak_write``: does not conflict with other weak writes.  Used for
#   container-level subscript tracking so that two ``STORE_SUBSCR`` on
#   disjoint keys don't create a spurious conflict, while still conflicting
#   with C-method reads (iteration, ``len()``, ...).
_AccessKind = Literal["read", "first_read", "write", "weak_read", "weak_write"]

# Each kind maps to ``(engine-method-name, kind-string-forwarded-to-engine)``.
_REPORT_DISPATCH: dict[str, tuple[str, str]] = {
    "read": ("report_access", "read"),
    "first_read": ("report_first_access", "read"),
    "write": ("report_access", "write"),
    "weak_read": ("report_access", "weak_read"),
    "weak_write": ("report_access", "weak_write"),
}


def _report_access(
    engine: Any,
    execution: Any,
    thread_id: int,
    obj: Any,
    name: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
    kind: _AccessKind,
) -> None:
    """Report a single shared-memory access to the DPOR engine.

    Consolidates what used to be five near-identical helpers
    (``_report_read``, ``_report_first_read``, ``_report_write``,
    ``_report_weak_read``, ``_report_weak_write``).  ``kind`` selects both
    the engine method (``report_access`` vs ``report_first_access``) and
    the kind string forwarded to it; see ``_REPORT_DISPATCH`` above.
    """
    if obj is None:
        return
    method_name, kind_str = _REPORT_DISPATCH[kind]
    key = _make_object_key(stable_ids.get(obj), name)
    _register_object_key(key, obj, name)
    # Forward to the per-thread access sink (outside the engine lock — the
    # replay sink may block to enforce access-anchored ordering).
    sink = _get_access_sink()
    if sink is not None:
        sink(thread_id, obj, name, kind_str, key)
    with lock:
        getattr(engine, method_name)(execution, thread_id, key, kind_str)


def _subscript_key_name(key: Any) -> Any:
    """Normalize a subscript key for object key computation.

    For string keys, return the string directly so it matches LOAD_ATTR/STORE_ATTR
    argval (e.g. 'value' instead of "'value'").  For non-string keys, use repr()
    as a fallback to distinguish types (e.g. int 0 vs string '0').
    """
    if isinstance(key, str):
        return key
    try:
        return repr(key)
    except Exception:
        return "__subscr__"


def _expand_slice_reads(
    engine: Any,
    execution: Any,
    thread_id: int,
    container: Any,
    key: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
) -> None:
    """Report per-element reads for slice accesses.

    When ``buf[0:4]`` is read, report individual keys "0", "1", "2", "3" so that
    DPOR sees conflicts between slice reads and individual element writes.
    """
    if not isinstance(key, slice):
        return
    try:
        length = len(container)
    except (TypeError, AttributeError):
        return
    try:
        for idx in range(*key.indices(length)):
            _report_access(engine, execution, thread_id, container, repr(idx), lock, stable_ids, "read")
    except (TypeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Scheduling coarsening: only yield at shared-access opcodes
# ---------------------------------------------------------------------------
#
# Most opcodes (LOAD_FAST, STORE_FAST, BINARY_OP +, COPY, SWAP, etc.) only
# manipulate thread-local state (the evaluation stack and f_locals).  They
# never call _report_access, so the DPOR engine doesn't need to
# consider them as scheduling points.  By skipping the scheduler yield for
# these opcodes, we dramatically reduce the DPOR search tree.
#
# Opcodes in _SHARED_ACCESS_OPCODES *may* report shared memory accesses and
# must go through the full scheduler yield path.  Everything else just updates
# the shadow stack without yielding.

_SHARED_ACCESS_OPCODES = frozenset(
    {
        # Attribute access on (potentially shared) objects
        "LOAD_ATTR",
        "STORE_ATTR",
        "DELETE_ATTR",
        "LOAD_METHOD",  # 3.10 only
        "LOAD_SPECIAL",  # 3.14+ context manager __enter__/__exit__
        # Global / name access
        "LOAD_GLOBAL",
        "STORE_GLOBAL",
        "LOAD_NAME",
        # Closure / cell variable access
        "LOAD_DEREF",
        "STORE_DEREF",
        # Subscript / slice access
        "BINARY_SUBSCR",
        "STORE_SUBSCR",
        "DELETE_SUBSCR",
        "BINARY_SLICE",
        "STORE_SLICE",
        # BINARY_OP may be subscript on 3.14 (checked at call site)
        "BINARY_OP",
        # Function / method calls (may invoke C-level methods)
        "CALL",
        "CALL_FUNCTION",
        "CALL_METHOD",
        "CALL_KW",
        "CALL_FUNCTION_KW",
        "CALL_FUNCTION_EX",
        # Iterator operations on (potentially mutable) containers
        "GET_ITER",
        "FOR_ITER",
    }
)


_CALL_OPCODES = frozenset({"CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_KW", "CALL_FUNCTION_KW", "CALL_FUNCTION_EX"})


def _call_might_report_access(shadow: ShadowStack, argc: int) -> bool:
    """Quick pre-check: does this CALL likely involve a C-method that reports access?

    Mirrors the detection strategies in ``_process_opcode``'s CALL handler.
    Returns True if any detectable C-method pattern is found on the shadow stack.
    False negatives would miss accesses; false positives are harmless (extra scheduling).
    """
    scan_depth = min(argc + 3, len(shadow.stack))
    for i in range(scan_depth):
        item = shadow.stack[-(i + 1)]
        if item is None:
            continue
        item_type = type(item)

        # Strategy 1: Passthrough builtins (setattr, getattr, len, etc.)
        if id(item) in _PASSTHROUGH_BUILTINS:
            return True

        # Strategy 2: Bound C methods on mutable __self__
        if item_type is _BUILTIN_METHOD_TYPE or item_type is _METHOD_WRAPPER_TYPE:
            self_obj = getattr(item, "__self__", None)
            if self_obj is not None:
                if not isinstance(self_obj, _IMMUTABLE_TYPES):
                    return True  # C method on mutable object
                # Check for immutable-self methods that read arguments (e.g. str.join)
                method_name = getattr(item, "__name__", None)
                if method_name in _IMMUTABLE_SELF_ARG_READERS:
                    return True

        # Strategy 2b: Container constructors (list, dict, etc.)
        if item_type is type and item in _CONTAINER_CONSTRUCTORS:
            return True

        # Strategy 3: Wrapper descriptors on mutable types
        if item_type is _WRAPPER_DESCRIPTOR_TYPE:
            objclass = getattr(item, "__objclass__", None)
            if objclass is not None and not issubclass(objclass, _IMMUTABLE_TYPES):
                return True

    return False


def _is_iter_source_marker(item: Any) -> bool:
    return isinstance(item, list) and len(item) == 3 and item[0] == "__iter_source__"


def _get_iter_might_report_access(shadow: ShadowStack) -> bool:
    iterable = shadow.peek()
    return iterable is not None and not isinstance(iterable, _IMMUTABLE_TYPES)


def _for_iter_might_report_access(shadow: ShadowStack) -> bool:
    return _is_iter_source_marker(shadow.peek())


def _is_shared_opcode(code: Any, instruction_offset: int) -> bool:
    """Check whether an opcode at the given offset might access shared state.

    Returns True for opcodes in _SHARED_ACCESS_OPCODES, with a special case
    for BINARY_OP: only subscript variants (``[]``, ``NB_SUBSCR``) are shared;
    arithmetic variants (+, -, *, etc.) are not.

    CALL opcodes are NOT checked here — they require shadow stack inspection
    via ``_call_might_report_access`` and are handled separately in the callbacks.
    """
    instrs = _get_instructions(code)
    instr = instrs.get(instruction_offset)
    if instr is None:
        return False
    op = instr.opname
    # CALL opcodes are handled separately (need shadow stack inspection)
    if op in _CALL_OPCODES:
        return False
    if op not in _SHARED_ACCESS_OPCODES:
        return False
    # BINARY_OP is only shared when it's a subscript operation (3.14+)
    if op == "BINARY_OP":
        argrepr = instr.argrepr
        if not argrepr or ("[" not in argrepr and "NB_SUBSCR" not in argrepr.upper()):
            return False
    return True


def _process_opcode(
    frame: Any,
    scheduler: Any,
    thread_id: int,
) -> None:
    """Process a single opcode, updating the shadow stack and reporting accesses.

    Handles opcodes across Python 3.10-3.14, including:
    - 3.13: LOAD_FAST_LOAD_FAST, STORE_FAST_STORE_FAST
    - 3.14: LOAD_FAST_BORROW, LOAD_FAST_BORROW_LOAD_FAST_BORROW,
            STORE_FAST_LOAD_FAST, STORE_FAST_MAYBE_NULL, LOAD_FAST_AND_CLEAR,
            LOAD_SMALL_INT, BINARY_SUBSCR removal
    """
    code = frame.f_code
    instrs = _get_instructions(code)
    instr = instrs.get(frame.f_lasti)
    if instr is None:
        return

    shadow = scheduler.get_shadow_stack(id(frame))
    op = instr.opname
    engine = scheduler.engine
    execution = scheduler.execution
    elock = scheduler._engine_lock
    recorder = scheduler.trace_recorder
    sids = scheduler._stable_ids

    # === LOAD instructions: push values onto the shadow stack ===

    if op in ("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"):
        # LOAD_FAST_BORROW is new in 3.14: same semantics as LOAD_FAST
        # but uses a borrowed reference internally.
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    elif op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
        # New in 3.13: pushes two locals in one instruction.
        # LOAD_FAST_BORROW_LOAD_FAST_BORROW is the 3.14 variant using
        # borrowed references internally (same observable semantics).
        # argval is a tuple of two variable names.
        argval = instr.argval
        if isinstance(argval, tuple) and len(argval) == 2:
            shadow.push(frame.f_locals.get(argval[0]))
            shadow.push(frame.f_locals.get(argval[1]))
        else:
            shadow.push(None)
            shadow.push(None)

    elif op == "LOAD_GLOBAL":
        val = frame.f_globals.get(instr.argval)
        if val is None:
            # Fall back to builtins (setattr, getattr, type, dict, object, etc.)
            _fb = getattr(frame, "f_builtins", None)
            if isinstance(_fb, dict):
                val = _fb.get(instr.argval)
        # On 3.11+, LOAD_GLOBAL with NULL flag (bit 0 of arg) pushes an
        # extra NULL slot.  The value/NULL order flipped in 3.13:
        #   3.11-3.12: [NULL, value]  (NULL below, value on TOS)
        #   3.13+:     [value, NULL]  (value below, NULL on TOS)
        # (see CPython dis argrepr: "NULL + name" pre-3.13 vs "name + NULL" 3.13+)
        if _PY_VERSION >= (3, 11) and instr.arg is not None and instr.arg & 1:
            if _PY_VERSION >= (3, 13):
                shadow.push(val)
                shadow.push(None)
            else:
                shadow.push(None)
                shadow.push(val)
        else:
            shadow.push(val)
        # Report a READ on the module's globals dict for this variable name.
        # Without this, LOAD_GLOBAL/STORE_GLOBAL races are invisible to DPOR.
        # Uses span semantics so repeated loads retain both the earliest and
        # latest backtrack anchors, enabling a write between two reads.
        _report_access(engine, execution, thread_id, frame.f_globals, instr.argval, elock, sids, "first_read")

    elif op == "LOAD_NAME":
        # Used in exec/eval code (module-level scope).  Like LOAD_GLOBAL
        # but checks locals first, then globals, then builtins.
        val = frame.f_locals.get(instr.argval)
        if val is None:
            val = frame.f_globals.get(instr.argval)
        if val is None:
            _fb = getattr(frame, "f_builtins", None)
            if isinstance(_fb, dict):
                val = _fb.get(instr.argval)
        shadow.push(val)

    elif op == "LOAD_DEREF":
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)
        # Report a READ on closure cell/free variables so DPOR sees
        # cross-thread conflicts.  Using code as the identity works because
        # threads sharing a closure function share the same code object.
        varname = instr.argval
        if varname in code.co_freevars or varname in code.co_cellvars:
            _report_access(engine, execution, thread_id, code, varname, elock, sids, "first_read")

    elif op in ("LOAD_CONST", "LOAD_CONST_IMMORTAL", "LOAD_CONST_MORTAL"):
        shadow.push(instr.argval)

    elif op == "LOAD_SMALL_INT":
        # New in 3.14: pushes a small integer (the oparg itself).
        shadow.push(instr.arg)

    # === Stack manipulation ===

    elif op == "COPY":
        n = instr.arg
        if n is not None and len(shadow.stack) >= n:
            shadow.push(shadow.stack[-n])
        else:
            shadow.push(None)

    elif op == "SWAP":
        n = instr.arg
        if n is not None and len(shadow.stack) >= n:
            shadow.stack[-1], shadow.stack[-n] = shadow.stack[-n], shadow.stack[-1]

    # --- Python 3.10 stack manipulation (replaced by COPY/SWAP in 3.11) ---

    elif op == "DUP_TOP":
        shadow.push(shadow.peek())

    elif op == "DUP_TOP_TWO":
        b = shadow.peek(0)
        a = shadow.peek(1)
        shadow.push(a)
        shadow.push(b)

    elif op == "ROT_TWO":
        if len(shadow.stack) >= 2:
            shadow.stack[-1], shadow.stack[-2] = shadow.stack[-2], shadow.stack[-1]

    elif op == "ROT_THREE":
        if len(shadow.stack) >= 3:
            shadow.stack[-1], shadow.stack[-2], shadow.stack[-3] = (
                shadow.stack[-2],
                shadow.stack[-3],
                shadow.stack[-1],
            )

    elif op == "ROT_FOUR":
        if len(shadow.stack) >= 4:
            shadow.stack[-1], shadow.stack[-2], shadow.stack[-3], shadow.stack[-4] = (
                shadow.stack[-2],
                shadow.stack[-3],
                shadow.stack[-4],
                shadow.stack[-1],
            )

    # === Attribute access: the instructions we care about most ===

    elif op == "LOAD_ATTR":
        obj = shadow.pop()
        attr = instr.argval
        # Skip I/O wrapper and client types — their conflicts are tracked by
        # higher-level reporters (file path, Redis key, SQL table), not by
        # Python object identity.
        if obj is None or not isinstance(obj, (_IO_WRAPPER_TYPES + _IO_CLIENT_TYPES)):
            _report_access(engine, execution, thread_id, obj, attr, elock, sids, "read")
        # Also report on obj.__dict__ so LOAD_ATTR conflicts with
        # STORE_SUBSCR on the same __dict__ (cross-path detection).
        # Off by default: doubles wakeup tree insertions for rare benefit.
        if getattr(scheduler, "_track_dunder_dict_accesses", False) and obj is not None:
            try:
                _obj_dict = object.__getattribute__(obj, "__dict__")
                _report_access(engine, execution, thread_id, _obj_dict, attr, elock, sids, "read")
            except AttributeError:
                pass
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=attr, obj=obj)
        if obj is not None:
            # Only the user-touching attribute resolution is guarded: resolving
            # ``obj.attr`` can fire a user descriptor/``__getattr__`` that raises,
            # and that must not abort tracing.  The weak-read report below is
            # frontrun-internal and must run UNGUARDED — swallowing a failure
            # there would silently drop the access (under-merging → false pass).
            try:
                val = _safe_getattr(obj, attr)
            except Exception:
                val = None
            shadow.push(val)
            # When the loaded value is a mutable object (but NOT a bound
            # method), report a WEAK READ on the object itself.  This
            # detects cases where a container is read indirectly —
            # e.g. passed to len() or iterated — creating a conflict
            # with C-level method WRITEs (append, add, etc.) reported
            # by the CALL handler below.
            #
            # We use weak_read (not read) so that loading a dict just
            # to subscript it doesn't conflict with STORE_SUBSCR's
            # weak_write on disjoint keys.
            #
            # We skip bound methods (loading .append is not a container
            # read) and immutable types (no mutation possible).
            if val is not None and type(val) is not _BUILTIN_METHOD_TYPE and isinstance(val, (list, dict, set)):
                _report_access(engine, execution, thread_id, val, "__cmethods__", elock, sids, "weak_read")
        else:
            shadow.push(None)
        # On 3.12+, LOAD_ATTR with method flag (bit 0 of arg) pushes an extra
        # self/NULL slot, matching LOAD_METHOD's stack layout.  On 3.11,
        # LOAD_ATTR with the method flag has stack_effect=0 (no extra push), so
        # we skip it there.  The value/NULL order flipped in 3.13 (same flip as
        # LOAD_GLOBAL):
        #   3.12:  [NULL, value]  (NULL/self below, value on TOS)
        #   3.13+: [value, NULL]  (value below, NULL/self on TOS)
        # The value was already pushed above, so for 3.13+ we push NULL on top,
        # and for 3.12 we insert NULL *below* the value.
        if _PY_VERSION >= (3, 12) and instr.arg is not None and instr.arg & 1:
            if _PY_VERSION >= (3, 13):
                shadow.push(None)
            elif shadow.stack:
                shadow.stack.insert(-1, None)
            else:
                shadow.push(None)

    elif op == "STORE_ATTR":
        obj = shadow.pop()  # TOS = object
        _val = shadow.pop()  # TOS1 = value
        # Skip I/O wrapper and client types — their conflicts are tracked by
        # higher-level reporters, not by Python object identity.
        if obj is None or not isinstance(obj, (_IO_WRAPPER_TYPES + _IO_CLIENT_TYPES)):
            _report_access(engine, execution, thread_id, obj, instr.argval, elock, sids, "write")
        # Also report on obj.__dict__ so STORE_ATTR conflicts with
        # STORE_SUBSCR on the same __dict__ (cross-path detection).
        # Off by default: doubles wakeup tree insertions for rare benefit.
        if getattr(scheduler, "_track_dunder_dict_accesses", False) and obj is not None:
            try:
                _obj_dict = object.__getattribute__(obj, "__dict__")
                _report_access(engine, execution, thread_id, _obj_dict, instr.argval, elock, sids, "write")
            except AttributeError:
                pass
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=instr.argval, obj=obj)

    elif op == "LOAD_METHOD":
        # Python 3.10 only (replaced by LOAD_ATTR with method flag in 3.11+).
        # Pops owner, pushes (method, self/NULL) — net stack effect +1.
        obj = shadow.pop()
        attr = instr.argval
        _report_access(engine, execution, thread_id, obj, attr, elock, sids, "read")
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=attr, obj=obj)
        if obj is not None:
            # Guard only the user attribute resolution (``_safe_getattr`` can
            # invoke a non-data descriptor whose getter raises); the shadow-stack
            # push is frontrun-internal and must not be masked.
            try:
                _method_val = _safe_getattr(obj, attr)
            except Exception:
                _method_val = None
            shadow.push(_method_val)
        else:
            shadow.push(None)
        # Extra push for the self/NULL slot (LOAD_METHOD pushes 2 values).
        shadow.push(None)

    elif op == "DELETE_ATTR":
        obj = shadow.pop()
        _report_access(engine, execution, thread_id, obj, instr.argval, elock, sids, "write")
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=instr.argval, obj=obj)

    elif op == "LOAD_SPECIAL":
        # New in 3.14: replaces LOAD_ATTR for ``__enter__`` / ``__exit__``
        # in ``with`` statements.  Pops owner, pushes (attr, self_or_null).
        # Stack effect = +1 (−1 pop + 2 push).
        _special_names = {0: "__enter__", 1: "__exit__", 2: "__aenter__", 3: "__aexit__"}
        _arg = instr.arg if instr.arg is not None else -1
        attr = _special_names.get(_arg, f"__special_{_arg}__")
        obj = shadow.pop()
        # Skip I/O wrapper and client types (same as LOAD_ATTR).
        if obj is None or not isinstance(obj, (_IO_WRAPPER_TYPES + _IO_CLIENT_TYPES)):
            _report_access(engine, execution, thread_id, obj, attr, elock, sids, "read")
        if obj is not None:
            # Guard only the user attribute resolution (``_safe_getattr`` can
            # invoke a non-data descriptor whose getter raises); the shadow-stack
            # push is frontrun-internal and must not be masked.
            try:
                val = _safe_getattr(obj, attr)
            except Exception:
                val = None
            shadow.push(val)
        else:
            shadow.push(None)
        shadow.push(None)  # self_or_null slot

    # === Subscript access (dict/list operations) ===

    elif op == "BINARY_SUBSCR":
        # Present on 3.10-3.13. Removed in 3.14 (replaced by BINARY_OP
        # with subscript oparg).
        key = shadow.pop()
        container = shadow.pop()
        _kname = _subscript_key_name(key)
        _report_access(engine, execution, thread_id, container, _kname, elock, sids, "read")
        # Container-level read for conflict with C-methods and different subscript keys.
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_access(engine, execution, thread_id, container, "__cmethods__", elock, sids, "read")
            # For slice accesses, also report reads on individual element keys
            # so DPOR sees per-element conflicts with STORE_SUBSCR writes.
            _expand_slice_reads(engine, execution, thread_id, container, key, elock, sids)
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=_kname, obj=container)
        shadow.push(None)

    elif op == "STORE_SUBSCR":
        key = shadow.pop()
        container = shadow.pop()
        _val = shadow.pop()
        _kname = _subscript_key_name(key)
        _report_access(engine, execution, thread_id, container, _kname, elock, sids, "write")
        # Report a container-level weak-write so subscript writes conflict
        # with C-method reads (e.g. len(), iteration) but two subscript
        # writes on disjoint keys do NOT conflict with each other.
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_access(engine, execution, thread_id, container, "__cmethods__", elock, sids, "weak_write")
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=_kname, obj=container)

    elif op == "BINARY_SLICE":
        # New in 3.12 (replaces BINARY_SUBSCR for slice operations).
        # Stack: [container, start, stop] → [result]
        _stop = shadow.pop()
        _start = shadow.pop()
        container = shadow.pop()
        _report_access(engine, execution, thread_id, container, "__slice__", elock, sids, "read")
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_access(engine, execution, thread_id, container, "__cmethods__", elock, sids, "read")
            _expand_slice_reads(engine, execution, thread_id, container, slice(_start, _stop), elock, sids)
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name="__slice__", obj=container)
        shadow.push(None)

    elif op == "STORE_SLICE":
        # New in 3.12 (replaces STORE_SUBSCR for slice operations).
        # Stack: [value, container, start, stop] → []
        _stop = shadow.pop()
        _start = shadow.pop()
        container = shadow.pop()
        _val = shadow.pop()
        _report_access(engine, execution, thread_id, container, "__slice__", elock, sids, "write")
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_access(engine, execution, thread_id, container, "__cmethods__", elock, sids, "weak_write")
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name="__slice__", obj=container)

    elif op == "DELETE_SUBSCR":
        key = shadow.pop()
        container = shadow.pop()
        _kname = _subscript_key_name(key)
        _report_access(engine, execution, thread_id, container, _kname, elock, sids, "write")
        # Container-level write for delete too (regular last-access semantics).
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_access(engine, execution, thread_id, container, "__cmethods__", elock, sids, "write")
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=_kname, obj=container)

    # === Arithmetic and binary operations ===

    elif op == "BINARY_OP":
        # On 3.14, BINARY_OP also handles subscript operations (replacing
        # the removed BINARY_SUBSCR). Check the argrepr for "[]" / subscript.
        argrepr = instr.argrepr
        if argrepr and ("[" in argrepr or "NB_SUBSCR" in argrepr.upper()):
            key = shadow.pop()
            container = shadow.pop()
            _kname = _subscript_key_name(key)
            _report_access(engine, execution, thread_id, container, _kname, elock, sids, "read")
            # Container-level read for subscript access (same as BINARY_SUBSCR).
            if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
                _report_access(engine, execution, thread_id, container, "__cmethods__", elock, sids, "read")
                # For slice accesses, also report reads on individual element keys
                _expand_slice_reads(engine, execution, thread_id, container, key, elock, sids)
            if recorder is not None and container is not None:
                recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=_kname, obj=container)
            shadow.push(None)
        else:
            shadow.pop()
            shadow.pop()
            shadow.push(None)

    elif op.startswith(("INPLACE_", "BINARY_")):
        # Python 3.10 INPLACE_ADD, INPLACE_SUBTRACT, etc. and
        # BINARY_ADD, BINARY_MULTIPLY, etc. All pop 2, push 1.
        # (BINARY_OP and BINARY_SUBSCR are already handled above.)
        shadow.pop()
        shadow.pop()
        shadow.push(None)

    # === Store instructions ===

    elif op == "STORE_GLOBAL":
        shadow.pop()
        # Report a WRITE on the module's globals dict for this variable name.
        _report_access(engine, execution, thread_id, frame.f_globals, instr.argval, elock, sids, "write")

    elif op == "STORE_DEREF":
        shadow.pop()
        # Report a WRITE on closure cell/free variables.
        varname = instr.argval
        if varname in code.co_freevars or varname in code.co_cellvars:
            _report_access(engine, execution, thread_id, code, varname, elock, sids, "write")

    elif op == "STORE_FAST":
        shadow.pop()

    elif op == "STORE_FAST_STORE_FAST":
        # New in 3.13: pops two values.
        shadow.pop()
        shadow.pop()

    elif op == "STORE_FAST_LOAD_FAST":
        # New in 3.14: stores TOS into one local, then loads another local.
        # Net stack effect is 0 (pop one, push one) but the fallback handler
        # sees effect=0 and does nothing, losing the pushed value.
        argval = instr.argval
        shadow.pop()
        if isinstance(argval, tuple) and len(argval) == 2:
            shadow.push(frame.f_locals.get(argval[1]))
        else:
            shadow.push(None)

    elif op == "STORE_FAST_MAYBE_NULL":
        # New in 3.14: like STORE_FAST but tolerates NULL on TOS.
        shadow.pop()

    elif op == "LOAD_FAST_AND_CLEAR":
        # Used in comprehensions: loads a local then clears it.
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    # === Return/pop ===

    elif op in ("RETURN_VALUE", "RETURN_CONST"):
        shadow.pop()

    elif op == "POP_TOP":
        shadow.pop()

    # === Build instructions (slice, list, etc.) ===

    elif op == "BUILD_SLICE":
        # BUILD_SLICE pops 2 or 3 items (start, stop, [step]) and pushes a slice object.
        # The fallback stack_effect handler adjusts the size correctly, but we need
        # the ACTUAL slice object on the shadow stack so that BINARY_SUBSCR can
        # detect slice accesses and expand them into per-element reads.
        argc = instr.arg or 2
        items: list[Any] = [shadow.pop() for _ in range(argc)]
        items.reverse()
        try:
            if argc == 2:
                shadow.push(slice(items[0], items[1]))
            else:
                shadow.push(slice(items[0], items[1], items[2]))
        except (TypeError, ValueError):
            shadow.push(None)

    # === Iterator operations ===
    # GET_ITER creates an iterator from a container. We record the mapping
    # so that FOR_ITER can report reads on the original container.

    elif op == "GET_ITER":
        # stack: [iterable] → [iterator]
        # GET_ITER pops the iterable and pushes an iterator (stack effect = 0).
        # We record the iterable→iterator mapping via a mutable marker on the
        # shadow stack so that FOR_ITER can report per-element reads.
        iterable = shadow.peek()
        if iterable is not None and not isinstance(iterable, _IMMUTABLE_TYPES):
            shadow.pop()
            # Mutable list marker: [tag, container, iteration_counter]
            # The counter tracks which element index FOR_ITER is reading,
            # enabling per-element conflict detection with STORE_SUBSCR writes.
            shadow.push(["__iter_source__", iterable, 0])
            # Report a container-level read now.  On Python 3.10, list
            # comprehensions are compiled as nested functions, so FOR_ITER
            # runs in a child frame with a fresh shadow stack that won't
            # see the __iter_source__ marker.  The read here ensures the
            # conflict with STORE_SUBSCR weak-writes is recorded in the
            # current frame before the iterator crosses frame boundaries.
            _report_access(engine, execution, thread_id, iterable, "__cmethods__", elock, sids, "first_read")
        else:
            shadow.pop()
            shadow.push(None)

    elif op == "FOR_ITER":
        # stack: [iterator] → [iterator, next_value] or [−iterator] (exhausted)
        # FOR_ITER calls __next__ on the iterator. If the iterator was created
        # from a mutable container (tracked via GET_ITER), report reads on it.
        # stack effect = +1 (pushes the yielded value; TOS is the iterator).
        # We peek at the iterator marker to find the underlying container.
        top = shadow.peek()
        if isinstance(top, list) and len(top) == 3 and top[0] == "__iter_source__":
            _iter_container = top[1]
            _iter_counter = top[2]
            if _iter_container is not None and not isinstance(_iter_container, _IMMUTABLE_TYPES):
                # Per-element read using the iteration counter as the key.
                # For lists, counter 0, 1, 2... matches STORE_SUBSCR keys "0", "1", "2"...
                # This creates per-element conflicts enabling fine-grained interleaving.
                _report_access(
                    engine, execution, thread_id, _iter_container, repr(_iter_counter), elock, sids, "first_read"
                )
                # Coarse-grained read for conflict with C-method writes (append,
                # insert, etc.) and other container-level operations.  Uses last-access
                # (regular) semantics: each iteration overwrites the previous read
                # position.  This means wakeup tree entries target the LAST iteration,
                # which allows the other thread to interleave after some elements have
                # already been read — catching mid-iteration mutation races (e.g.
                # enumerate + insert).
                _report_access(engine, execution, thread_id, _iter_container, "__cmethods__", elock, sids, "read")
            # Increment counter for next iteration (mutable list, in-place update).
            top[2] = _iter_counter + 1
        shadow.push(None)  # push the yielded value

    elif op == "END_FOR":
        # End of for loop — pops the exhausted iterator value.
        shadow.pop()

    elif op == "POP_ITER":
        # Python 3.14: pops the iterator itself at end of for loop.
        shadow.pop()

    # === Function/method calls ===

    elif op == "PRECALL":
        # Python 3.11 only.  PRECALL is a no-op for the evaluation stack
        # (it's a cache/optimization hint for the interpreter).  However,
        # dis.stack_effect reports a negative effect equal to -argc, which
        # would corrupt the shadow stack if handled by the fallback.
        pass

    elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_KW", "CALL_FUNCTION_KW", "CALL_FUNCTION_EX"):
        # Detect C-level method calls and classify as read or write.
        #
        # Three detection strategies, tried in order:
        # 1. Passthrough builtins (setattr, getattr, operator.setitem, etc.)
        #    — operate on their arguments rather than __self__
        # 2. Bound C methods (list.append, dict.update, etc.)
        #    — __self__ is the mutable target object
        # 3. Wrapper descriptors (object.__setattr__, dict.__setitem__, etc.)
        #    — unbound C type methods, first argument is the target
        argc = instr.arg or 0
        scan_depth = min(argc + 3, len(shadow.stack))
        _call_handled = False
        # When a container constructor (enumerate, zip, etc.) wraps a mutable
        # iterable, we save the source container so that GET_ITER → FOR_ITER
        # can report per-element reads on the underlying container.
        _constructor_source: Any = None

        for i in range(scan_depth):
            item = shadow.stack[-(i + 1)]
            if item is None:
                continue
            item_type = type(item)

            # --- Strategy 1: Passthrough builtins ---
            # These are builtins whose __self__ is a module (builtins, operator)
            # but that access their ARGUMENTS.  Identified by id(function).
            _pt = _PASSTHROUGH_BUILTINS.get(id(item))
            if _pt is not None:
                _pt_kind, _pt_obj_idx, _pt_name_idx = _pt
                # Arguments are always in the top `argc` positions on the stack
                # regardless of Python version (3.10: [func, args], 3.11-3.13:
                # [NULL, func, args], 3.14: [func, NULL, args]).
                _slen = len(shadow.stack)
                _obj_depth = argc - _pt_obj_idx
                if argc >= _pt_obj_idx + 1 and 0 < _obj_depth <= _slen:
                    _pt_target = shadow.stack[-_obj_depth]
                    _pt_attr: Any = "__cmethods__"
                    if _pt_name_idx is not None and argc >= _pt_name_idx + 1:
                        _name_depth = argc - _pt_name_idx
                        if 0 < _name_depth <= _slen:
                            _raw = shadow.stack[-_name_depth]
                            if isinstance(_raw, str):
                                _pt_attr = _raw
                    if _pt_target is not None and not isinstance(_pt_target, _IMMUTABLE_TYPES):
                        if _pt_kind == "read":
                            _report_access(engine, execution, thread_id, _pt_target, _pt_attr, elock, sids, "read")
                        else:
                            _report_access(engine, execution, thread_id, _pt_target, _pt_attr, elock, sids, "write")
                _call_handled = True
                break

            # --- Strategy 2: Bound C methods (existing behavior) ---
            if item_type is _BUILTIN_METHOD_TYPE or item_type is _METHOD_WRAPPER_TYPE:
                self_obj = getattr(item, "__self__", None)
                if self_obj is not None and not isinstance(self_obj, _IMMUTABLE_TYPES):
                    if isinstance(self_obj, (_IO_WRAPPER_TYPES + _IO_CLIENT_TYPES)):
                        # I/O wrapper and client methods are tracked via
                        # higher-level reporters; skip __cmethods__ to avoid
                        # false races from id() reuse on short-lived objects.
                        # Exception: DB cursor types — two threads sharing a
                        # single cursor constitutes a real conflict that the
                        # SQL-level reporter (table/row granularity) misses.
                        if not isinstance(self_obj, _DB_CURSOR_TYPES):
                            _call_handled = True
                            break
                    method_name = getattr(item, "__name__", None)
                    if method_name in _C_METHOD_READ_ONLY:
                        _report_access(engine, execution, thread_id, self_obj, "__cmethods__", elock, sids, "read")
                    else:
                        _report_access(engine, execution, thread_id, self_obj, "__cmethods__", elock, sids, "write")
                    _call_handled = True
                    break
                # __self__ is immutable (e.g. str, module) — check if the method
                # iterates its first argument (e.g. str.join reads the iterable).
                if self_obj is not None:
                    method_name = getattr(item, "__name__", None)
                    if method_name in _IMMUTABLE_SELF_ARG_READERS and argc >= 1 and argc <= len(shadow.stack):
                        _arg_target = shadow.stack[-argc]
                        if _arg_target is not None and not isinstance(_arg_target, _IMMUTABLE_TYPES):
                            _report_access(
                                engine, execution, thread_id, _arg_target, "__cmethods__", elock, sids, "read"
                            )
                        _call_handled = True
                        break
                    # Otherwise fall through to continue scan

            # --- Strategy 2b: Type constructors that iterate arguments ---
            # list(iterable), dict(iterable), bytes(iterable), enumerate(iterable),
            # zip(iter1, iter2), map(func, iterable), filter(func, iterable), etc.
            if item_type is type and item in _CONTAINER_CONSTRUCTORS:
                # Report a READ on each mutable argument (they get iterated).
                # Also save the first mutable arg as the "source container" so that
                # if this constructor result is iterated via FOR_ITER, the reads
                # are attributed to the underlying container (not the wrapper).
                for _ci in range(argc):
                    _c_depth = argc - _ci
                    if _c_depth < 1 or _c_depth > len(shadow.stack):
                        continue
                    _c_arg = shadow.stack[-_c_depth]
                    if _c_arg is not None and not isinstance(_c_arg, _IMMUTABLE_TYPES):
                        _report_access(engine, execution, thread_id, _c_arg, "__cmethods__", elock, sids, "read")
                        if _constructor_source is None:
                            _constructor_source = _c_arg
                _call_handled = True
                break

            # --- Strategy 3: Wrapper descriptors (unbound C type methods) ---
            if item_type is _WRAPPER_DESCRIPTOR_TYPE:
                objclass = getattr(item, "__objclass__", None)
                if objclass is not None and not issubclass(objclass, _IMMUTABLE_TYPES):
                    # First argument (self) is always at the bottom of the argc args
                    if argc >= 1 and argc <= len(shadow.stack):
                        _wd_target = shadow.stack[-argc]
                        if _wd_target is not None and not isinstance(_wd_target, _IMMUTABLE_TYPES):
                            method_name = getattr(item, "__name__", None)
                            if method_name in _C_METHOD_READ_ONLY:
                                _report_access(
                                    engine, execution, thread_id, _wd_target, "__cmethods__", elock, sids, "read"
                                )
                            else:
                                _report_access(
                                    engine, execution, thread_id, _wd_target, "__cmethods__", elock, sids, "write"
                                )
                _call_handled = True
                break

        # Standard stack effect handling.
        try:
            effect = dis.stack_effect(instr.opcode, instr.arg or 0)
            # On Python 3.11, PRECALL reported a stack effect of -argc
            # but we handle it as a no-op (it doesn't touch the real
            # stack).  Compensate by adding the missing pops to CALL.
            if _PY_VERSION[:2] == (3, 11) and op == "CALL" and argc > 0:
                effect -= argc
            for _ in range(max(0, -effect)):
                shadow.pop()
            for _ in range(max(0, effect)):
                shadow.push(None)
        except (ValueError, TypeError):
            shadow.clear()

        # CALL replaces the callable/args with the return value. Plain
        # stack_effect accounting leaves the bottom-most operand in place,
        # which can be the callable itself on 3.10+ and pollute subsequent
        # attribute/subscript tracking.
        if shadow.stack:
            shadow.stack[-1] = None
        else:
            shadow.push(None)

        # Fixup: when a container constructor (enumerate, zip, map, etc.) wraps
        # a mutable iterable, replace the None result on TOS with the source
        # container.  This way GET_ITER picks it up and FOR_ITER can report
        # per-element reads on the underlying container during iteration.
        if _constructor_source is not None and shadow.stack:
            shadow.stack[-1] = _constructor_source

    else:
        # Fallback: use dis.stack_effect for unknown opcodes.
        # This handles PUSH_NULL, RESUME, and any version-specific
        # opcodes we don't explicitly handle.
        try:
            effect = dis.stack_effect(instr.opcode, instr.arg or 0)
            for _ in range(max(0, -effect)):
                shadow.pop()
            for _ in range(max(0, effect)):
                shadow.push(None)
        except (ValueError, TypeError):
            shadow.clear()


# ---------------------------------------------------------------------------
# Shared sys.monitoring helpers (used by dpor.py and async_dpor.py)
# ---------------------------------------------------------------------------


ToolKind = Literal["profiler", "optimizer"]


def _resolve_tool_id(tool_kind: ToolKind) -> int:
    """Return the sys.monitoring tool ID matching *tool_kind*.

    ``"profiler"`` (default for DPOR) or ``"optimizer"`` (used by
    BytecodeShuffler so that random fuzzing and DPOR can coexist when
    nested, although in practice they aren't).
    """
    mon = sys.monitoring
    if tool_kind == "profiler":
        return mon.PROFILER_ID  # type: ignore[attr-defined,no-any-return]
    if tool_kind == "optimizer":
        return mon.OPTIMIZER_ID  # type: ignore[attr-defined,no-any-return]
    raise ValueError(f"Unknown tool_kind: {tool_kind!r}")


# ---------------------------------------------------------------------------
# sys.monitoring tool-ID ownership registry (finding 3)
# ---------------------------------------------------------------------------
#
# sys.monitoring tool IDs are a global, process-wide resource.  When
# ``use_tool_id`` raises ValueError the slot is already held.  We must only
# force-steal a slot we KNOW belongs to a previous frontrun run that has since
# finished or died — never an external profiler/optimizer or a concurrent
# frontrun run in another (still-alive) thread.  This registry maps each
# claimed tool_id to the ident of the frontrun thread that claimed it, so the
# recovery path can tell "stale frontrun run" apart from "live holder".  The
# tool name is part of the claim: a thread id alone is insufficient because an
# external tool can reclaim the interpreter slot while a leaked registry entry
# from that same long-lived thread remains.

_TOOL_OWNERS: dict[int, tuple[int, str]] = {}
_TOOL_OWNERS_LOCK = threading.Lock()


def _owner_thread_alive(owner_ident: int) -> bool:
    """Whether the thread that claimed a tool_id is still running."""
    return any(t.ident == owner_ident and t.is_alive() for t in threading.enumerate())


def setup_opcode_monitoring(
    *,
    tool_name: str,
    handle_py_start: Any,
    handle_py_return: Any,
    handle_instruction: Any,
    handle_py_unwind: Any = None,
    tool_kind: ToolKind = "profiler",
    monitor_returns: bool = True,
) -> int:
    """Set up sys.monitoring for opcode tracing. Returns the tool ID.

    Handles the full tool ID lifecycle including defensive cleanup of
    stale tool IDs from interrupted runs (e.g. pytest-timeout kills a
    test before teardown).  Such recovery is ONLY performed when the
    ownership registry shows the slot belongs to a frontrun run that has
    since finished or whose owning thread has died — a live external tool
    (or a concurrent frontrun run) is never destroyed; instead a clear
    error is raised (finding 3).

    *tool_kind* selects the sys.monitoring tool ID slot
    (``"profiler"`` for DPOR, ``"optimizer"`` for the BytecodeShuffler).
    When *monitor_returns* is False, PY_RETURN/PY_UNWIND are not registered
    (used by callers that don't need shadow-stack cleanup).
    """
    mon = sys.monitoring
    tool_id: int = _resolve_tool_id(tool_kind)

    with _TOOL_OWNERS_LOCK:
        try:
            mon.use_tool_id(tool_id, tool_name)  # type: ignore[attr-defined]
        except ValueError:
            prior = _TOOL_OWNERS.get(tool_id)
            if prior is None:
                # Not a frontrun-owned slot: held by an external profiler or
                # another tool.  Do NOT steal it.
                held_by = mon.get_tool(tool_id)  # type: ignore[attr-defined]
                raise RuntimeError(
                    f"sys.monitoring tool id {tool_id} ({tool_kind}) is already in use "
                    f"by {held_by!r} and is not owned by frontrun; refusing to steal it. "
                    "Stop the other monitoring tool before running frontrun."
                ) from None
            prior_ident, prior_name = prior
            held_by = mon.get_tool(tool_id)  # type: ignore[attr-defined]
            if held_by != prior_name:
                # The interpreter is authoritative: another tool reclaimed
                # the slot while our registry entry leaked.
                del _TOOL_OWNERS[tool_id]
                raise RuntimeError(
                    f"sys.monitoring tool id {tool_id} ({tool_kind}) is already in use "
                    f"by {held_by!r} and is not owned by frontrun; refusing to steal it. "
                    "Stop the other monitoring tool before running frontrun."
                ) from None
            if prior_ident != threading.get_ident() and _owner_thread_alive(prior_ident):
                # A concurrent frontrun run in another live thread owns it.
                raise RuntimeError(
                    f"sys.monitoring tool id {tool_id} ({tool_kind}) is in use by a "
                    f"concurrent frontrun run (thread {prior_ident}); "
                    "frontrun runs that share a tool-id slot cannot overlap."
                ) from None
            # Genuinely stale — either the owning thread has died, or the owner
            # is THIS very thread (a previous run on this same sequential thread
            # leaked the slot without tearing down, e.g. a launch failure
            # bypassed the teardown callback).  A single thread cannot overlap
            # with itself, so a same-ident owner is always a stale slot, not a
            # live concurrent holder.  Force-free and reclaim.
            _force_free_tool_id(tool_id)
            mon.use_tool_id(tool_id, tool_name)  # type: ignore[attr-defined]

        _TOOL_OWNERS[tool_id] = (threading.get_ident(), tool_name)

    events = mon.events.PY_START | mon.events.INSTRUCTION  # type: ignore[attr-defined]
    if monitor_returns:
        events |= mon.events.PY_RETURN  # type: ignore[attr-defined]
        events |= mon.events.PY_UNWIND  # type: ignore[attr-defined]
    mon.set_events(tool_id, events)  # type: ignore[attr-defined]
    mon.register_callback(tool_id, mon.events.PY_START, handle_py_start)  # type: ignore[attr-defined]
    if monitor_returns:
        mon.register_callback(tool_id, mon.events.PY_RETURN, handle_py_return)  # type: ignore[attr-defined]
        # PY_UNWIND mirrors PY_RETURN cleanup for exception-exiting frames
        # (finding 4).  Fall back to handle_py_return when an explicit unwind
        # handler is not provided (it performs the same shadow-stack removal).
        mon.register_callback(  # type: ignore[attr-defined]
            tool_id, mon.events.PY_UNWIND, handle_py_unwind if handle_py_unwind is not None else handle_py_return
        )
    mon.register_callback(tool_id, mon.events.INSTRUCTION, handle_instruction)  # type: ignore[attr-defined]
    return tool_id


def _force_free_tool_id(tool_id: int) -> None:
    """Unregister all callbacks and free *tool_id* (caller verified staleness)."""
    mon = sys.monitoring
    mon.set_events(tool_id, 0)  # type: ignore[attr-defined]
    mon.register_callback(tool_id, mon.events.PY_START, None)  # type: ignore[attr-defined]
    mon.register_callback(tool_id, mon.events.PY_RETURN, None)  # type: ignore[attr-defined]
    mon.register_callback(tool_id, mon.events.PY_UNWIND, None)  # type: ignore[attr-defined]
    mon.register_callback(tool_id, mon.events.INSTRUCTION, None)  # type: ignore[attr-defined]
    mon.free_tool_id(tool_id)  # type: ignore[attr-defined]


def teardown_opcode_monitoring(tool_id: int | None) -> None:
    """Tear down sys.monitoring for opcode tracing."""
    if tool_id is None:
        return
    with _TOOL_OWNERS_LOCK:
        # Only free a slot we still own; another run may have reclaimed it after
        # we died, in which case we must not stomp on it.
        owner = _TOOL_OWNERS.get(tool_id)
        if owner is not None and owner[0] == threading.get_ident():
            del _TOOL_OWNERS[tool_id]
            if sys.monitoring.get_tool(tool_id) == owner[1]:  # type: ignore[attr-defined]
                _force_free_tool_id(tool_id)


# ---------------------------------------------------------------------------
# Scheduling coarsening helper
# ---------------------------------------------------------------------------


def process_opcode_with_coarsening(
    code: Any,
    offset: int,
    frame: Any,
    scheduler: Any,
    thread_id: int,
    detect_io: bool,
) -> bool:
    """Process an opcode with scheduling coarsening, returning whether we yielded.

    Shared opcodes (LOAD_ATTR, STORE_ATTR, etc.) trigger
    ``scheduler.report_and_wait`` which processes the shadow stack AND
    yields to the scheduler.  Non-shared opcodes only update the shadow
    stack via ``_process_opcode`` without yielding.

    CALL opcodes get special treatment: they are non-shared by default but
    may be upgraded to a scheduling point if the shadow stack indicates the
    callable is a C-method that accesses shared state (e.g. ``list.append``).
    When *detect_io* is True, ALL CALL opcodes yield because I/O events
    arrive from C-level LD_PRELOAD interception.

    Returns True if ``report_and_wait`` was called (the thread may have
    waited), False if only shadow-stack processing occurred.
    """
    instrs = _get_instructions(code)
    instr = instrs.get(offset)
    if instr is not None and instr.opname in ("GET_ITER", "FOR_ITER"):
        shadow = scheduler.get_shadow_stack(id(frame))
        might_report = _get_iter_might_report_access if instr.opname == "GET_ITER" else _for_iter_might_report_access
        if might_report(shadow):
            scheduler.report_and_wait(frame, thread_id)
            return True
        _process_opcode(frame, scheduler, thread_id)
        return False
    if not _is_shared_opcode(code, offset):
        if instr is not None and instr.opname in _CALL_OPCODES:
            if detect_io:
                scheduler.report_and_wait(frame, thread_id)
                return True
            shadow = scheduler.get_shadow_stack(id(frame))
            argc = instr.arg or 0
            if _call_might_report_access(shadow, argc):
                scheduler.report_and_wait(frame, thread_id)
                return True
        _process_opcode(frame, scheduler, thread_id)
        return False
    scheduler.report_and_wait(frame, thread_id)
    return True


# ---------------------------------------------------------------------------
# Unified settrace / sys.monitoring callback factories
# ---------------------------------------------------------------------------
#
# Both dpor.py (sync threads) and async_dpor.py (async tasks) need
# essentially the same instrumentation wiring for sys.settrace (3.10-3.11)
# and sys.monitoring (3.12+).  The differences are parameterised:
#
#   get_thread_id()       — how to identify the current thread/task
#   on_opcode(c,o,f,tid)  — what to do per opcode (coarsening or plain)
#   remove_shadow_stack() — cleanup on frame return
#   detect_io             — affects dynamic-code filtering
#   is_active()           — early-out when scheduler is done (sync only)


def make_settrace_callback(
    *,
    get_thread_id: Any,
    on_opcode: Any,
    remove_shadow_stack: Any,
    detect_io: bool = False,
    is_active: Any = None,
) -> Any:
    """Create a ``sys.settrace`` callback for opcode tracing.

    Parameters
    ----------
    get_thread_id:
        ``() -> int | None``.  Returns the current thread/task ID if we
        are inside an active scheduler context, ``None`` otherwise.
    on_opcode:
        ``(code, offset, frame, thread_id) -> bool``.  Called for every
        traced opcode.  Returns ``True`` if the thread yielded (triggers
        ``frame.f_locals`` refresh on 3.10-3.11 to work around the
        CPython LocalsToFast bug).
    remove_shadow_stack:
        ``(frame_id: int) -> None``.  Called on function return to clean
        up the shadow stack for *frame_id*.
    detect_io:
        When True, dynamically generated code (``<string>`` etc.) is
        skipped unconditionally rather than checking the caller chain.
    is_active:
        ``() -> bool`` or ``None``.  When provided, the trace function
        returns ``None`` (disabling tracing) if ``is_active()`` is False.
        Used by the sync scheduler to stop tracing when ``_finished`` or
        ``_error`` is set.
    """

    def trace(frame: Any, event: str, arg: Any) -> Any:
        if is_active is not None and not is_active():
            return None

        if event == "call":
            filename = frame.f_code.co_filename
            if _should_trace_file(filename):
                if _is_dynamic_code(filename) and not _is_cmdline_user_code(filename, frame.f_globals):
                    if detect_io:
                        return None
                    caller = frame.f_back
                    if caller is None or not _should_trace_file(caller.f_code.co_filename):
                        return None
                frame.f_trace_opcodes = True
                return trace
            return None

        if event == "opcode":
            thread_id = get_thread_id()
            if thread_id is not None:
                yielded = on_opcode(frame.f_code, frame.f_lasti, frame, thread_id)
                if yielded:
                    # CPython 3.10-3.11 bug workaround: after the trace
                    # callback returns, CPython calls
                    # PyFrame_LocalsToFast(frame, 1) which copies f_locals
                    # dict values back to cell/free variable cells.  If this
                    # thread waited in report_and_wait while another thread
                    # modified a shared cell, the stale f_locals snapshot
                    # would overwrite the new value.  Re-accessing
                    # frame.f_locals triggers PyFrame_FastToLocals, refreshing
                    # the snapshot so LocalsToFast writes back the current
                    # value.  Not needed on 3.12+ (PEP 667 removed
                    # LocalsToFast from the trace path).
                    frame.f_locals  # noqa: B018
            return trace

        if event == "return":
            remove_shadow_stack(id(frame))
            return trace

        return trace

    return trace


def make_monitoring_callbacks(
    *,
    get_thread_id: Any,
    on_opcode: Any,
    remove_shadow_stack: Any,
    detect_io: bool = False,
    is_active: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """Create ``sys.monitoring`` callbacks for opcode tracing.

    Returns ``(handle_py_start, handle_py_return, handle_py_unwind,
    handle_instruction)`` suitable for passing to
    :func:`setup_opcode_monitoring`.

    Parameters are the same as :func:`make_settrace_callback`.
    """
    mon = sys.monitoring

    def handle_py_start(code: Any, instruction_offset: int) -> Any:
        # Only use mon.DISABLE for code that should *never* be traced
        # (stdlib, site-packages, frontrun internals).  Do NOT disable
        # for transient conditions like scheduler._finished — DISABLE
        # permanently removes INSTRUCTION events from the code object.
        if not _should_trace_file(code.co_filename):
            return mon.DISABLE  # type: ignore[attr-defined]
        # In I/O-detection mode, skip dynamically generated code
        # (e.g. dataclass __init__ from exec/compile in libraries).
        if detect_io and code.co_filename.startswith("<"):
            return mon.DISABLE  # type: ignore[attr-defined]
        return None

    def handle_py_return(code: Any, instruction_offset: int, retval: Any) -> Any:
        if not _should_trace_file(code.co_filename):
            return None
        thread_id = get_thread_id()
        if thread_id is not None:
            frame = sys._getframe(1)
            remove_shadow_stack(id(frame))
        return None

    # A frame exiting via exception fires PY_UNWIND, not PY_RETURN.  The
    # cleanup is identical (remove the shadow stack keyed by id(frame), else a
    # reused frame id inherits a dead operand stack — finding 4), and neither
    # callback looks at its third argument (retval vs. exception).
    handle_py_unwind = handle_py_return

    def handle_instruction(code: Any, instruction_offset: int) -> Any:
        if is_active is not None and not is_active():
            return None
        if not _should_trace_file(code.co_filename):
            return None
        # Skip dynamically generated code (<string>, etc.) unless its
        # caller is user code.  Libraries use exec/compile internally
        # (dataclass __init__, SQLAlchemy methods) creating thousands of
        # scheduling points in non-user code.  In I/O mode, skip all
        # dynamic code unconditionally.  Exception: python -c mode,
        # where functions defined in the -c string ARE user code.
        if _is_dynamic_code(code.co_filename):
            frame = sys._getframe(1)
            if not _is_cmdline_user_code(code.co_filename, frame.f_globals):
                if detect_io:
                    return None
                caller = frame.f_back
                if caller is None or not _should_trace_file(caller.f_code.co_filename):
                    return None

        thread_id = get_thread_id()
        if thread_id is None:
            return None

        frame = sys._getframe(1)
        on_opcode(code, instruction_offset, frame, thread_id)
        return None

    return handle_py_start, handle_py_return, handle_py_unwind, handle_instruction


# ---------------------------------------------------------------------------
# High-level tracer-backend API
# ---------------------------------------------------------------------------
#
# This is the only API the rest of the codebase should use to install opcode-
# or line-level tracers.  It owns the dispatch between ``sys.settrace``
# (Python 3.10-3.11) and ``sys.monitoring`` (3.12+, mandatory on 3.14t free-
# threaded due to CPython issue #118415).
#
# Usage pattern for opcode tracing
# --------------------------------
# Opcode tracing is logically "global per scheduler".  ``sys.monitoring`` is
# globally registered and fires for every thread automatically; ``sys.settrace``
# is per-thread and must be installed inside each worker thread.
# The handle returned by ``start_opcode_trace`` captures both modes:
#
#     handle = start_opcode_trace(get_thread_id=..., on_opcode=..., ...)
#     try:
#         # In each worker thread:
#         install_thread_opcode_trace(handle)   # no-op on sys.monitoring
#         try:
#             ...do work...
#         finally:
#             uninstall_thread_opcode_trace(handle)
#     finally:
#         stop_opcode_trace(handle)
#
# Usage pattern for line tracing
# ------------------------------
# Line tracing always uses ``sys.settrace`` (no opcode events, so the
# free-threading bug does not apply).  Install per-thread:
#
#     install_thread_line_trace(trace_function)
#     try:
#         ...do work...
#     finally:
#         uninstall_thread_line_trace()


class OpcodeTraceHandle:
    """Opaque handle returned by :func:`start_opcode_trace`.

    Carries enough state to install/uninstall per-thread tracers (settrace
    backend) and to tear down the global monitoring tool (monitoring backend).
    """

    __slots__ = ("_settrace_callback", "_tool_id", "using_monitoring")

    def __init__(
        self,
        *,
        using_monitoring: bool,
        tool_id: int | None,
        settrace_callback: Any,
    ) -> None:
        self.using_monitoring = using_monitoring
        self._tool_id = tool_id
        self._settrace_callback = settrace_callback


def start_opcode_trace(
    *,
    get_thread_id: Any,
    on_opcode: Any,
    remove_shadow_stack: Any = None,
    detect_io: bool = False,
    is_active: Any = None,
    tool_name: str = "frontrun",
    tool_kind: ToolKind = "profiler",
    monitor_returns: bool = True,
) -> OpcodeTraceHandle:
    """Start opcode-level tracing using the appropriate backend.

    On Python 3.12+ this immediately installs ``sys.monitoring`` callbacks
    (which fire globally for all threads).  On Python 3.10-3.11 it only
    constructs a ``sys.settrace`` callback; per-thread activation happens
    via :func:`install_thread_opcode_trace`.

    Parameters
    ----------
    get_thread_id, on_opcode, remove_shadow_stack, detect_io, is_active:
        Forwarded to :func:`make_settrace_callback` /
        :func:`make_monitoring_callbacks`.  ``remove_shadow_stack`` defaults
        to a no-op for callers that don't track a shadow stack.
    tool_name, tool_kind, monitor_returns:
        Forwarded to :func:`setup_opcode_monitoring` (sys.monitoring path
        only).
    """

    def _noop_remove_shadow_stack(_frame_id: int) -> None:
        return None

    _remove_shadow_stack = remove_shadow_stack if remove_shadow_stack is not None else _noop_remove_shadow_stack

    if _USE_SYS_MONITORING:
        handle_py_start, handle_py_return, handle_py_unwind, handle_instruction = make_monitoring_callbacks(
            get_thread_id=get_thread_id,
            on_opcode=on_opcode,
            remove_shadow_stack=_remove_shadow_stack,
            detect_io=detect_io,
            is_active=is_active,
        )
        tool_id = setup_opcode_monitoring(
            tool_name=tool_name,
            handle_py_start=handle_py_start,
            handle_py_return=handle_py_return,
            handle_py_unwind=handle_py_unwind,
            handle_instruction=handle_instruction,
            tool_kind=tool_kind,
            monitor_returns=monitor_returns,
        )
        return OpcodeTraceHandle(using_monitoring=True, tool_id=tool_id, settrace_callback=None)

    trace_callback = make_settrace_callback(
        get_thread_id=get_thread_id,
        on_opcode=on_opcode,
        remove_shadow_stack=_remove_shadow_stack,
        detect_io=detect_io,
        is_active=is_active,
    )
    return OpcodeTraceHandle(using_monitoring=False, tool_id=None, settrace_callback=trace_callback)


def install_thread_opcode_trace(handle: OpcodeTraceHandle) -> None:
    """Install opcode tracing on the current thread.

    No-op when the sys.monitoring backend is in use (it is global, not
    per-thread).  On the sys.settrace backend, this calls
    ``sys.settrace(handle's callback)``.
    """
    if handle.using_monitoring:
        return
    sys.settrace(handle._settrace_callback)


def uninstall_thread_opcode_trace(handle: OpcodeTraceHandle) -> None:
    """Uninstall opcode tracing on the current thread.

    No-op when the sys.monitoring backend is in use.  On the sys.settrace
    backend, this calls ``sys.settrace(None)``.
    """
    if handle.using_monitoring:
        return
    sys.settrace(None)


def stop_opcode_trace(handle: OpcodeTraceHandle) -> None:
    """Stop opcode tracing and free any backend resources."""
    if handle.using_monitoring:
        teardown_opcode_monitoring(handle._tool_id)


def install_thread_line_trace(trace_fn: Any) -> None:
    """Install a sys.settrace line-level trace function on the current thread.

    Used by ``_trace_marker_runtime`` for marker-based scheduling.  Line
    tracing is per-thread and does not use opcode events, so the
    free-threading PEP 669 path does not apply.
    """
    sys.settrace(trace_fn)


def uninstall_thread_line_trace() -> None:
    """Remove the sys.settrace function from the current thread."""
    sys.settrace(None)
