"""Redis client monkey-patching for key-level conflict detection.

Intercepts Redis command execution on major Python Redis clients to
extract key-level read/write sets.  Reports each key as a separate
resource to the I/O reporter, suppressing the coarser endpoint-level
socket I/O reports.

Follows the same monkey-patching pattern as ``_sql_cursor.py``.

Supported sync clients:

* **redis-py** (``redis.Redis``, ``redis.StrictRedis``)

The interception hooks into the low-level ``execute_command`` method
that all high-level Redis methods funnel through, so every Redis
operation is captured regardless of whether the user calls ``r.get()``,
``r.set()``, ``r.hset()``, etc.
"""

from __future__ import annotations

import contextlib
import importlib
import threading
from collections.abc import Callable, Generator, Iterator
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._deadlock import SchedulerAbort
from frontrun._io_detection import _io_tls, get_io_reporter
from frontrun._io_detection import get_dpor_context as _get_dpor_context
from frontrun._patching import patch_method, restore_patches, wrap_method_metadata
from frontrun._redis_parsing import parse_redis_access
from frontrun._redis_patch_registry import SYNC_REDIS_TARGETS

_suppress_tids: set[int] = set()
_suppress_lock = _rt.lock()

# When True, Redis interception forces scheduling points even without
# the IO reporter.  Used during counterexample reproduction to enforce
# the DPOR schedule at Redis command boundaries.  See defect #9.
_redis_replay_mode = False


@contextlib.contextmanager
def _suppress_endpoint_io() -> Generator[None, None, None]:
    """Temporarily suppress endpoint-level I/O for the current thread."""
    tid = threading.get_native_id()
    _io_tls._redis_suppress = True
    with _suppress_lock:
        _suppress_tids.add(tid)
    try:
        yield
    finally:
        with _suppress_lock:
            _suppress_tids.discard(tid)
        _io_tls._redis_suppress = False


def is_redis_tid_suppressed(tid: int) -> bool:
    """Check if a thread ID is currently suppressed (for LD_PRELOAD bridge)."""
    with _suppress_lock:
        return tid in _suppress_tids


# ---------------------------------------------------------------------------
# Resource ID construction
# ---------------------------------------------------------------------------


def _redis_resource_id(key: str, *, db_scope: str | None = None) -> str:
    """Build a resource ID for a Redis key."""
    resource = f"redis:{key}"
    if db_scope is not None:
        resource = f"{resource}:db={db_scope}"
    return resource


def _redis_keyspace_resource_id(db_scope: str | None = None) -> str:
    """Build the resource ID for the database-wide keyspace intent-lock.

    FLUSHDB/FLUSHALL take a write here and per-key commands take a read, so
    DPOR detects FLUSH*-vs-key races.  Scoped per database (matching key
    resources) so a flush only conflicts with traffic in the same db.
    """
    return _redis_resource_id("keyspace", db_scope=db_scope)


def _get_redis_db_scope(client: Any) -> str | None:
    """Extract a stable database scope from a Redis client object."""
    # redis-py exposes connection_pool.connection_kwargs
    pool = getattr(client, "connection_pool", None)
    if pool is not None:
        kwargs = getattr(pool, "connection_kwargs", {})
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 6379)
        db = kwargs.get("db", 0)
        return f"redis:{host}:{port}/{db}"
    return None


# ---------------------------------------------------------------------------
# Pipeline command parsing
# ---------------------------------------------------------------------------


def _iter_pipeline_commands(pipeline: Any) -> Iterator[tuple[str, tuple[Any, ...]]]:
    """Yield ``(command_name, command_args)`` from a Redis pipeline."""
    command_stack = getattr(pipeline, "command_stack", [])
    for cmd in command_stack:
        # redis-py Pipeline stores commands as PipelineCommand or tuples.
        if hasattr(cmd, "args"):
            cmd_args_full = cmd.args
        elif isinstance(cmd, (list, tuple)):
            cmd_args_full = cmd[0] if cmd and isinstance(cmd[0], (list, tuple)) else cmd
        else:
            continue
        if cmd_args_full:
            cmd_name = str(cmd_args_full[0])
            cmd_cmd_args = tuple(cmd_args_full[1:])
            yield cmd_name, cmd_cmd_args


def _report_pipeline_commands(pipeline: Any) -> bool:
    """Parse a redis Pipeline's command_stack and report all queued accesses.

    Returns ``True`` if any command was reported to the I/O layer.
    """
    reported = False
    for cmd_name, cmd_args in _iter_pipeline_commands(pipeline):
        if _report_redis_access(cmd_name, cmd_args, client=pipeline):
            reported = True
    return reported


# ---------------------------------------------------------------------------
# Core interception
# ---------------------------------------------------------------------------


def _report_redis_access(
    cmd_name: str,
    cmd_args: tuple[object, ...],
    *,
    client: Any = None,
) -> bool:
    """Parse a Redis command and report key accesses to the per-thread reporter.

    Returns ``True`` if any Redis-level reporting was performed (which means
    endpoint-level I/O should be suppressed for the subsequent Redis call).
    """
    reporter = get_io_reporter()
    if reporter is None:
        return False

    access = parse_redis_access(cmd_name, cmd_args)

    # Transaction control — no key-level reporting needed.
    if access.is_transaction_control and not access.read_keys and not access.write_keys:
        return True  # Still suppress endpoint I/O for protocol overhead.

    if not access.read_keys and not access.write_keys and access.keyspace is None:
        return False

    db_scope = _get_redis_db_scope(client) if client is not None else None

    for key in access.read_keys:
        res_id = _redis_resource_id(key, db_scope=db_scope)
        reporter(res_id, "read")

    for key in access.write_keys:
        res_id = _redis_resource_id(key, db_scope=db_scope)
        reporter(res_id, "write")

    # Database-wide keyspace intent-lock: FLUSHDB/FLUSHALL write the whole
    # keyspace, while per-key commands and keyspace scans read it.  This lets
    # DPOR explore FLUSH*-vs-key races without over-serializing key traffic.
    if access.keyspace is not None:
        reporter(_redis_keyspace_resource_id(db_scope=db_scope), access.keyspace)

    return True


def _replay_needs_scheduling_point(access: Any) -> bool:
    """Whether a Redis command needs a replay scheduling point.

    During replay the I/O reporter is ``None``, so ``_report_redis_access``
    reports nothing and ``reported`` is ``False``.  We must still recreate a
    scheduling point for every command that created one during exploration —
    i.e. any command carrying a key-level *or* keyspace-level intent-lock —
    so the io-anchored replay schedule stays aligned.  Connection-setup
    commands (AUTH, SELECT, CLIENT SETNAME, ...) carry none and are skipped.
    """
    return bool(access.read_keys or access.write_keys or access.keyspace is not None)


def _parse_and_report_execute_command(
    args: tuple[Any, ...],
    client: Any,
) -> tuple[str, tuple[Any, ...], bool] | None:
    """Parse ``execute_command`` *args* and report key accesses.

    Returns ``(cmd_name, cmd_args, reported)`` for the command, or ``None``
    if *args* is empty (in which case the caller should pass through to the
    original method without any reporting).  Shared by the sync and async
    ``execute_command`` interceptors.
    """
    if not args:
        return None
    cmd_name = str(args[0])
    cmd_args = args[1:]
    reported = _report_redis_access(cmd_name, cmd_args, client=client)
    return cmd_name, cmd_args, reported


def _run_sync_dpor_envelope(
    execute_fn: Callable[[], Any],
    resource_id: str,
    reported: bool,
    needs_scheduling_point: bool,
) -> Any:
    """Run *execute_fn* inside the sync DPOR scheduling envelope.

    Bookends the call with ``before_io`` / ``after_io`` when
    *needs_scheduling_point* is True, and suppresses endpoint-level I/O
    when *reported* is True.  Shared by ``_intercept_execute_command``
    and ``_intercept_pipeline_execute``.
    """
    dpor_ctx = None
    if needs_scheduling_point:
        dpor_ctx = _get_dpor_context()
        if dpor_ctx is not None and dpor_ctx[0].before_io(dpor_ctx[1], resource_id) is False:
            # An explicit False means the scheduler denied the boundary;
            # running the command anyway would mutate real Redis outside any
            # schedule.
            raise SchedulerAbort("scheduler aborted before Redis execution")

    try:
        if reported:
            with _suppress_endpoint_io():
                return execute_fn()
        else:
            return execute_fn()
    finally:
        if dpor_ctx is not None:
            dpor_ctx[0].after_io(dpor_ctx[1], resource_id)


def _intercept_execute_command(
    original_method: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Intercept redis.Redis.execute_command to report key-level accesses."""
    parsed = _parse_and_report_execute_command(args, self)
    if parsed is None:
        return original_method(self, *args, **kwargs)

    cmd_name, cmd_args, reported = parsed

    # In replay mode (defect #9 fix), force a scheduling point even
    # without IO reporting so the replay scheduler can enforce the
    # interleaving at Redis command boundaries.  Only do this for
    # data commands (those that have keys), not connection-setup
    # commands (AUTH, SELECT, CLIENT SETNAME, etc.) which didn't
    # create scheduling points during exploration.
    needs_scheduling_point = reported
    if not needs_scheduling_point and _redis_replay_mode:
        access = parse_redis_access(cmd_name, cmd_args)
        needs_scheduling_point = _replay_needs_scheduling_point(access)

    # Build a structured resource ID for IO-anchored replay.  Fields are
    # joined with the unit separator (\x1f) so the replay scheduler can
    # split them unambiguously (keys and db scopes may contain colons).
    # The key field may embed run-specific random values (UUIDs, ULIDs
    # from e.g. redis-om primary keys); _IOAnchoredReplayScheduler
    # canonicalises it via bijective rebinding so replays whose setup()
    # generates fresh keys still match the recorded anchors.
    resource_id = ""
    if needs_scheduling_point:
        db_scope = _get_redis_db_scope(self) or ""
        first_key = str(cmd_args[0]) if cmd_args else ""
        resource_id = "\x1f".join(("redis", cmd_name, first_key, db_scope))

    return _run_sync_dpor_envelope(
        lambda: original_method(self, *args, **kwargs),
        resource_id,
        reported,
        needs_scheduling_point,
    )


def _intercept_pipeline_execute(
    original_method: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Intercept redis.Pipeline.execute to report all queued commands."""
    reported = _report_pipeline_commands(self)

    # In replay mode (defect #9 fix), force a scheduling point even
    # without IO reporting so the replay scheduler can enforce the
    # interleaving at Redis command boundaries.  Without this, pipeline
    # commands (used by get_many/set_many in e.g. Flask-Caching @memoize)
    # silently skip their scheduling points during replay, causing the
    # schedule to misalign.  See defect #10.
    needs_scheduling_point = reported
    if not needs_scheduling_point and _redis_replay_mode:
        needs_scheduling_point = any(
            _replay_needs_scheduling_point(parse_redis_access(cmd_name, cmd_args))
            for cmd_name, cmd_args in _iter_pipeline_commands(self)
        )

    return _run_sync_dpor_envelope(
        lambda: original_method(self, *args, **kwargs),
        "redis:PIPELINE",
        reported,
        needs_scheduling_point,
    )


# ---------------------------------------------------------------------------
# Global patching state
# ---------------------------------------------------------------------------

_redis_patched = False
_PATCHES: list[tuple[Any, str, Any]] = []
_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}


def _patch_redis_py() -> None:
    """Patch redis-py ``Redis.execute_command`` and ``Pipeline.execute``."""
    for target in SYNC_REDIS_TARGETS:
        try:
            module = importlib.import_module(target.module_name)
            target_cls = getattr(module, target.class_name)
        except (ImportError, AttributeError):
            continue

        def _make_patched(orig: Any, *, _target: Any = target_cls, _method_name: str = target.method_name) -> Any:
            if _method_name == "execute":

                def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return _intercept_pipeline_execute(orig, self, *args, **kwargs)
            else:

                def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return _intercept_execute_command(orig, self, *args, **kwargs)

            return wrap_method_metadata(_patched, orig, name=_method_name)

        patch_method(
            target_cls,
            target.method_name,
            originals=_ORIGINAL_METHODS,
            patches=_PATCHES,
            make_wrapper=_make_patched,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch_redis() -> None:
    """Monkey-patch Redis clients for key-level conflict detection."""
    global _redis_patched  # noqa: PLW0603
    if _redis_patched:
        return
    _patch_redis_py()
    _redis_patched = True


def unpatch_redis() -> None:
    """Restore original Redis client methods."""
    global _redis_patched  # noqa: PLW0603
    if not _redis_patched:
        return
    restore_patches(_PATCHES)
    _PATCHES.clear()
    _ORIGINAL_METHODS.clear()
    _redis_patched = False


def set_redis_replay_mode(enabled: bool) -> None:
    """Enable/disable Redis replay mode for counterexample reproduction.

    When enabled, Redis command interception creates scheduling points
    even without the IO reporter, so the replay scheduler can enforce
    the DPOR schedule at Redis command boundaries.  See defect #9.
    """
    global _redis_replay_mode  # noqa: PLW0603
    _redis_replay_mode = enabled
