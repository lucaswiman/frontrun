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
import ipaddress
import os
import socket
import threading
import weakref
from collections.abc import Callable, Generator, Iterator
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._deadlock import SchedulerAbort
from frontrun._io_detection import _io_tls, external_operation_scope, get_io_reporter
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


def _redis_server_keyspace_resource_id(server_scope: str) -> str:
    """Build the server-wide intent resource used to model ``FLUSHALL``."""
    return f"redis:server-keyspace:server={server_scope}"


def _redis_server_pubsub_resource_id(server_scope: str) -> str:
    """Build a server-wide intent resource for overlapping Pub/Sub operations."""
    return f"redis:pubsub:server={server_scope}"


def _redis_arg_text(value: object) -> str:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "surrogateescape")
    return str(value)


# Cached name → canonical-address resolution.  Identity must be stable for
# a whole session (record and replay resolve to the same string) and must
# never cost a network lookup per access, so each name is resolved at most
# once per process.
_canonical_host_cache: dict[str, str] = {}


def _canonical_host(host: str) -> str:
    """Resolve *host* to a stable canonical server-address string.

    Textual aliases of one server (``localhost`` vs ``127.0.0.1``) must map
    to a single identity — otherwise two clients reaching the same Redis
    instance get disjoint resources and DPOR misses their dependency.
    Resolution picks the lexicographically first resolved address for
    determinism and folds every loopback address to ``127.0.0.1``.  On
    resolution failure the raw string is kept (the previous behavior).
    """
    cached = _canonical_host_cache.get(host)
    if cached is not None:
        return cached
    canonical = host
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addresses = sorted({str(info[4][0]) for info in infos})
        if addresses:
            canonical = addresses[0]
    except (OSError, UnicodeError):  # unresolvable or non-DNS name → keep raw
        pass
    try:
        if ipaddress.ip_address(canonical).is_loopback:
            canonical = "127.0.0.1"
    except ValueError:
        pass
    _canonical_host_cache[host] = canonical
    return canonical


def _query_unix_socket_tcp_port(path: str) -> str | None:
    """Ask the Redis server behind *path* for its TCP port (RESP2, suppressed I/O)."""
    if not hasattr(socket, "AF_UNIX"):
        return None
    try:
        with _suppress_endpoint_io(), contextlib.closing(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)) as sock:
            sock.settimeout(1.0)
            sock.connect(path)
            sock.sendall(b"*3\r\n$6\r\nCONFIG\r\n$3\r\nGET\r\n$4\r\nport\r\n")
            data = b""
            while data.count(b"\r\n") < 5 and len(data) < 512:
                if data.startswith(b"-") and b"\r\n" in data:
                    return None  # error reply (CONFIG disabled, NOAUTH, ...)
                chunk = sock.recv(256)
                if not chunk:
                    break
                data += chunk
    except OSError:
        return None
    lines = data.split(b"\r\n")
    if len(lines) < 5 or lines[0] != b"*2" or lines[2].lower() != b"port":
        return None
    port = lines[4].decode("ascii", "replace")
    return port if port.isdigit() and port != "0" else None


# Unix-socket pools carry no host/port.  Resolve each socket path once to the
# TCP identity of the server behind it (CONFIG GET port over the socket
# itself) so a unix-socket client and a TCP client reaching the same server
# share resources.  On any failure keep a distinct per-path identity rather
# than the old localhost:6379 default, which claimed an endpoint the socket
# never touches.  Cached per real path: deterministic within a session, no
# per-access I/O.
_unix_path_server_parts: dict[str, tuple[str, str]] = {}


def _unix_socket_server_parts(path: str) -> tuple[str, str]:
    real_path = os.path.realpath(path)
    cached = _unix_path_server_parts.get(real_path)
    if cached is None:
        tcp_port = _query_unix_socket_tcp_port(real_path)
        cached = ("localhost", tcp_port) if tcp_port is not None else (f"unix:{real_path}", "0")
        _unix_path_server_parts[real_path] = cached
    return cached


# Live SELECT tracking: ``connection_kwargs["db"]`` goes stale once a client
# issues SELECT, so later commands would be attributed to the wrong database
# (a missed dependency).  A single-connection client switches exactly; on a
# pooled client only one pooled connection switched, so the configured and
# selected databases are all kept as candidates (sound over-approximation).
# State lives on the client object itself when possible; the id-keyed
# registries (weakref-evicted, mirroring
# ``_sql_db_scope._register_connection_db_scope``) are a fallback for
# non-settable clients.  A client that is neither settable nor weakref-able
# stays untracked: id() values are reusable, and a bare id-keyed entry would
# hand a dead client's database to whatever object next occupies the address.
_EXACT_DB_ATTR = "_frontrun_selected_db"
_POSSIBLE_DBS_ATTR = "_frontrun_possible_dbs"
_client_exact_dbs: dict[int, str] = {}
_client_possible_dbs: dict[int, set[str]] = {}


def _set_client_scope_state(client: Any, attr: str, registry: dict[int, Any], value: Any) -> bool:
    """Attach SELECT state to *client*; return False if it cannot be tracked safely."""
    try:
        setattr(client, attr, value)
        return True
    except (AttributeError, TypeError):
        pass
    key = id(client)
    try:
        weakref.finalize(client, registry.pop, key, None)
    except TypeError:
        return False
    registry[key] = value
    return True


def _get_client_scope_state(client: Any, attr: str, registry: dict[int, Any]) -> Any:
    value = getattr(client, attr, None)
    if value is not None:
        return value
    return registry.get(id(client))


def _record_client_select(client: Any, db: object) -> None:
    """Record a live SELECT so later accesses land in the right database scope."""
    db_text = _redis_arg_text(db)
    if getattr(client, "connection", None) is not None:  # single_connection_client
        _set_client_scope_state(client, _EXACT_DB_ATTR, _client_exact_dbs, db_text)
        return
    possible = _get_client_scope_state(client, _POSSIBLE_DBS_ATTR, _client_possible_dbs)
    if possible is None:
        possible = set()
        if not _set_client_scope_state(client, _POSSIBLE_DBS_ATTR, _client_possible_dbs, possible):
            return
    possible.add(db_text)


def _get_redis_scope_parts(client: Any) -> tuple[str, str, str] | None:
    """Extract ``(host, port, database)`` strings from a Redis client."""
    pool = getattr(client, "connection_pool", None)
    if pool is None:
        return None
    kwargs = getattr(pool, "connection_kwargs", {})
    exact_db = _get_client_scope_state(client, _EXACT_DB_ATTR, _client_exact_dbs)
    db = exact_db if isinstance(exact_db, str) else _redis_arg_text(kwargs.get("db", 0))
    path = kwargs.get("path")
    if path is not None:
        host, port = _unix_socket_server_parts(_redis_arg_text(path))
        return host, port, db
    host = _redis_arg_text(kwargs.get("host", "localhost"))
    port = _redis_arg_text(kwargs.get("port", 6379))
    return host, port, db


def _format_redis_db_scope(host: object, port: object, db: object) -> str:
    return f"redis:{_canonical_host(_redis_arg_text(host))}:{_redis_arg_text(port)}/{_redis_arg_text(db)}"


def _format_redis_server_scope(host: object, port: object) -> str:
    return f"redis:{_canonical_host(_redis_arg_text(host))}:{_redis_arg_text(port)}"


def _get_redis_db_scope(client: Any) -> str | None:
    """Extract a stable database scope from a Redis client object."""
    parts = _get_redis_scope_parts(client)
    return _format_redis_db_scope(*parts) if parts is not None else None


def _find_redis_option(cmd_args: tuple[object, ...], option: str) -> object | None:
    """Return a COPY option value after its required source/destination keys."""
    for index, arg in enumerate(cmd_args[2:-1], start=2):
        value = _redis_arg_text(arg)
        if value.upper() == option:
            return cmd_args[index + 1]
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
            cmd_name = _redis_arg_text(cmd_args_full[0])
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
    upper = cmd_name.upper().split(" ", 1)[0]

    # Track live database switches even without a reporter (setup and replay
    # both run reporter-less); otherwise exploration and replay would derive
    # different scopes for the same client and anchors would misalign.
    if upper == "SELECT" and cmd_args and client is not None:
        _record_client_select(client, cmd_args[0])

    reporter = get_io_reporter()
    if reporter is None:
        return False

    pubsub_commands = {
        "PUBLISH",
        "SPUBLISH",
        "SUBSCRIBE",
        "PSUBSCRIBE",
        "SSUBSCRIBE",
        "UNSUBSCRIBE",
        "PUNSUBSCRIBE",
        "SUNSUBSCRIBE",
    }
    access = parse_redis_access(cmd_name, cmd_args)

    # Transaction control — no key-level reporting needed.
    if access.is_transaction_control and not access.read_keys and not access.write_keys:
        return True  # Still suppress endpoint I/O for protocol overhead.

    if (
        not access.read_keys
        and not access.write_keys
        and access.keyspace is None
        and upper not in pubsub_commands
        and upper != "PUBSUB"
    ):
        return False

    scope_parts = _get_redis_scope_parts(client) if client is not None else None
    db_scope = _format_redis_db_scope(*scope_parts) if scope_parts is not None else None

    key_accesses: list[tuple[str, str, str | None]] = []
    keyspace_accesses: set[tuple[str | None, str]] = set()

    if upper == "COPY" and len(cmd_args) >= 2:
        destination_scope = db_scope
        destination_db = _find_redis_option(cmd_args, "DB")
        if scope_parts is not None and destination_db is not None:
            destination_scope = _format_redis_db_scope(scope_parts[0], scope_parts[1], destination_db)
        key_accesses.append((access.read_keys[0], "read", db_scope))
        key_accesses.append((access.write_keys[0], "write", destination_scope))
    elif upper == "MOVE" and len(cmd_args) >= 2:
        destination_scope = db_scope
        if scope_parts is not None:
            destination_scope = _format_redis_db_scope(scope_parts[0], scope_parts[1], cmd_args[1])
        key = access.write_keys[0]
        key_accesses.extend(((key, "read", db_scope), (key, "write", db_scope), (key, "write", destination_scope)))
    elif upper == "MIGRATE" and len(cmd_args) >= 4:
        key_accesses.extend((key, "read", db_scope) for key in access.read_keys)
        key_accesses.extend((key, "write", db_scope) for key in access.write_keys)
        destination_scope = _format_redis_db_scope(cmd_args[0], cmd_args[1], cmd_args[3])
        key_accesses.extend((key, "write", destination_scope) for key in access.write_keys)
    elif upper in pubsub_commands:
        channel_scope = _format_redis_server_scope(scope_parts[0], scope_parts[1]) if scope_parts is not None else None
        key_accesses.extend((key, "read", channel_scope) for key in access.read_keys)
        key_accesses.extend((key, "write", channel_scope) for key in access.write_keys)
        if channel_scope is not None:
            # Pattern subscriptions overlap dynamically named channels, and
            # zero-argument unsubscribe affects an unknown set. Keep exact
            # channel resources, plus this conservative server-wide intent.
            pubsub_kind = "write" if upper in {"PUBLISH", "SPUBLISH"} else "read"
            reporter(_redis_server_pubsub_resource_id(channel_scope), pubsub_kind)
    elif upper == "PUBSUB":
        if scope_parts is not None:
            reporter(
                _redis_server_pubsub_resource_id(_format_redis_server_scope(scope_parts[0], scope_parts[1])),
                "write",
            )
    elif upper == "SWAPDB":
        if len(cmd_args) >= 2 and scope_parts is not None:
            keyspace_accesses.add((_format_redis_db_scope(scope_parts[0], scope_parts[1], cmd_args[0]), "write"))
            keyspace_accesses.add((_format_redis_db_scope(scope_parts[0], scope_parts[1], cmd_args[1]), "write"))
        else:
            keyspace_accesses.add((db_scope, "write"))
    else:
        key_accesses.extend((key, "read", db_scope) for key in access.read_keys)
        key_accesses.extend((key, "write", db_scope) for key in access.write_keys)
        if access.keyspace is not None:
            keyspace_accesses.add((db_scope, access.keyspace))

    for key, kind, scope in key_accesses:
        reporter(_redis_resource_id(key, db_scope=scope), kind)
        if access.keyspace is not None:
            keyspace_accesses.add((scope, "read"))

    for scope, kind in sorted(keyspace_accesses, key=lambda item: (item[0] or "", item[1])):
        reporter(_redis_keyspace_resource_id(db_scope=scope), kind)

    # A pooled client that issued SELECT may run any command on either the
    # configured or a selected database (only one pooled connection actually
    # switched).  Mirror the primary-scope accesses into every candidate
    # scope so no ordering is missed (sound over-approximation).
    if client is not None and scope_parts is not None:
        possible_dbs = _get_client_scope_state(client, _POSSIBLE_DBS_ATTR, _client_possible_dbs)
        if possible_dbs:
            candidate_scopes = {
                _format_redis_db_scope(scope_parts[0], scope_parts[1], candidate_db) for candidate_db in possible_dbs
            }
            candidate_scopes.discard(db_scope)
            for extra_scope in sorted(candidate_scopes):
                for key, kind, scope in key_accesses:
                    if scope == db_scope:
                        reporter(_redis_resource_id(key, db_scope=extra_scope), kind)
                for scope, kind in sorted(keyspace_accesses, key=lambda item: (item[0] or "", item[1])):
                    if scope == db_scope:
                        reporter(_redis_keyspace_resource_id(db_scope=extra_scope), kind)

    # FLUSHALL mutates every database on a server.  A server-wide intent
    # resource keeps ordinary traffic read-read while making FLUSHALL conflict
    # with accesses from clients selected into any database on that server.
    server_scopes: set[str] = set()
    for scope, _kind in keyspace_accesses:
        if scope is not None:
            server_scopes.add(scope.rsplit("/", 1)[0])
    if upper == "MIGRATE" and len(cmd_args) >= 2:
        server_scopes.add(_format_redis_server_scope(cmd_args[0], cmd_args[1]))
    server_kind = "write" if upper == "FLUSHALL" else "read"
    for server_scope in sorted(server_scopes):
        reporter(_redis_server_keyspace_resource_id(server_scope), server_kind)

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
    cmd_name = _redis_arg_text(args[0])
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
    if not args:
        return original_method(self, *args, **kwargs)
    with external_operation_scope():
        return _intercept_execute_command_scoped(original_method, self, *args, **kwargs)


def _intercept_execute_command_scoped(
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
        first_key = _redis_arg_text(cmd_args[0]) if cmd_args else ""
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
    with external_operation_scope():
        return _intercept_pipeline_execute_scoped(original_method, self, *args, **kwargs)


def _intercept_pipeline_execute_scoped(
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
    """Patch redis-py command, immediate transaction, and pipeline execution."""
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
