"""Redis command classification for key-level conflict detection.

Classifies Redis commands into read/write operations and extracts the
key(s) each command accesses.  This enables DPOR to detect key-level
conflicts between threads/tasks operating on the same Redis keys.

Key extraction rules are derived from the official Redis ``commands.json``
key specifications (``begin_search`` / ``find_keys`` / access flags).  A
generic interpreter replaces per-command special-case logic.  Only a
handful of overrides remain for DPOR-specific semantics (transaction
control, pub/sub channel prefixing, Lua script atomicity).
"""

from __future__ import annotations

from typing import NamedTuple

from frontrun._redis_command_data import (
    _COMMAND_KEY_SPECS,
    _EVAL_CMDS,
    _KEYSPACE_READ_CMDS,
    _KEYSPACE_WRITE_CMDS,
    _NO_KEY_CMDS,
    _SUBCOMMAND_PARENTS,
    _TX_CONTROL_CMDS,
    _KeySpec,
)

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class RedisAccessResult(NamedTuple):
    """Result of classifying a Redis command."""

    read_keys: list[str]
    write_keys: list[str]
    is_transaction_control: bool  # MULTI, EXEC, DISCARD, WATCH, UNWATCH
    # Database-wide keyspace intent-lock kind (``"read"``, ``"write"`` or
    # ``None``).  FLUSHDB/FLUSHALL write the whole keyspace; per-key commands
    # and keyspace scans (KEYS/SCAN/RANDOMKEY/DBSIZE) read it.  This lets DPOR
    # detect FLUSH*-vs-key races without over-serializing ordinary key traffic
    # (which stays read-read on the keyspace resource).  See
    # ``_keyspace_kind`` and the report path in ``_redis_client.py``.
    keyspace: str | None = None


# ---------------------------------------------------------------------------
# Generic key-spec interpreter
# ---------------------------------------------------------------------------


def _extract_keys_from_spec(
    cmd_args: tuple[object, ...],
    begin: tuple[str | int, ...],
    find: tuple[str | int, ...],
) -> list[str]:
    """Extract keys from *cmd_args* using a single key-spec entry.

    The ``begin`` tuple locates the starting position; ``find`` enumerates
    keys from that position.  Both formats are documented above.

    Note: ``cmd_args`` does **not** include the command name.  The indices
    in the key-spec table are 1-based (position 1 = first argument after
    the command name), so we subtract 1 when indexing into *cmd_args*.
    """
    n_args = len(cmd_args)

    # --- Resolve begin position (0-based into cmd_args) ---
    if begin[0] == "idx":
        start = int(begin[1]) - 1  # 1-based → 0-based
        if start < 0 or start >= n_args:
            return []
    elif begin[0] == "kw":
        keyword = str(begin[1]).upper()
        startfrom = int(begin[2])
        # startfrom can be negative (search from end); convert to 0-based cmd_args index.
        if startfrom >= 0:
            search_start = max(startfrom - 1, 0)
        else:
            search_start = max(n_args + startfrom, 0)
        args_upper = [str(a).upper() for a in cmd_args]
        found = -1
        for j in range(search_start, n_args):
            if args_upper[j] == keyword:
                found = j
                break
        if found < 0:
            return []
        start = found + 1  # keys begin after the keyword
        if start >= n_args:
            return []
    else:
        return []

    # --- Enumerate keys from *start* ---
    if find[0] == "rng":
        lastkey = int(find[1])
        keystep = max(int(find[2]), 1)
        limit = int(find[3])

        if lastkey == 0:
            # Single key at start.
            return [str(cmd_args[start])]
        elif lastkey > 0:
            # Fixed number of additional keys.
            end = min(start + lastkey, n_args - 1)
            return [str(cmd_args[i]) for i in range(start, end + 1, keystep)]
        else:
            # lastkey < 0: keys extend to the end (adjusted by lastkey).
            # lastkey=-1 → up to and including the last arg
            # lastkey=-2 → up to and including the second-to-last arg
            end = n_args + lastkey + 1  # exclusive upper bound
            if end <= start:
                return []
            count = end - start
            # Apply limit divisor (e.g. XREAD: limit=2 means first half are keys).
            if limit >= 2:
                count = count // limit
            if count <= 0:
                return []
            return [str(cmd_args[start + i * keystep]) for i in range(count) if start + i * keystep < n_args]

    elif find[0] == "kn":
        keynumidx = int(find[1])
        firstkey_offset = int(find[2])
        keystep = max(int(find[3]), 1)

        numkeys_pos = start + keynumidx
        if numkeys_pos >= n_args:
            return []
        try:
            numkeys = int(str(cmd_args[numkeys_pos]))
        except (ValueError, TypeError):
            return []

        first_pos = start + firstkey_offset
        keys: list[str] = []
        for i in range(numkeys):
            pos = first_pos + i * keystep
            if pos >= n_args:
                break
            keys.append(str(cmd_args[pos]))
        return keys

    return []


def _lookup_key_specs(cmd_name_upper: str, cmd_args: tuple[object, ...]) -> list[_KeySpec] | None:
    """Look up key specs, trying subcommand dispatch first."""
    if cmd_args:
        sub = cmd_name_upper + " " + str(cmd_args[0]).upper()
        specs = _COMMAND_KEY_SPECS.get(sub)
        if specs is not None:
            return specs
    return _COMMAND_KEY_SPECS.get(cmd_name_upper)


# ---------------------------------------------------------------------------
# SORT special handling (STORE keyword has "unknown" key_spec in Redis)
# ---------------------------------------------------------------------------


def _parse_sort(cmd_args: tuple[object, ...]) -> RedisAccessResult:
    """Handle SORT which has an optional STORE dest that can't be expressed
    via static key_specs (Redis marks it as ``unknown``)."""
    if not cmd_args:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    first_key = str(cmd_args[0])
    # Walk args, skipping values after BY, GET, LIMIT to avoid treating
    # user-supplied patterns (e.g. "store") as the STORE keyword.
    i = 1
    while i < len(cmd_args):
        token = str(cmd_args[i]).upper()
        if token in ("BY", "GET"):
            i += 2
        elif token == "LIMIT":
            i += 3
        elif token == "STORE" and i + 1 < len(cmd_args):
            dest = str(cmd_args[i + 1])
            return RedisAccessResult(read_keys=[first_key], write_keys=[dest], is_transaction_control=False)
        else:
            i += 1
    return RedisAccessResult(read_keys=[first_key], write_keys=[], is_transaction_control=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _keyspace_kind(upper: str, result: RedisAccessResult) -> str | None:
    """Decide the database-wide keyspace intent-lock kind for a command.

    ``"write"`` for whole-keyspace mutations (FLUSHDB/FLUSHALL), ``"read"``
    for keyspace scans (KEYS/SCAN/RANDOMKEY/DBSIZE) and for any command that
    touches specific keys, and ``None`` otherwise (transaction control, Lua
    scripts, pub/sub, and plain server/connection commands).

    Key-touching commands take a *read* on the keyspace so ordinary key-key
    traffic stays read-read (no new conflicts); only FLUSH* takes a write,
    so it conflicts with every concurrent key access.
    """
    if result.is_transaction_control:
        # MULTI/EXEC/DISCARD/UNWATCH/WATCH and Lua scripts (atomic) — the
        # keyspace lock would only add spurious serialization here.
        return None
    if upper in _KEYSPACE_WRITE_CMDS:
        return "write"
    if upper in _KEYSPACE_READ_CMDS:
        return "read"
    # Pub/sub channels are not keyspace operations; ``channel:`` keys are a
    # disjoint namespace, so only real key accesses imply a keyspace read.
    if any(not k.startswith("channel:") for k in result.read_keys):
        return "read"
    if any(not k.startswith("channel:") for k in result.write_keys):
        return "read"
    return None


def parse_redis_access(cmd_name: str, cmd_args: tuple[object, ...]) -> RedisAccessResult:
    """Classify a Redis command and extract the key(s) it accesses.

    Args:
        cmd_name: The Redis command name (e.g. "GET", "SET", "HSET").
        cmd_args: The command arguments (after the command name).

    Returns:
        A ``RedisAccessResult`` with the read/write key sets and the
        database-wide keyspace intent-lock kind (see ``_keyspace_kind``).
    """
    upper = cmd_name.upper()
    result = _parse_redis_access_keys(upper, cmd_args)
    # Use the *normalized* command for keyspace classification (compound names
    # like "XGROUP CREATE" are split inside the helper, but our keyspace sets
    # only contain single-word commands, so the first word is what matters).
    keyspace = _keyspace_kind(upper.split(" ", 1)[0], result)
    if keyspace is None:
        return result
    return result._replace(keyspace=keyspace)


def _parse_redis_access_keys(upper: str, cmd_args: tuple[object, ...]) -> RedisAccessResult:
    """Extract read/write key sets for an *uppercased* command (without keyspace tagging)."""
    # Normalize compound command names from redis-py 4.2+ (e.g.
    # "XGROUP CREATE") by splitting the subcommand back into cmd_args.
    if " " in upper:
        parts = upper.split(" ", 1)
        upper = parts[0]
        cmd_args = (parts[1],) + cmd_args

    # --- Transaction control (no keys) ---
    if upper in _TX_CONTROL_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=True)

    # WATCH — reads keys for conflict detection, flagged as transaction control.
    if upper == "WATCH":
        keys = [str(a) for a in cmd_args]
        return RedisAccessResult(read_keys=keys, write_keys=[], is_transaction_control=True)

    # EVAL/EVALSHA/FCALL — atomic scripts, treated as transaction control (defect #8).
    if upper in _EVAL_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=True)

    # Pub/Sub — channels, not keys.  Prefix with "channel:" to distinguish.
    # Includes sharded pub/sub (SPUBLISH, SSUBSCRIBE, SUNSUBSCRIBE).
    if upper in ("PUBLISH", "SPUBLISH") and cmd_args:
        return RedisAccessResult(read_keys=[], write_keys=[f"channel:{cmd_args[0]}"], is_transaction_control=False)
    if upper in ("SUBSCRIBE", "PSUBSCRIBE", "SSUBSCRIBE"):
        return RedisAccessResult(
            read_keys=[f"channel:{a}" for a in cmd_args], write_keys=[], is_transaction_control=False
        )
    if upper in ("UNSUBSCRIBE", "PUNSUBSCRIBE", "SUNSUBSCRIBE"):
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    # SORT/SORT_RO — STORE keyword has "unknown" key_spec; use dedicated parser.
    if upper in ("SORT", "SORT_RO"):
        return _parse_sort(cmd_args)

    # Server/connection commands — no key-level conflicts.
    if upper in _NO_KEY_CMDS:
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    # --- Generic key-spec extraction ---
    specs = _lookup_key_specs(upper, cmd_args)

    if specs is None:
        # If this is a parent that has subcommand variants (e.g. OBJECT, XGROUP)
        # but the specific subcommand wasn't found, return no keys.
        if upper in _SUBCOMMAND_PARENTS:
            return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)
        # Truly unknown command with args → conservative fallback: treat first arg as write key.
        if cmd_args:
            return RedisAccessResult(read_keys=[], write_keys=[str(cmd_args[0])], is_transaction_control=False)
        return RedisAccessResult(read_keys=[], write_keys=[], is_transaction_control=False)

    read_keys: list[str] = []
    write_keys: list[str] = []
    for begin, find, is_read, is_write in specs:
        keys = _extract_keys_from_spec(cmd_args, begin, find)
        if is_read:
            read_keys.extend(keys)
        if is_write:
            write_keys.extend(keys)

    if upper == "MIGRATE":
        # Redis allows an empty-string positional placeholder when the actual
        # keys are supplied via the KEYS clause. That placeholder is not a
        # real key and must not create false conflicts.
        read_keys = [key for key in read_keys if key != ""]
        write_keys = [key for key in write_keys if key != ""]

    return RedisAccessResult(read_keys=read_keys, write_keys=write_keys, is_transaction_control=False)
