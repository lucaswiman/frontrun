"""Redis command classification data for key-level conflict detection.

Contains the key-spec table (derived from Redis ``commands.json``) and the
command classification sets used by ``_redis_parsing.parse_redis_access``.
Keeping this data separate from the parsing logic makes it easy to audit
coverage and add new commands without touching the interpreter.

Key-spec entry format: ``(begin_search, find_keys, is_read, is_write)``

begin_search formats:
  ("idx", N)                  — keys start at argument index N (1-based
                                from the full Redis protocol; we subtract
                                1 to index into cmd_args)
  ("kw", keyword, startfrom)  — keys follow the first occurrence of
                                *keyword* (searched from ``startfrom``)

find_keys formats:
  ("rng", lastkey, keystep, limit)
      lastkey=0  → single key
      lastkey=-1 → all remaining args (limit divides count if >1)
      lastkey=-2 → all args except the last
      lastkey=N>0 → N additional keys after the first
      keystep    → stride between keys (1=every arg, 2=every other)
      limit      → divisor for remaining-arg count (0 or 1 = no limit)
  ("kn", keynumidx, firstkey, keystep)
      keynumidx → offset from begin to the arg holding the key count
      firstkey  → offset from begin to the first actual key
      keystep   → stride between keys
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

_KeySpec = tuple[tuple[str | int, ...], tuple[str | int, ...], bool, bool]

# ---------------------------------------------------------------------------
# Key-spec data table — derived from Redis commands.json
# ---------------------------------------------------------------------------

# fmt: off
_COMMAND_KEY_SPECS: dict[str, list[_KeySpec]] = {
    "APPEND": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "BITCOUNT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "BITFIELD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "BITFIELD_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "BITOP": [(("idx", 2), ("rng", 0, 1, 0), False, True), (("idx", 3), ("rng", -1, 1, 0), True, False)],
    "BITPOS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "BLMOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "BLMPOP": [(("idx", 2), ("kn", 0, 1, 1), True, True)],
    "BLPOP": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "BRPOP": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "BRPOPLPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "BZMPOP": [(("idx", 2), ("kn", 0, 1, 1), True, True)],
    "BZPOPMAX": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "BZPOPMIN": [(("idx", 1), ("rng", -2, 1, 0), True, True)],
    "COPY": [(("idx", 1), ("rng", 0, 1, 0), True, False), (("idx", 2), ("rng", 0, 1, 0), False, True)],
    "DECR": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "DECRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "DEL": [(("idx", 1), ("rng", -1, 1, 0), False, True)],
    "DUMP": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "EXISTS": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "EXPIRE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "EXPIREAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "EXPIRETIME": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "FCALL": [(("idx", 2), ("kn", 0, 1, 1), True, True)],
    "FCALL_RO": [(("idx", 2), ("kn", 0, 1, 1), True, False)],
    "GEOADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "GEODIST": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOHASH": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOPOS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEORADIUS": [(("idx", 1), ("rng", 0, 1, 0), True, False), (("kw", "STORE", 6), ("rng", 0, 1, 0), False, True), (("kw", "STOREDIST", 6), ("rng", 0, 1, 0), False, True)],
    "GEORADIUSBYMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False), (("kw", "STORE", 5), ("rng", 0, 1, 0), False, True), (("kw", "STOREDIST", 5), ("rng", 0, 1, 0), False, True)],
    "GEORADIUSBYMEMBER_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEORADIUS_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOSEARCH": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GEOSEARCHSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", 0, 1, 0), True, False)],
    "GET": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GETBIT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GETDEL": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "GETEX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "GETRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "GETSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HDEL": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HEXISTS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HGET": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HGETALL": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HINCRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HINCRBYFLOAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HKEYS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HMGET": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HMSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HRANDFIELD": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HSCAN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HSETNX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "HSTRLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "HVALS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "INCR": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "INCRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "INCRBYFLOAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LCS": [(("idx", 1), ("rng", 1, 1, 0), True, False)],
    "LINDEX": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LINSERT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LMOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "LMPOP": [(("idx", 1), ("kn", 0, 1, 1), True, True)],
    "LPOP": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LPOS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LPUSHX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "LREM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LSET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "LTRIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "MEMORY USAGE": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "MGET": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "MIGRATE": [(("idx", 3), ("rng", 0, 1, 0), True, True), (("kw", "KEYS", 5), ("rng", -1, 1, 0), True, True)],
    "MOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "MSET": [(("idx", 1), ("rng", -1, 2, 0), False, True)],
    "MSETNX": [(("idx", 1), ("rng", -1, 2, 0), False, True)],
    "OBJECT ENCODING": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "OBJECT FREQ": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "OBJECT IDLETIME": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "OBJECT REFCOUNT": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "PERSIST": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PEXPIRE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PEXPIREAT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PEXPIRETIME": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "PFADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "PFCOUNT": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "PFDEBUG": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "PFMERGE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "PSETEX": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "PTTL": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "RENAME": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), False, True)],
    "RENAMENX": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), False, True)],
    "RESTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "RESTORE-ASKING": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "RPOP": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "RPOPLPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "RPUSH": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "RPUSHX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SCARD": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SDIFF": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "SDIFFSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "SET": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SETBIT": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SETEX": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "SETNX": [(("idx", 1), ("rng", 0, 1, 0), False, True)],
    "SETRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SINTER": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "SINTERCARD": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "SINTERSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "SISMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SMEMBERS": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SMISMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SMOVE": [(("idx", 1), ("rng", 0, 1, 0), True, True), (("idx", 2), ("rng", 0, 1, 0), True, True)],
    "SORT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SORT_RO": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SPOP": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SRANDMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SREM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "SSCAN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "STRLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SUBSTR": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "SUNION": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "SUNIONSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", -1, 1, 0), True, False)],
    "TOUCH": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "TTL": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "TYPE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "UNLINK": [(("idx", 1), ("rng", -1, 1, 0), False, True)],
    "WATCH": [(("idx", 1), ("rng", -1, 1, 0), True, False)],
    "XACK": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XAUTOCLAIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XCLAIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XDEL": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XGROUP CREATE": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP CREATECONSUMER": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP DELCONSUMER": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP DESTROY": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XGROUP SETID": [(("idx", 2), ("rng", 0, 1, 0), True, True)],
    "XINFO CONSUMERS": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "XINFO GROUPS": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "XINFO STREAM": [(("idx", 2), ("rng", 0, 1, 0), True, False)],
    "XLEN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XPENDING": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XREAD": [(("kw", "STREAMS", 1), ("rng", -1, 1, 2), True, False)],
    "XREADGROUP": [(("kw", "STREAMS", 4), ("rng", -1, 1, 2), True, True)],
    "XREVRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "XSETID": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "XTRIM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZADD": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZCARD": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZCOUNT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZDIFF": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZDIFFSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("kn", 0, 1, 1), True, False)],
    "ZINCRBY": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZINTER": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZINTERCARD": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZINTERSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("kn", 0, 1, 1), True, False)],
    "ZLEXCOUNT": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZMPOP": [(("idx", 1), ("kn", 0, 1, 1), True, True)],
    "ZMSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZPOPMAX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZPOPMIN": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZRANDMEMBER": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGEBYLEX": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGEBYSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZRANGESTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("rng", 0, 1, 0), True, False)],
    "ZRANK": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREM": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREMRANGEBYLEX": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREMRANGEBYRANK": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREMRANGEBYSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, True)],
    "ZREVRANGE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREVRANGEBYLEX": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREVRANGEBYSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZREVRANK": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZSCAN": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZSCORE": [(("idx", 1), ("rng", 0, 1, 0), True, False)],
    "ZUNION": [(("idx", 1), ("kn", 0, 1, 1), True, False)],
    "ZUNIONSTORE": [(("idx", 1), ("rng", 0, 1, 0), False, True), (("idx", 2), ("kn", 0, 1, 1), True, False)],
}
# fmt: on

# ---------------------------------------------------------------------------
# Command classification sets
# ---------------------------------------------------------------------------

# Transaction control commands — no keys, flag only.
_TX_CONTROL_CMDS: frozenset[str] = frozenset({"MULTI", "EXEC", "DISCARD", "UNWATCH"})

# EVAL/EVALSHA are atomic (Lua scripts execute without interleaving).
# Treat as transaction control with no key-level accesses.  See defect #8.
_EVAL_CMDS: frozenset[str] = frozenset({"EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO"})

# Commands that write the *entire* keyspace of a database.  They take no
# specific key, so the key-spec machinery yields nothing, but they conflict
# with every key access.  Modeled as a write on the database-wide keyspace
# resource (intent-lock pattern) so DPOR explores the racing order against
# concurrent key traffic.  See the keyspace intent-lock note in
# ``_redis_parsing.parse_redis_access``.
_KEYSPACE_WRITE_CMDS: frozenset[str] = frozenset({"FLUSHDB", "FLUSHALL", "SWAPDB"})

# Commands that read the *entire* keyspace (enumeration / existence over all
# keys) without naming a specific key.  Modeled as a read on the database-wide
# keyspace resource: this conflicts with FLUSH* (and nothing else new, since
# the keyspace read is read-read against ordinary key traffic).
_KEYSPACE_READ_CMDS: frozenset[str] = frozenset({"KEYS", "SCAN", "RANDOMKEY", "DBSIZE"})

# Server/connection/cluster commands that never operate on specific keys.
# These must not hit the conservative fallback (which treats arg[0] as a key).
#
# NOTE: FLUSHDB/FLUSHALL and KEYS/SCAN/RANDOMKEY/DBSIZE remain listed here
# because they still take no *specific* key; their database-wide keyspace
# access is layered on separately via the _KEYSPACE_* sets above.
_NO_KEY_CMDS: frozenset[str] = frozenset(
    {
        "ACL",
        "ASKING",
        "AUTH",
        "BGREWRITEAOF",
        "BGSAVE",
        "CLIENT",
        "CLUSTER",
        "COMMAND",
        "CONFIG",
        "DBSIZE",
        "DEBUG",
        "ECHO",
        "FAILOVER",
        "FLUSHALL",
        "FLUSHDB",
        "FUNCTION",
        "HELLO",
        "INFO",
        "KEYS",
        "LASTSAVE",
        "LATENCY",
        "LOLWUT",
        "MODULE",
        "MONITOR",
        "PFSELFTEST",
        "PING",
        "PSYNC",
        "PUBSUB",
        "QUIT",
        "RANDOMKEY",
        "READONLY",
        "READWRITE",
        "REPLCONF",
        "REPLICAOF",
        "RESET",
        "ROLE",
        "SAVE",
        "SCAN",
        "SCRIPT",
        "SELECT",
        "SHUTDOWN",
        "SLAVEOF",
        "SLOWLOG",
        "SWAPDB",
        "SYNC",
        "TIME",
        "WAIT",
        "WAITAOF",
    }
)

# Parent commands that dispatch to subcommands in the key-spec table.
# When a specific subcommand (e.g. "OBJECT HELP") is not found, we return
# no-keys instead of hitting the conservative fallback.
_SUBCOMMAND_PARENTS: frozenset[str] = frozenset(
    {parent for key in _COMMAND_KEY_SPECS if " " in key for parent in [key.split(" ", 1)[0]]}
)
