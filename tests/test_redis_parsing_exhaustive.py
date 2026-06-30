"""Exhaustive tests for Redis command key extraction.

Tests every Redis command that has key_specs in the official commands.json,
verifying that parse_redis_access correctly extracts keys and classifies
them as reads/writes.  Test data was generated from the Redis documentation
(https://github.com/redis/redis-doc) commands.json.

This covers all 193 commands with key specifications, including:
- String, hash, list, set, sorted-set, stream, geo, bitmap, hyperloglog
- Multi-key, keynum, keyword-based, and subcommand dispatch patterns
- DPOR-specific overrides (transaction control, pub/sub channels, Lua atomicity)
"""

from __future__ import annotations

import pytest

from frontrun._redis_parsing import parse_redis_access

# ---------------------------------------------------------------------------
# Exhaustive parametrized tests for every Redis command with key specs
#
# Format: (label, cmd_for_parse, cmd_args, expected_read_keys,
#          expected_write_keys, expected_is_tx_control)
# ---------------------------------------------------------------------------

_EXHAUSTIVE_CASES: list[tuple[str, str, tuple[object, ...], list[str], list[str], bool]] = [
    # ── String commands ──────────────────────────────────────────────────
    ("APPEND", "APPEND", ("mykey", "extra"), ["mykey"], ["mykey"], False),
    ("DECR", "DECR", ("counter",), ["counter"], ["counter"], False),
    ("DECRBY", "DECRBY", ("counter", "5"), ["counter"], ["counter"], False),
    ("GET", "GET", ("mykey",), ["mykey"], [], False),
    ("GETDEL", "GETDEL", ("mykey",), ["mykey"], ["mykey"], False),
    ("GETEX", "GETEX", ("mykey", "EX", "100"), ["mykey"], ["mykey"], False),
    ("GETRANGE", "GETRANGE", ("mykey", "0", "10"), ["mykey"], [], False),
    ("GETSET", "GETSET", ("mykey", "newval"), ["mykey"], ["mykey"], False),
    ("INCR", "INCR", ("counter",), ["counter"], ["counter"], False),
    ("INCRBY", "INCRBY", ("counter", "5"), ["counter"], ["counter"], False),
    ("INCRBYFLOAT", "INCRBYFLOAT", ("counter", "1.5"), ["counter"], ["counter"], False),
    ("LCS", "LCS", ("key1", "key2"), ["key1", "key2"], [], False),
    ("MGET", "MGET", ("k1", "k2", "k3"), ["k1", "k2", "k3"], [], False),
    ("MSET", "MSET", ("k1", "v1", "k2", "v2"), [], ["k1", "k2"], False),
    ("MSETNX", "MSETNX", ("k1", "v1", "k2", "v2"), [], ["k1", "k2"], False),
    ("PSETEX", "PSETEX", ("mykey", "10000", "val"), [], ["mykey"], False),
    ("SET", "SET", ("mykey", "myvalue"), ["mykey"], ["mykey"], False),
    ("SETEX", "SETEX", ("mykey", "60", "val"), [], ["mykey"], False),
    ("SETNX", "SETNX", ("mykey", "val"), [], ["mykey"], False),
    ("SETRANGE", "SETRANGE", ("mykey", "0", "val"), ["mykey"], ["mykey"], False),
    ("STRLEN", "STRLEN", ("mykey",), ["mykey"], [], False),
    ("SUBSTR", "SUBSTR", ("mykey", "0", "10"), ["mykey"], [], False),
    # ── Hash commands ────────────────────────────────────────────────────
    ("HDEL", "HDEL", ("myhash", "f1"), ["myhash"], ["myhash"], False),
    ("HEXISTS", "HEXISTS", ("myhash", "f1"), ["myhash"], [], False),
    ("HGET", "HGET", ("myhash", "f1"), ["myhash"], [], False),
    ("HGETALL", "HGETALL", ("myhash",), ["myhash"], [], False),
    ("HINCRBY", "HINCRBY", ("myhash", "f1", "5"), ["myhash"], ["myhash"], False),
    ("HINCRBYFLOAT", "HINCRBYFLOAT", ("myhash", "f1", "1.5"), ["myhash"], ["myhash"], False),
    ("HKEYS", "HKEYS", ("myhash",), ["myhash"], [], False),
    ("HLEN", "HLEN", ("myhash",), ["myhash"], [], False),
    ("HMGET", "HMGET", ("myhash", "f1", "f2"), ["myhash"], [], False),
    ("HMSET", "HMSET", ("myhash", "f1", "v1"), ["myhash"], ["myhash"], False),
    ("HRANDFIELD", "HRANDFIELD", ("myhash",), ["myhash"], [], False),
    ("HSCAN", "HSCAN", ("myhash", "0"), ["myhash"], [], False),
    ("HSET", "HSET", ("myhash", "f1", "v1"), ["myhash"], ["myhash"], False),
    ("HSETNX", "HSETNX", ("myhash", "f1", "v1"), ["myhash"], ["myhash"], False),
    ("HSTRLEN", "HSTRLEN", ("myhash", "f1"), ["myhash"], [], False),
    ("HVALS", "HVALS", ("myhash",), ["myhash"], [], False),
    # ── List commands ────────────────────────────────────────────────────
    ("BLMOVE", "BLMOVE", ("src", "dst", "LEFT", "RIGHT", "0"), ["src", "dst"], ["src", "dst"], False),
    ("BLMPOP", "BLMPOP", ("0", "2", "key1", "key2", "LEFT"), ["key1", "key2"], ["key1", "key2"], False),
    ("BLPOP", "BLPOP", ("key1", "key2", "0"), ["key1", "key2"], ["key1", "key2"], False),
    ("BRPOP", "BRPOP", ("key1", "key2", "0"), ["key1", "key2"], ["key1", "key2"], False),
    ("BRPOPLPUSH", "BRPOPLPUSH", ("src", "dst", "0"), ["src", "dst"], ["src", "dst"], False),
    ("LINDEX", "LINDEX", ("mylist", "0"), ["mylist"], [], False),
    ("LINSERT", "LINSERT", ("mylist", "BEFORE", "pivot", "val"), ["mylist"], ["mylist"], False),
    ("LLEN", "LLEN", ("mylist",), ["mylist"], [], False),
    ("LMOVE", "LMOVE", ("src", "dst", "LEFT", "RIGHT"), ["src", "dst"], ["src", "dst"], False),
    ("LMPOP", "LMPOP", ("2", "key1", "key2", "LEFT"), ["key1", "key2"], ["key1", "key2"], False),
    ("LPOP", "LPOP", ("mylist",), ["mylist"], ["mylist"], False),
    ("LPOS", "LPOS", ("mylist", "val"), ["mylist"], [], False),
    ("LPUSH", "LPUSH", ("mylist", "val"), ["mylist"], ["mylist"], False),
    ("LPUSHX", "LPUSHX", ("mylist", "val"), ["mylist"], ["mylist"], False),
    ("LRANGE", "LRANGE", ("mylist", "0", "-1"), ["mylist"], [], False),
    ("LREM", "LREM", ("mylist", "0", "val"), ["mylist"], ["mylist"], False),
    ("LSET", "LSET", ("mylist", "0", "val"), ["mylist"], ["mylist"], False),
    ("LTRIM", "LTRIM", ("mylist", "0", "10"), ["mylist"], ["mylist"], False),
    ("RPOP", "RPOP", ("mylist",), ["mylist"], ["mylist"], False),
    ("RPOPLPUSH", "RPOPLPUSH", ("src", "dst"), ["src", "dst"], ["src", "dst"], False),
    ("RPUSH", "RPUSH", ("mylist", "val"), ["mylist"], ["mylist"], False),
    ("RPUSHX", "RPUSHX", ("mylist", "val"), ["mylist"], ["mylist"], False),
    # ── Set commands ─────────────────────────────────────────────────────
    ("SADD", "SADD", ("myset", "m1"), ["myset"], ["myset"], False),
    ("SCARD", "SCARD", ("myset",), ["myset"], [], False),
    ("SDIFF", "SDIFF", ("s1", "s2", "s3"), ["s1", "s2", "s3"], [], False),
    ("SDIFFSTORE", "SDIFFSTORE", ("dst", "s1", "s2"), ["s1", "s2"], ["dst"], False),
    ("SINTER", "SINTER", ("s1", "s2", "s3"), ["s1", "s2", "s3"], [], False),
    ("SINTERCARD", "SINTERCARD", ("2", "k1", "k2"), ["k1", "k2"], [], False),
    ("SINTERSTORE", "SINTERSTORE", ("dst", "s1", "s2"), ["s1", "s2"], ["dst"], False),
    ("SISMEMBER", "SISMEMBER", ("myset", "m1"), ["myset"], [], False),
    ("SMEMBERS", "SMEMBERS", ("myset",), ["myset"], [], False),
    ("SMISMEMBER", "SMISMEMBER", ("myset", "m1", "m2"), ["myset"], [], False),
    ("SMOVE", "SMOVE", ("src", "dst", "member"), ["src", "dst"], ["src", "dst"], False),
    ("SPOP", "SPOP", ("myset",), ["myset"], ["myset"], False),
    ("SRANDMEMBER", "SRANDMEMBER", ("myset",), ["myset"], [], False),
    ("SREM", "SREM", ("myset", "m1"), ["myset"], ["myset"], False),
    ("SSCAN", "SSCAN", ("myset", "0"), ["myset"], [], False),
    ("SUNION", "SUNION", ("s1", "s2", "s3"), ["s1", "s2", "s3"], [], False),
    ("SUNIONSTORE", "SUNIONSTORE", ("dst", "s1", "s2"), ["s1", "s2"], ["dst"], False),
    # ── Sorted set commands ──────────────────────────────────────────────
    ("BZMPOP", "BZMPOP", ("0", "2", "zk1", "zk2", "MIN"), ["zk1", "zk2"], ["zk1", "zk2"], False),
    ("BZPOPMAX", "BZPOPMAX", ("zk1", "zk2", "0"), ["zk1", "zk2"], ["zk1", "zk2"], False),
    ("BZPOPMIN", "BZPOPMIN", ("zk1", "zk2", "0"), ["zk1", "zk2"], ["zk1", "zk2"], False),
    ("ZADD", "ZADD", ("myzset", "1", "m1"), ["myzset"], ["myzset"], False),
    ("ZCARD", "ZCARD", ("myzset",), ["myzset"], [], False),
    ("ZCOUNT", "ZCOUNT", ("myzset", "-inf", "+inf"), ["myzset"], [], False),
    ("ZDIFF", "ZDIFF", ("2", "zs1", "zs2"), ["zs1", "zs2"], [], False),
    ("ZDIFFSTORE", "ZDIFFSTORE", ("dst", "2", "zs1", "zs2"), ["zs1", "zs2"], ["dst"], False),
    ("ZINCRBY", "ZINCRBY", ("myzset", "1", "m1"), ["myzset"], ["myzset"], False),
    ("ZINTER", "ZINTER", ("2", "zs1", "zs2"), ["zs1", "zs2"], [], False),
    ("ZINTERCARD", "ZINTERCARD", ("2", "zk1", "zk2"), ["zk1", "zk2"], [], False),
    ("ZINTERSTORE", "ZINTERSTORE", ("dst", "2", "zs1", "zs2"), ["zs1", "zs2"], ["dst"], False),
    ("ZLEXCOUNT", "ZLEXCOUNT", ("myzset", "-", "+"), ["myzset"], [], False),
    ("ZMPOP", "ZMPOP", ("2", "zk1", "zk2", "MIN"), ["zk1", "zk2"], ["zk1", "zk2"], False),
    ("ZMSCORE", "ZMSCORE", ("myzset", "m1"), ["myzset"], [], False),
    ("ZPOPMAX", "ZPOPMAX", ("myzset",), ["myzset"], ["myzset"], False),
    ("ZPOPMIN", "ZPOPMIN", ("myzset",), ["myzset"], ["myzset"], False),
    ("ZRANDMEMBER", "ZRANDMEMBER", ("myzset",), ["myzset"], [], False),
    ("ZRANGE", "ZRANGE", ("myzset", "0", "-1"), ["myzset"], [], False),
    ("ZRANGEBYLEX", "ZRANGEBYLEX", ("myzset", "-", "+"), ["myzset"], [], False),
    ("ZRANGEBYSCORE", "ZRANGEBYSCORE", ("myzset", "-inf", "+inf"), ["myzset"], [], False),
    ("ZRANGESTORE", "ZRANGESTORE", ("dst", "src", "0", "-1"), ["src"], ["dst"], False),
    ("ZRANK", "ZRANK", ("myzset", "m1"), ["myzset"], [], False),
    ("ZREM", "ZREM", ("myzset", "m1"), ["myzset"], ["myzset"], False),
    ("ZREMRANGEBYLEX", "ZREMRANGEBYLEX", ("myzset", "-", "+"), ["myzset"], ["myzset"], False),
    ("ZREMRANGEBYRANK", "ZREMRANGEBYRANK", ("myzset", "0", "1"), ["myzset"], ["myzset"], False),
    ("ZREMRANGEBYSCORE", "ZREMRANGEBYSCORE", ("myzset", "-inf", "+inf"), ["myzset"], ["myzset"], False),
    ("ZREVRANGE", "ZREVRANGE", ("myzset", "0", "-1"), ["myzset"], [], False),
    ("ZREVRANGEBYLEX", "ZREVRANGEBYLEX", ("myzset", "+", "-"), ["myzset"], [], False),
    ("ZREVRANGEBYSCORE", "ZREVRANGEBYSCORE", ("myzset", "+inf", "-inf"), ["myzset"], [], False),
    ("ZREVRANK", "ZREVRANK", ("myzset", "m1"), ["myzset"], [], False),
    ("ZSCAN", "ZSCAN", ("myzset", "0"), ["myzset"], [], False),
    ("ZSCORE", "ZSCORE", ("myzset", "m1"), ["myzset"], [], False),
    ("ZUNION", "ZUNION", ("2", "zs1", "zs2"), ["zs1", "zs2"], [], False),
    ("ZUNIONSTORE", "ZUNIONSTORE", ("dst", "2", "zs1", "zs2"), ["zs1", "zs2"], ["dst"], False),
    # ── Generic/key commands ─────────────────────────────────────────────
    ("COPY", "COPY", ("src", "dst"), ["src"], ["dst"], False),
    ("DEL", "DEL", ("k1", "k2", "k3"), [], ["k1", "k2", "k3"], False),
    ("DUMP", "DUMP", ("mykey",), ["mykey"], [], False),
    ("EXISTS", "EXISTS", ("k1", "k2", "k3"), ["k1", "k2", "k3"], [], False),
    ("EXPIRE", "EXPIRE", ("mykey", "60"), ["mykey"], ["mykey"], False),
    ("EXPIREAT", "EXPIREAT", ("mykey", "1234567890"), ["mykey"], ["mykey"], False),
    ("EXPIRETIME", "EXPIRETIME", ("mykey",), ["mykey"], [], False),
    ("MIGRATE", "MIGRATE", ("host", "6379", "mykey", "0", "1000"), ["mykey"], ["mykey"], False),
    ("MOVE", "MOVE", ("mykey", "1"), ["mykey"], ["mykey"], False),
    ("PERSIST", "PERSIST", ("mykey",), ["mykey"], ["mykey"], False),
    ("PEXPIRE", "PEXPIRE", ("mykey", "60000"), ["mykey"], ["mykey"], False),
    ("PEXPIREAT", "PEXPIREAT", ("mykey", "1234567890000"), ["mykey"], ["mykey"], False),
    ("PEXPIRETIME", "PEXPIRETIME", ("mykey",), ["mykey"], [], False),
    ("PTTL", "PTTL", ("mykey",), ["mykey"], [], False),
    ("RENAME", "RENAME", ("old", "new"), ["old"], ["old", "new"], False),
    ("RENAMENX", "RENAMENX", ("old", "new"), ["old"], ["old", "new"], False),
    ("RESTORE", "RESTORE", ("mykey", "0", "data"), [], ["mykey"], False),
    ("RESTORE-ASKING", "RESTORE-ASKING", ("mykey", "0", "data"), [], ["mykey"], False),
    ("TOUCH", "TOUCH", ("k1", "k2", "k3"), ["k1", "k2", "k3"], [], False),
    ("TTL", "TTL", ("mykey",), ["mykey"], [], False),
    ("TYPE", "TYPE", ("mykey",), ["mykey"], [], False),
    ("UNLINK", "UNLINK", ("k1", "k2", "k3"), [], ["k1", "k2", "k3"], False),
    # ── Bitmap commands ──────────────────────────────────────────────────
    ("BITCOUNT", "BITCOUNT", ("mykey", "0", "10"), ["mykey"], [], False),
    ("BITFIELD", "BITFIELD", ("mykey", "SET", "u8", "0", "255"), ["mykey"], ["mykey"], False),
    ("BITFIELD_RO", "BITFIELD_RO", ("mykey", "GET", "u8", "0"), ["mykey"], [], False),
    ("BITOP", "BITOP", ("AND", "dst", "src1", "src2"), ["src1", "src2"], ["dst"], False),
    ("BITPOS", "BITPOS", ("mykey", "1"), ["mykey"], [], False),
    ("GETBIT", "GETBIT", ("mykey", "7"), ["mykey"], [], False),
    ("SETBIT", "SETBIT", ("mykey", "7", "1"), ["mykey"], ["mykey"], False),
    # ── HyperLogLog commands ─────────────────────────────────────────────
    ("PFADD", "PFADD", ("myhll", "elem1", "elem2"), ["myhll"], ["myhll"], False),
    ("PFCOUNT", "PFCOUNT", ("hll1", "hll2"), ["hll1", "hll2"], [], False),
    ("PFDEBUG", "PFDEBUG", ("GETREG", "mykey"), ["mykey"], ["mykey"], False),
    ("PFMERGE", "PFMERGE", ("dst", "src1", "src2"), ["dst", "src1", "src2"], ["dst"], False),
    # ── Geo commands ─────────────────────────────────────────────────────
    ("GEOADD", "GEOADD", ("mygeo", "13.36", "38.12", "Palermo"), ["mygeo"], ["mygeo"], False),
    ("GEODIST", "GEODIST", ("mygeo", "Palermo", "Catania"), ["mygeo"], [], False),
    ("GEOHASH", "GEOHASH", ("mygeo", "Palermo"), ["mygeo"], [], False),
    ("GEOPOS", "GEOPOS", ("mygeo", "Palermo"), ["mygeo"], [], False),
    ("GEORADIUS read-only", "GEORADIUS", ("mygeo", "15", "37", "100", "km"), ["mygeo"], [], False),
    (
        "GEORADIUS with STORE",
        "GEORADIUS",
        ("mygeo", "15", "37", "100", "km", "STORE", "dst"),
        ["mygeo"],
        ["dst"],
        False,
    ),
    (
        "GEORADIUS with STOREDIST",
        "GEORADIUS",
        ("mygeo", "15", "37", "100", "km", "STOREDIST", "dst"),
        ["mygeo"],
        ["dst"],
        False,
    ),
    ("GEORADIUSBYMEMBER read-only", "GEORADIUSBYMEMBER", ("mygeo", "Palermo", "100", "km"), ["mygeo"], [], False),
    (
        "GEORADIUSBYMEMBER with STORE",
        "GEORADIUSBYMEMBER",
        ("mygeo", "Palermo", "100", "km", "STORE", "dst"),
        ["mygeo"],
        ["dst"],
        False,
    ),
    ("GEORADIUSBYMEMBER_RO", "GEORADIUSBYMEMBER_RO", ("mygeo", "Palermo", "100", "km"), ["mygeo"], [], False),
    ("GEORADIUS_RO", "GEORADIUS_RO", ("mygeo", "15", "37", "100", "km"), ["mygeo"], [], False),
    ("GEOSEARCH", "GEOSEARCH", ("mygeo", "FROMLONLAT", "15", "37", "BYRADIUS", "100", "km"), ["mygeo"], [], False),
    (
        "GEOSEARCHSTORE",
        "GEOSEARCHSTORE",
        ("dst", "src", "FROMLONLAT", "15", "37", "BYRADIUS", "100", "km"),
        ["src"],
        ["dst"],
        False,
    ),
    # ── Stream commands ──────────────────────────────────────────────────
    ("XACK", "XACK", ("mystream", "grp", "id1"), ["mystream"], ["mystream"], False),
    ("XADD", "XADD", ("mystream", "*", "field", "value"), ["mystream"], ["mystream"], False),
    ("XAUTOCLAIM", "XAUTOCLAIM", ("mystream", "grp", "consumer", "0", "0-0"), ["mystream"], ["mystream"], False),
    ("XCLAIM", "XCLAIM", ("mystream", "grp", "consumer", "0", "id1"), ["mystream"], ["mystream"], False),
    ("XDEL", "XDEL", ("mystream", "id1"), ["mystream"], ["mystream"], False),
    ("XGROUP CREATE", "XGROUP", ("CREATE", "mystream", "grp", "0"), ["mystream"], ["mystream"], False),
    (
        "XGROUP CREATECONSUMER",
        "XGROUP",
        ("CREATECONSUMER", "mystream", "grp", "consumer"),
        ["mystream"],
        ["mystream"],
        False,
    ),
    ("XGROUP DELCONSUMER", "XGROUP", ("DELCONSUMER", "mystream", "grp", "consumer"), ["mystream"], ["mystream"], False),
    ("XGROUP DESTROY", "XGROUP", ("DESTROY", "mystream", "grp"), ["mystream"], ["mystream"], False),
    ("XGROUP SETID", "XGROUP", ("SETID", "mystream", "grp", "0"), ["mystream"], ["mystream"], False),
    ("XINFO CONSUMERS", "XINFO", ("CONSUMERS", "mystream", "grp"), ["mystream"], [], False),
    ("XINFO GROUPS", "XINFO", ("GROUPS", "mystream"), ["mystream"], [], False),
    ("XINFO STREAM", "XINFO", ("STREAM", "mystream"), ["mystream"], [], False),
    ("XLEN", "XLEN", ("mystream",), ["mystream"], [], False),
    ("XPENDING", "XPENDING", ("mystream", "grp"), ["mystream"], [], False),
    ("XRANGE", "XRANGE", ("mystream", "-", "+"), ["mystream"], [], False),
    ("XREAD single stream", "XREAD", ("STREAMS", "s1", "0-0"), ["s1"], [], False),
    ("XREAD two streams", "XREAD", ("COUNT", "10", "STREAMS", "s1", "s2", "0-0", "0-0"), ["s1", "s2"], [], False),
    (
        "XREADGROUP",
        "XREADGROUP",
        ("GROUP", "g", "c", "STREAMS", "s1", "s2", ">", ">"),
        ["s1", "s2"],
        ["s1", "s2"],
        False,
    ),
    ("XREVRANGE", "XREVRANGE", ("mystream", "+", "-"), ["mystream"], [], False),
    ("XSETID", "XSETID", ("mystream", "1-1"), ["mystream"], ["mystream"], False),
    ("XTRIM", "XTRIM", ("mystream", "MAXLEN", "1000"), ["mystream"], ["mystream"], False),
    # ── OBJECT subcommands ───────────────────────────────────────────────
    ("OBJECT ENCODING", "OBJECT", ("ENCODING", "mykey"), ["mykey"], [], False),
    ("OBJECT FREQ", "OBJECT", ("FREQ", "mykey"), ["mykey"], [], False),
    ("OBJECT IDLETIME", "OBJECT", ("IDLETIME", "mykey"), ["mykey"], [], False),
    ("OBJECT REFCOUNT", "OBJECT", ("REFCOUNT", "mykey"), ["mykey"], [], False),
    ("OBJECT HELP no keys", "OBJECT", ("HELP",), [], [], False),
    # ── MEMORY subcommand ────────────────────────────────────────────────
    ("MEMORY USAGE", "MEMORY", ("USAGE", "mykey"), ["mykey"], [], False),
    # ── SORT special handling ────────────────────────────────────────────
    ("SORT read-only", "SORT", ("mylist", "ASC"), ["mylist"], [], False),
    ("SORT with STORE", "SORT", ("mylist", "STORE", "dst"), ["mylist"], ["dst"], False),
    ("SORT_RO", "SORT_RO", ("mylist", "ASC"), ["mylist"], [], False),
    # ── Transaction control (DPOR override) ──────────────────────────────
    ("MULTI", "MULTI", (), [], [], True),
    ("EXEC", "EXEC", (), [], [], True),
    ("DISCARD", "DISCARD", (), [], [], True),
    ("UNWATCH", "UNWATCH", (), [], [], True),
    ("WATCH", "WATCH", ("k1", "k2"), ["k1", "k2"], [], True),
    # ── Lua scripts (DPOR override — treated as atomic/tx control) ──────
    ("EVAL", "EVAL", ("script", "2", "key1", "key2", "arg1"), [], [], True),
    ("EVALSHA", "EVALSHA", ("sha1", "2", "key1", "key2"), [], [], True),
    ("EVAL_RO", "EVAL_RO", ("script", "1", "key1"), [], [], True),
    ("EVALSHA_RO", "EVALSHA_RO", ("sha1", "1", "key1"), [], [], True),
    ("FCALL", "FCALL", ("func", "2", "key1", "key2"), [], [], True),
    ("FCALL_RO", "FCALL_RO", ("func", "1", "key1"), [], [], True),
    # ── Pub/Sub (DPOR override — channel prefix) ────────────────────────
    ("PUBLISH", "PUBLISH", ("mychan", "msg"), [], ["channel:mychan"], False),
    ("SUBSCRIBE", "SUBSCRIBE", ("ch1", "ch2"), ["channel:ch1", "channel:ch2"], [], False),
    ("PSUBSCRIBE", "PSUBSCRIBE", ("pat*",), ["channel:pat*"], [], False),
    ("SPUBLISH", "SPUBLISH", ("mychan", "msg"), [], ["channel:mychan"], False),
    ("SSUBSCRIBE", "SSUBSCRIBE", ("ch1", "ch2"), ["channel:ch1", "channel:ch2"], [], False),
    ("UNSUBSCRIBE", "UNSUBSCRIBE", ("ch1",), [], [], False),
    ("PUNSUBSCRIBE", "PUNSUBSCRIBE", ("pat*",), [], [], False),
    ("SUNSUBSCRIBE", "SUNSUBSCRIBE", ("ch1",), [], [], False),
    # ── Server/no-key commands ───────────────────────────────────────────
    ("PING", "PING", (), [], [], False),
    ("PING with msg", "PING", ("hello",), [], [], False),
    ("INFO", "INFO", ("server",), [], [], False),
    ("CONFIG", "CONFIG", ("GET", "maxmemory"), [], [], False),
    ("DBSIZE", "DBSIZE", (), [], [], False),
    ("FLUSHDB", "FLUSHDB", (), [], [], False),
    ("FLUSHALL", "FLUSHALL", (), [], [], False),
    ("SELECT", "SELECT", ("0",), [], [], False),
    ("AUTH", "AUTH", ("password",), [], [], False),
    ("SCAN", "SCAN", ("0",), [], [], False),
    ("KEYS", "KEYS", ("*",), [], [], False),
    ("RANDOMKEY", "RANDOMKEY", (), [], [], False),
    ("COMMAND", "COMMAND", (), [], [], False),
    ("TIME", "TIME", (), [], [], False),
    ("CLUSTER", "CLUSTER", ("INFO",), [], [], False),
    ("CLIENT", "CLIENT", ("LIST",), [], [], False),
    ("SLOWLOG", "SLOWLOG", ("GET", "10"), [], [], False),
    ("DEBUG", "DEBUG", ("SLEEP", "0"), [], [], False),
    ("WAIT", "WAIT", ("1", "0"), [], [], False),
    ("WAITAOF", "WAITAOF", ("1", "1", "0"), [], [], False),
    ("ECHO", "ECHO", ("hello",), [], [], False),
    ("RESET", "RESET", (), [], [], False),
    ("QUIT", "QUIT", (), [], [], False),
    ("ACL", "ACL", ("LIST",), [], [], False),
    ("SCRIPT", "SCRIPT", ("EXISTS", "sha1"), [], [], False),
    ("FUNCTION", "FUNCTION", ("LIST",), [], [], False),
    ("LATENCY", "LATENCY", ("LATEST",), [], [], False),
    ("MODULE", "MODULE", ("LIST",), [], [], False),
    # ── Fallback behavior ────────────────────────────────────────────────
    ("Unknown cmd with args", "THISDOESNOTEXIST", ("arg1",), [], ["arg1"], False),
    ("Unknown cmd no args", "THISDOESNOTEXIST", (), [], [], False),
]


@pytest.mark.parametrize(
    "label,cmd,args,expected_read,expected_write,expected_tx",
    _EXHAUSTIVE_CASES,
    ids=[c[0] for c in _EXHAUSTIVE_CASES],
)
def test_exhaustive_command_classification(
    label: str,
    cmd: str,
    args: tuple[object, ...],
    expected_read: list[str],
    expected_write: list[str],
    expected_tx: bool,
) -> None:
    result = parse_redis_access(cmd, args)
    assert result.read_keys == expected_read, f"{label}: read_keys mismatch"
    assert result.write_keys == expected_write, f"{label}: write_keys mismatch"
    assert result.is_transaction_control == expected_tx, f"{label}: is_transaction_control mismatch"


class TestKeySpecInterpreterEdgeCases:
    """Edge cases for the generic key-spec interpreter."""

    def test_empty_args_returns_empty(self) -> None:
        # Commands that normally take keys but get no args.
        result = parse_redis_access("GET", ())
        assert result.read_keys == []
        assert result.write_keys == []

    def test_case_insensitive(self) -> None:
        result = parse_redis_access("get", ("mykey",))
        assert result.read_keys == ["mykey"]

    def test_xread_without_streams_keyword(self) -> None:
        # XREAD without STREAMS keyword → no keys found.
        result = parse_redis_access("XREAD", ("COUNT", "10"))
        assert result.read_keys == []

    def test_xread_three_streams(self) -> None:
        result = parse_redis_access("XREAD", ("STREAMS", "a", "b", "c", "0", "0", "0"))
        assert result.read_keys == ["a", "b", "c"]

    def test_mset_odd_args_handles_gracefully(self) -> None:
        # MSET with odd args (missing last value) — extracts keys at even positions.
        result = parse_redis_access("MSET", ("k1", "v1", "k2"))
        # keystep=2: k1 at 0, k2 at 2 → only k1 (since lastkey=-1 means all remaining,
        # but with keystep=2 we get indices 0, 2 out of 3 args → k1, k2)
        assert "k1" in result.write_keys

    def test_zunionstore_numkeys(self) -> None:
        result = parse_redis_access("ZUNIONSTORE", ("dst", "3", "z1", "z2", "z3", "WEIGHTS", "1", "2", "3"))
        assert result.write_keys == ["dst"]
        assert result.read_keys == ["z1", "z2", "z3"]

    def test_georadius_store_and_storedist(self) -> None:
        # Both STORE and STOREDIST — first one found wins.
        result = parse_redis_access(
            "GEORADIUS", ("mygeo", "15", "37", "100", "km", "STORE", "dst1", "STOREDIST", "dst2")
        )
        assert "mygeo" in result.read_keys
        assert "dst1" in result.write_keys
        assert "dst2" in result.write_keys

    def test_blmpop_single_key(self) -> None:
        result = parse_redis_access("BLMPOP", ("0", "1", "mylist", "LEFT"))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["mylist"]

    def test_migrate_single_key(self) -> None:
        # MIGRATE host port key db timeout — single key at index 3.
        result = parse_redis_access("MIGRATE", ("host", "6379", "mykey", "0", "1000"))
        assert "mykey" in result.read_keys
        assert "mykey" in result.write_keys

    def test_migrate_with_keys_keyword(self) -> None:
        # MIGRATE with KEYS keyword: needs enough trailing args for the -2 startfrom
        # to find KEYS. The keyword search starts from n_args-2, so KEYS must be
        # at that position or later.
        # MIGRATE host port "" db timeout COPY REPLACE AUTH pass KEYS k1
        result = parse_redis_access(
            "MIGRATE", ("host", "6379", "", "0", "1000", "COPY", "REPLACE", "AUTH", "pass", "KEYS", "k1")
        )
        assert "k1" in result.read_keys
        assert "k1" in result.write_keys

    def test_sort_by_pattern_named_store(self) -> None:
        """SORT mylist BY store — 'store' is a BY pattern, not the STORE keyword."""
        result = parse_redis_access("SORT", ("mylist", "BY", "store"))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_sort_limit_then_store(self) -> None:
        result = parse_redis_access("SORT", ("mylist", "LIMIT", "0", "10", "STORE", "dst"))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["dst"]

    def test_xgroup_unknown_subcommand_no_keys(self) -> None:
        """Unknown XGROUP subcommand should return no keys (not hit fallback)."""
        result = parse_redis_access("XGROUP", ("HELP",))
        assert result.read_keys == []
        assert result.write_keys == []

    def test_xinfo_unknown_subcommand_no_keys(self) -> None:
        result = parse_redis_access("XINFO", ("HELP",))
        assert result.read_keys == []
        assert result.write_keys == []

    def test_memory_unknown_subcommand_no_keys(self) -> None:
        result = parse_redis_access("MEMORY", ("DOCTOR",))
        assert result.read_keys == []
        assert result.write_keys == []


def test_command_key_specs_includes_fcall() -> None:
    """_COMMAND_KEY_SPECS is the complete catalog — FCALL/FCALL_RO must be present even though _EVAL_CMDS shadows them."""
    from frontrun._redis_command_data import _COMMAND_KEY_SPECS

    assert "FCALL" in _COMMAND_KEY_SPECS
    assert "FCALL_RO" in _COMMAND_KEY_SPECS
