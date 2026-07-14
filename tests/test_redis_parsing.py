"""Tests for Redis command classification and key extraction."""

from __future__ import annotations

import pytest

from frontrun._redis_parsing import parse_redis_access


class TestStringCommands:
    """Test classification of Redis string commands."""

    def test_get_is_read(self) -> None:
        result = parse_redis_access("GET", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_set_is_read_write(self) -> None:
        result = parse_redis_access("SET", ("mykey", "myvalue"))
        assert "mykey" in result.read_keys
        assert "mykey" in result.write_keys

    def test_setnx_is_write(self) -> None:
        result = parse_redis_access("SETNX", ("mykey", "myvalue"))
        assert result.write_keys == ["mykey"]

    def test_setex_is_write(self) -> None:
        result = parse_redis_access("SETEX", ("mykey", 60, "myvalue"))
        assert result.write_keys == ["mykey"]

    def test_incr_is_read_write(self) -> None:
        result = parse_redis_access("INCR", ("counter",))
        assert "counter" in result.write_keys
        assert "counter" in result.read_keys

    def test_incrby_is_read_write(self) -> None:
        result = parse_redis_access("INCRBY", ("counter", 5))
        assert "counter" in result.write_keys
        assert "counter" in result.read_keys

    def test_decr_is_read_write(self) -> None:
        result = parse_redis_access("DECR", ("counter",))
        assert "counter" in result.write_keys
        assert "counter" in result.read_keys

    def test_append_is_read_write(self) -> None:
        result = parse_redis_access("APPEND", ("mykey", "extra"))
        assert "mykey" in result.write_keys
        assert "mykey" in result.read_keys

    def test_strlen_is_read(self) -> None:
        result = parse_redis_access("STRLEN", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_mget_reads_all_keys(self) -> None:
        result = parse_redis_access("MGET", ("key1", "key2", "key3"))
        assert result.read_keys == ["key1", "key2", "key3"]
        assert result.write_keys == []

    def test_mset_writes_key_value_pairs(self) -> None:
        result = parse_redis_access("MSET", ("key1", "val1", "key2", "val2"))
        assert result.read_keys == []
        assert result.write_keys == ["key1", "key2"]

    def test_getset_is_read_write(self) -> None:
        result = parse_redis_access("GETSET", ("mykey", "newvalue"))
        assert "mykey" in result.read_keys
        assert "mykey" in result.write_keys

    def test_getdel_is_read_write(self) -> None:
        result = parse_redis_access("GETDEL", ("mykey",))
        assert "mykey" in result.read_keys
        assert "mykey" in result.write_keys


class TestBytesKeys:
    """Equivalent ``bytes`` and ``str`` keys share a resource ID."""

    def test_get_bytes_key(self) -> None:
        result = parse_redis_access("GET", (b"mykey",))
        assert result.read_keys == ["mykey"]

    def test_set_bytes_key(self) -> None:
        result = parse_redis_access("SET", (b"counter", b"1"))
        assert "counter" in result.read_keys
        assert "counter" in result.write_keys

    def test_bytes_and_str_same_key_collapse(self) -> None:
        bytes_keys = parse_redis_access("INCR", (b"counter",)).write_keys
        str_keys = parse_redis_access("INCR", ("counter",)).write_keys
        assert bytes_keys == str_keys == ["counter"]

    def test_mset_bytes_keys(self) -> None:
        result = parse_redis_access("MSET", (b"k1", b"v1", b"k2", b"v2"))
        assert result.write_keys == ["k1", "k2"]

    def test_publish_bytes_channel(self) -> None:
        result = parse_redis_access("PUBLISH", (b"chan", b"msg"))
        assert result.write_keys == ["channel:chan"]


class TestHashCommands:
    """Test classification of Redis hash commands."""

    def test_hget_is_read(self) -> None:
        result = parse_redis_access("HGET", ("myhash", "field1"))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []

    def test_hset_is_read_write(self) -> None:
        result = parse_redis_access("HSET", ("myhash", "field1", "value1"))
        assert "myhash" in result.read_keys
        assert "myhash" in result.write_keys

    def test_hmset_is_read_write(self) -> None:
        result = parse_redis_access("HMSET", ("myhash", "f1", "v1", "f2", "v2"))
        assert "myhash" in result.read_keys
        assert "myhash" in result.write_keys

    def test_hgetall_is_read(self) -> None:
        result = parse_redis_access("HGETALL", ("myhash",))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []

    def test_hdel_is_read_write(self) -> None:
        result = parse_redis_access("HDEL", ("myhash", "field1"))
        assert "myhash" in result.write_keys
        assert "myhash" in result.read_keys

    def test_hincrby_is_read_write(self) -> None:
        result = parse_redis_access("HINCRBY", ("myhash", "field1", 5))
        assert "myhash" in result.write_keys
        assert "myhash" in result.read_keys

    def test_hexists_is_read(self) -> None:
        result = parse_redis_access("HEXISTS", ("myhash", "field1"))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []

    def test_hlen_is_read(self) -> None:
        result = parse_redis_access("HLEN", ("myhash",))
        assert result.read_keys == ["myhash"]
        assert result.write_keys == []


class TestListCommands:
    """Test classification of Redis list commands."""

    def test_lpush_is_write(self) -> None:
        result = parse_redis_access("LPUSH", ("mylist", "value1"))
        assert "mylist" in result.write_keys

    def test_rpush_is_write(self) -> None:
        result = parse_redis_access("RPUSH", ("mylist", "value1"))
        assert "mylist" in result.write_keys

    def test_lpop_is_write(self) -> None:
        result = parse_redis_access("LPOP", ("mylist",))
        assert "mylist" in result.write_keys

    def test_rpop_is_write(self) -> None:
        result = parse_redis_access("RPOP", ("mylist",))
        assert "mylist" in result.write_keys

    def test_lrange_is_read(self) -> None:
        result = parse_redis_access("LRANGE", ("mylist", 0, -1))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_llen_is_read(self) -> None:
        result = parse_redis_access("LLEN", ("mylist",))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_lindex_is_read(self) -> None:
        result = parse_redis_access("LINDEX", ("mylist", 0))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_lmpop_uses_numkeys(self) -> None:
        """LMPOP numkeys key [key ...] LEFT|RIGHT — first arg is numkeys, not a key."""
        result = parse_redis_access("LMPOP", (2, "list1", "list2", "LEFT"))
        assert "list1" in result.write_keys
        assert "list2" in result.write_keys
        assert "list1" in result.read_keys
        assert "list2" in result.read_keys

    def test_lmpop_single_key(self) -> None:
        result = parse_redis_access("LMPOP", (1, "mylist", "LEFT"))
        assert "mylist" in result.write_keys
        assert "mylist" in result.read_keys

    def test_blpop_is_read_write(self) -> None:
        """BLPOP pops (removes) and returns the value — both read and write."""
        result = parse_redis_access("BLPOP", ("mylist", 0))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["mylist"]

    def test_brpop_is_read_write(self) -> None:
        """BRPOP pops (removes) and returns the value — both read and write."""
        result = parse_redis_access("BRPOP", ("mylist", 0))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["mylist"]

    def test_blpop_multiple_keys(self) -> None:
        """BLPOP key [key ...] timeout — all keys are read+write."""
        result = parse_redis_access("BLPOP", ("list1", "list2", 0))
        assert result.read_keys == ["list1", "list2"]
        assert result.write_keys == ["list1", "list2"]

    def test_brpop_multiple_keys(self) -> None:
        """BRPOP key [key ...] timeout — all keys are read+write."""
        result = parse_redis_access("BRPOP", ("list1", "list2", 0))
        assert result.read_keys == ["list1", "list2"]
        assert result.write_keys == ["list1", "list2"]


class TestSetCommands:
    """Test classification of Redis set commands."""

    def test_sadd_is_write(self) -> None:
        result = parse_redis_access("SADD", ("myset", "member1"))
        assert "myset" in result.write_keys

    def test_srem_is_write(self) -> None:
        result = parse_redis_access("SREM", ("myset", "member1"))
        assert "myset" in result.write_keys

    def test_smembers_is_read(self) -> None:
        result = parse_redis_access("SMEMBERS", ("myset",))
        assert result.read_keys == ["myset"]
        assert result.write_keys == []

    def test_scard_is_read(self) -> None:
        result = parse_redis_access("SCARD", ("myset",))
        assert result.read_keys == ["myset"]
        assert result.write_keys == []

    def test_sismember_is_read(self) -> None:
        result = parse_redis_access("SISMEMBER", ("myset", "member1"))
        assert result.read_keys == ["myset"]
        assert result.write_keys == []

    def test_sunionstore_reads_sources_writes_dest(self) -> None:
        result = parse_redis_access("SUNIONSTORE", ("dest", "src1", "src2"))
        assert result.read_keys == ["src1", "src2"]
        assert result.write_keys == ["dest"]

    def test_sunion_reads_all(self) -> None:
        result = parse_redis_access("SUNION", ("set1", "set2", "set3"))
        assert result.read_keys == ["set1", "set2", "set3"]
        assert result.write_keys == []

    def test_sinterstore_dest_is_write_only(self) -> None:
        """SINTERSTORE overwrites dest without reading it — dest should be write-only.

        Analogous to SDIFFSTORE, SUNIONSTORE, etc. which all mark dest as
        write-only (is_read=False, is_write=True).
        """
        result = parse_redis_access("SINTERSTORE", ("dest", "src1", "src2"))
        assert result.write_keys == ["dest"], "dest should be a write key"
        assert "dest" not in result.read_keys, "dest should NOT be a read key (write-only like SDIFFSTORE)"
        assert result.read_keys == ["src1", "src2"], "sources should be read keys"

    def test_sdiffstore_dest_is_write_only(self) -> None:
        """Sanity check: SDIFFSTORE dest is write-only (reference for SINTERSTORE fix)."""
        result = parse_redis_access("SDIFFSTORE", ("dest", "src1", "src2"))
        assert "dest" not in result.read_keys
        assert result.write_keys == ["dest"]
        assert result.read_keys == ["src1", "src2"]


class TestMigrateKeys:
    """Test MIGRATE command key extraction, especially the KEYS clause."""

    def test_migrate_single_key_positional(self) -> None:
        """MIGRATE host port key db timeout — single positional key."""
        result = parse_redis_access("MIGRATE", ("host", "6379", "mykey", "0", "1000"))
        assert "mykey" in result.read_keys or "mykey" in result.write_keys

    def test_migrate_keys_clause_single(self) -> None:
        """MIGRATE host port '' db timeout KEYS key1 — single key via KEYS clause."""
        result = parse_redis_access("MIGRATE", ("host", "6379", "", "0", "1000", "KEYS", "key1"))
        assert "key1" in result.read_keys or "key1" in result.write_keys

    def test_migrate_keys_clause_multiple(self) -> None:
        """MIGRATE host port '' db timeout KEYS key1 key2 — multiple keys via KEYS clause.

        Regression: the kw startfrom=-2 searches from n_args-2, which misses
        the KEYS token when there are 2+ keys after it.
        """
        result = parse_redis_access("MIGRATE", ("host", "6379", "", "0", "1000", "KEYS", "key1", "key2"))
        keys = result.read_keys + result.write_keys
        assert "key1" in keys, f"key1 should be extracted from KEYS clause, got {keys}"
        assert "key2" in keys, f"key2 should be extracted from KEYS clause, got {keys}"

    def test_migrate_keys_clause_with_copy_replace(self) -> None:
        """MIGRATE host port '' db timeout COPY REPLACE KEYS key1 key2 key3."""
        result = parse_redis_access(
            "MIGRATE", ("host", "6379", "", "0", "1000", "COPY", "REPLACE", "KEYS", "key1", "key2", "key3")
        )
        keys = result.read_keys + result.write_keys
        assert "key1" in keys, f"key1 missing from {keys}"
        assert "key2" in keys, f"key2 missing from {keys}"
        assert "key3" in keys, f"key3 missing from {keys}"


class TestSortedSetCommands:
    """Test classification of Redis sorted set commands."""

    def test_zadd_is_write(self) -> None:
        result = parse_redis_access("ZADD", ("myzset", 1, "member1"))
        assert "myzset" in result.write_keys

    def test_zrange_is_read(self) -> None:
        result = parse_redis_access("ZRANGE", ("myzset", 0, -1))
        assert result.read_keys == ["myzset"]
        assert result.write_keys == []

    def test_zcard_is_read(self) -> None:
        result = parse_redis_access("ZCARD", ("myzset",))
        assert result.read_keys == ["myzset"]
        assert result.write_keys == []

    def test_zincrby_is_write(self) -> None:
        result = parse_redis_access("ZINCRBY", ("myzset", 1, "member1"))
        assert "myzset" in result.write_keys

    def test_zmpop_uses_numkeys(self) -> None:
        """ZMPOP numkeys key [key ...] MIN|MAX — first arg is numkeys."""
        result = parse_redis_access("ZMPOP", (2, "zset1", "zset2", "MIN"))
        assert "zset1" in result.write_keys
        assert "zset2" in result.write_keys
        assert "zset1" in result.read_keys
        assert "zset2" in result.read_keys

    def test_bzmpop_uses_numkeys(self) -> None:
        """BZMPOP timeout numkeys key [key ...] MIN|MAX — second arg is numkeys."""
        result = parse_redis_access("BZMPOP", (0, 2, "zset1", "zset2", "MIN"))
        assert "zset1" in result.write_keys
        assert "zset2" in result.write_keys
        assert "zset1" in result.read_keys
        assert "zset2" in result.read_keys

    def test_bzpopmin_is_read_write(self) -> None:
        """BZPOPMIN pops (removes) and returns the min element — both read and write."""
        result = parse_redis_access("BZPOPMIN", ("myzset", 0))
        assert result.read_keys == ["myzset"]
        assert result.write_keys == ["myzset"]

    def test_bzpopmax_is_read_write(self) -> None:
        """BZPOPMAX pops (removes) and returns the max element — both read and write."""
        result = parse_redis_access("BZPOPMAX", ("myzset", 0))
        assert result.read_keys == ["myzset"]
        assert result.write_keys == ["myzset"]


class TestKeyCommands:
    """Test classification of Redis key commands."""

    def test_del_writes_all_keys(self) -> None:
        result = parse_redis_access("DEL", ("key1", "key2"))
        assert result.read_keys == []
        assert result.write_keys == ["key1", "key2"]

    def test_exists_reads_all_keys(self) -> None:
        result = parse_redis_access("EXISTS", ("key1", "key2"))
        assert result.read_keys == ["key1", "key2"]
        assert result.write_keys == []

    def test_expire_is_write(self) -> None:
        result = parse_redis_access("EXPIRE", ("mykey", 60))
        assert "mykey" in result.write_keys

    def test_ttl_is_read(self) -> None:
        result = parse_redis_access("TTL", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_type_is_read(self) -> None:
        result = parse_redis_access("TYPE", ("mykey",))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_rename_reads_source_writes_both(self) -> None:
        result = parse_redis_access("RENAME", ("old", "new"))
        assert "old" in result.read_keys
        assert "old" in result.write_keys
        assert "new" in result.write_keys

    def test_persist_is_read_write(self) -> None:
        """PERSIST modifies TTL metadata — should be read+write."""
        result = parse_redis_access("PERSIST", ("mykey",))
        assert "mykey" in result.read_keys
        assert "mykey" in result.write_keys

    def test_getex_is_read_write(self) -> None:
        """GETEX reads value and may modify TTL — should be read+write."""
        result = parse_redis_access("GETEX", ("mykey",))
        assert "mykey" in result.read_keys
        assert "mykey" in result.write_keys


class TestTransactionCommands:
    """Test classification of Redis transaction commands."""

    def test_multi_is_tx_control(self) -> None:
        result = parse_redis_access("MULTI", ())
        assert result.is_transaction_control is True
        assert result.read_keys == []
        assert result.write_keys == []

    def test_exec_is_tx_control(self) -> None:
        result = parse_redis_access("EXEC", ())
        assert result.is_transaction_control is True

    def test_discard_is_tx_control(self) -> None:
        result = parse_redis_access("DISCARD", ())
        assert result.is_transaction_control is True

    def test_watch_reads_keys(self) -> None:
        result = parse_redis_access("WATCH", ("key1", "key2"))
        assert result.is_transaction_control is True
        assert result.read_keys == ["key1", "key2"]


class TestSpecialCommands:
    """Test classification of special Redis commands."""

    def test_rpoplpush_reads_and_writes_both(self) -> None:
        result = parse_redis_access("RPOPLPUSH", ("source", "dest"))
        assert "source" in result.read_keys
        assert "dest" in result.read_keys
        assert "source" in result.write_keys
        assert "dest" in result.write_keys

    def test_sort_with_store(self) -> None:
        result = parse_redis_access("SORT", ("mylist", "STORE", "dest"))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == ["dest"]

    def test_sort_without_store(self) -> None:
        result = parse_redis_access("SORT", ("mylist",))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_sort_by_pattern_named_store(self) -> None:
        """SORT mylist BY store ... — 'store' is a BY pattern value, not the STORE keyword."""
        result = parse_redis_access("SORT", ("mylist", "BY", "store", "LIMIT", "0", "10"))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == [], (
            f"BY pattern 'store' should not be treated as STORE keyword, got write_keys={result.write_keys}"
        )

    def test_sort_get_pattern_named_store(self) -> None:
        """SORT mylist GET store — 'store' is a GET pattern, not STORE keyword."""
        result = parse_redis_access("SORT", ("mylist", "GET", "store", "ASC"))
        assert result.read_keys == ["mylist"]
        assert result.write_keys == []

    def test_lmpop_numkeys_exceeds_actual_keys(self) -> None:
        """LMPOP with numkeys=3 but only 2 actual keys then LEFT — the keynum extractor
        reads up to numkeys from available args.  With numkeys=3 and args (3, list1, list2, LEFT),
        the third 'key' is LEFT since the command is technically malformed."""
        result = parse_redis_access("LMPOP", (3, "list1", "list2", "LEFT"))
        # The keynum extractor reads numkeys=3 args starting at offset 1: list1, list2, LEFT.
        # In a real Redis call, numkeys would match the actual key count.
        assert "list1" in result.write_keys
        assert "list2" in result.write_keys

    def test_bzmpop_numkeys_exceeds_actual_keys(self) -> None:
        """BZMPOP with numkeys=3 but only 2 actual keys then MIN — malformed command."""
        result = parse_redis_access("BZMPOP", (0, 3, "zset1", "zset2", "MIN"))
        assert "zset1" in result.write_keys
        assert "zset2" in result.write_keys

    def test_copy_reads_source_writes_dest(self) -> None:
        result = parse_redis_access("COPY", ("source", "dest"))
        assert "source" in result.read_keys
        assert "dest" in result.write_keys

    def test_eval_is_atomic(self) -> None:
        # EVAL/EVALSHA are atomic (Lua scripts execute without interleaving).
        # Declared keys still form one command-level dependency.
        result = parse_redis_access("EVAL", ("script", 2, "key1", "key2", "arg1"))
        assert result.read_keys == []
        assert result.write_keys == ["key1", "key2"]
        assert result.is_transaction_control

    def test_pfadd_is_write(self) -> None:
        result = parse_redis_access("PFADD", ("myhll", "elem1"))
        assert "myhll" in result.write_keys

    def test_pfcount_is_read(self) -> None:
        result = parse_redis_access("PFCOUNT", ("myhll",))
        assert "myhll" in result.read_keys

    def test_pfcount_is_not_write(self) -> None:
        """PFCOUNT is read-only; it must not appear in write_keys."""
        result = parse_redis_access("PFCOUNT", ("hll1", "hll2"))
        assert result.read_keys == ["hll1", "hll2"]
        assert result.write_keys == []

    def test_pfmerge_reads_sources_writes_dest(self) -> None:
        result = parse_redis_access("PFMERGE", ("dest", "src1", "src2"))
        assert "src1" in result.read_keys
        assert "src2" in result.read_keys
        assert "dest" in result.write_keys

    def test_bitop_reads_sources_writes_dest(self) -> None:
        result = parse_redis_access("BITOP", ("AND", "dest", "src1", "src2"))
        assert result.read_keys == ["src1", "src2"]
        assert result.write_keys == ["dest"]


class TestObjectCommand:
    """Test Redis OBJECT subcommands."""

    def test_object_help_no_keys(self) -> None:
        result = parse_redis_access("OBJECT", ("HELP",))
        assert result.read_keys == []
        assert result.write_keys == []

    def test_object_encoding_reads_key(self) -> None:
        result = parse_redis_access("OBJECT", ("ENCODING", "mykey"))
        assert result.read_keys == ["mykey"]
        assert result.write_keys == []

    def test_object_no_args(self) -> None:
        result = parse_redis_access("OBJECT", ())
        assert result.read_keys == []
        assert result.write_keys == []


class TestServerCommands:
    """Test that server commands produce no key conflicts."""

    @pytest.mark.parametrize("cmd", ["PING", "INFO", "DBSIZE", "FLUSHDB", "SELECT", "AUTH"])
    def test_server_commands_no_keys(self, cmd: str) -> None:
        result = parse_redis_access(cmd, ())
        assert result.read_keys == []
        assert result.write_keys == []

    def test_ping_with_message(self) -> None:
        # PING is in the no-keys command set, so even with args it returns no keys.
        result = parse_redis_access("PING", ("hello",))
        assert result.read_keys == []
        assert result.write_keys == []


class TestKeyspaceIntentLock:
    """FLUSHDB/FLUSHALL write the whole keyspace; per-key commands read it.

    This models the database-wide keyspace as a shared resource so DPOR
    explores the racing order between FLUSH* and ordinary key traffic.
    Ordinary key-key traffic stays read-read on the keyspace (no new
    conflicts); FLUSH* takes a keyspace write that conflicts with every
    key access.
    """

    def test_flushdb_writes_keyspace(self) -> None:
        result = parse_redis_access("FLUSHDB", ())
        assert result.keyspace == "write"
        assert result.read_keys == []
        assert result.write_keys == []

    def test_flushall_writes_keyspace(self) -> None:
        result = parse_redis_access("FLUSHALL", ())
        assert result.keyspace == "write"

    def test_get_reads_keyspace(self) -> None:
        result = parse_redis_access("GET", ("k",))
        assert result.read_keys == ["k"]
        assert result.keyspace == "read"

    def test_set_reads_keyspace(self) -> None:
        result = parse_redis_access("SET", ("k", "v"))
        assert "k" in result.write_keys
        assert result.keyspace == "read"

    def test_del_reads_keyspace(self) -> None:
        result = parse_redis_access("DEL", ("k1", "k2"))
        assert result.write_keys == ["k1", "k2"]
        assert result.keyspace == "read"

    def test_keys_reads_keyspace(self) -> None:
        result = parse_redis_access("KEYS", ("*",))
        assert result.read_keys == []
        assert result.write_keys == []
        assert result.keyspace == "read"

    def test_scan_reads_keyspace(self) -> None:
        result = parse_redis_access("SCAN", ("0",))
        assert result.keyspace == "read"

    def test_randomkey_reads_keyspace(self) -> None:
        result = parse_redis_access("RANDOMKEY", ())
        assert result.keyspace == "read"

    def test_dbsize_reads_keyspace(self) -> None:
        result = parse_redis_access("DBSIZE", ())
        assert result.keyspace == "read"

    def test_plain_server_command_no_keyspace(self) -> None:
        # PING/INFO etc. do not touch the keyspace.
        assert parse_redis_access("PING", ()).keyspace is None
        assert parse_redis_access("INFO", ()).keyspace is None

    def test_transaction_control_no_keyspace(self) -> None:
        assert parse_redis_access("MULTI", ()).keyspace is None
        assert parse_redis_access("EXEC", ()).keyspace is None

    def test_eval_declared_key_reads_keyspace(self) -> None:
        # The script remains atomic, but must conflict with FLUSHDB/FLUSHALL.
        assert parse_redis_access("EVAL", ("script", "1", "k")).keyspace == "read"

    def test_publish_no_keyspace(self) -> None:
        # Pub/sub channels are not keyspace operations.
        assert parse_redis_access("PUBLISH", ("ch", "msg")).keyspace is None

    def test_colon_namespaced_key_reads_keyspace(self) -> None:
        # A real data key literally named "channel:session" must still take a
        # keyspace read so it conflicts with FLUSHDB/FLUSHALL.  The "channel:"
        # prefix is an internal pub/sub sentinel, but colon-namespaced keys are
        # idiomatic Redis and must not be mistaken for pub/sub channels.
        result = parse_redis_access("GET", ("channel:session",))
        assert result.read_keys == ["channel:session"]
        assert result.keyspace == "read"


class TestKeyspaceReporting:
    """The report path must emit a keyspace access alongside per-key accesses."""

    def _capture(self, cmd: str, args: tuple[object, ...]) -> list[tuple[str, str]]:
        from frontrun import _redis_client
        from frontrun._io_detection import set_io_reporter

        captured: list[tuple[str, str]] = []
        set_io_reporter(lambda res_id, kind: captured.append((res_id, kind)))
        try:
            _redis_client._report_redis_access(cmd, args)
        finally:
            set_io_reporter(None)
        return captured

    def test_flushdb_emits_keyspace_write(self) -> None:
        captured = self._capture("FLUSHDB", ())
        keyspace = [c for c in captured if "keyspace" in c[0].lower()]
        assert keyspace, f"FLUSHDB should report a keyspace access, got {captured}"
        assert all(kind == "write" for _, kind in keyspace), captured

    def test_get_emits_keyspace_read_and_key_read(self) -> None:
        captured = self._capture("GET", ("k",))
        keyspace = [c for c in captured if "keyspace" in c[0].lower()]
        keys = [c for c in captured if "keyspace" not in c[0].lower()]
        assert keyspace and all(kind == "read" for _, kind in keyspace), captured
        assert any(kind == "read" and "k" in res for res, kind in keys), captured

    def test_flushdb_vs_get_conflict_on_keyspace(self) -> None:
        flush = self._capture("FLUSHDB", ())
        get = self._capture("GET", ("k",))

        def keyspace_res(events: list[tuple[str, str]]) -> set[str]:
            return {res for res, _ in events if "keyspace" in res.lower()}

        flush_res = keyspace_res(flush)
        get_res = keyspace_res(get)
        # They must share a keyspace resource id and disagree on kind (write vs read)
        # so DPOR sees a conflict.
        assert flush_res & get_res, f"FLUSHDB and GET must share a keyspace resource: {flush_res} vs {get_res}"


class TestPubSubCommands:
    """Test Pub/Sub channel handling."""

    def test_publish_writes_channel(self) -> None:
        result = parse_redis_access("PUBLISH", ("mychannel", "message"))
        assert result.write_keys == ["channel:mychannel"]

    def test_subscribe_reads_channels(self) -> None:
        result = parse_redis_access("SUBSCRIBE", ("ch1", "ch2"))
        assert result.read_keys == ["channel:ch1", "channel:ch2"]

    def test_psubscribe_reads_channels(self) -> None:
        result = parse_redis_access("PSUBSCRIBE", ("pattern*",))
        assert result.read_keys == ["channel:pattern*"]


class TestStreamCommands:
    """Test stream command key extraction."""

    def test_xadd_is_write(self) -> None:
        result = parse_redis_access("XADD", ("mystream", "*", "field", "value"))
        assert "mystream" in result.write_keys

    def test_xread_extracts_stream_keys(self) -> None:
        result = parse_redis_access("XREAD", ("COUNT", "10", "STREAMS", "s1", "s2", "0-0", "0-0"))
        assert result.read_keys == ["s1", "s2"]
        assert result.write_keys == []

    def test_xreadgroup_extracts_stream_keys(self) -> None:
        result = parse_redis_access("XREADGROUP", ("GROUP", "g1", "c1", "STREAMS", "s1", "s2", ">", ">"))
        assert "s1" in result.read_keys
        assert "s2" in result.read_keys
        assert "s1" in result.write_keys
        assert "s2" in result.write_keys

    def test_xlen_is_read(self) -> None:
        result = parse_redis_access("XLEN", ("mystream",))
        assert result.read_keys == ["mystream"]
        assert result.write_keys == []

    def test_xgroup_create_writes_key(self) -> None:
        result = parse_redis_access("XGROUP", ("CREATE", "mystream", "mygroup", "0"))
        assert "mystream" in result.write_keys


class TestGeoCommands:
    """Test geo command key extraction."""

    def test_geoadd_is_write(self) -> None:
        result = parse_redis_access("GEOADD", ("mygeo", 13.361389, 38.115556, "Palermo"))
        assert "mygeo" in result.write_keys

    def test_geodist_is_read(self) -> None:
        result = parse_redis_access("GEODIST", ("mygeo", "Palermo", "Catania"))
        assert result.read_keys == ["mygeo"]
        assert result.write_keys == []

    def test_geosearchstore_reads_source_writes_dest(self) -> None:
        result = parse_redis_access(
            "GEOSEARCHSTORE", ("dest", "src", "FROMLONLAT", "15", "37", "BYRADIUS", "100", "km")
        )
        assert "src" in result.read_keys
        assert "dest" in result.write_keys

    def test_georadius_with_store(self) -> None:
        result = parse_redis_access("GEORADIUS", ("mygeo", 15, 37, 100, "km", "STORE", "dest"))
        assert "mygeo" in result.read_keys
        assert "dest" in result.write_keys

    def test_georadius_without_store(self) -> None:
        result = parse_redis_access("GEORADIUS", ("mygeo", 15, 37, 100, "km"))
        assert "mygeo" in result.read_keys
        assert result.write_keys == []


class TestCaseSensitivity:
    """Test case-insensitive command handling."""

    def test_lowercase_command(self) -> None:
        result = parse_redis_access("get", ("mykey",))
        assert result.read_keys == ["mykey"]

    def test_mixed_case_command(self) -> None:
        result = parse_redis_access("Set", ("mykey", "val"))
        assert "mykey" in result.write_keys


class TestFallbackBehavior:
    """Test handling of unknown commands."""

    def test_unknown_command_with_args_falls_back_to_write(self) -> None:
        result = parse_redis_access("UNKNOWNCMD", ("arg1", "arg2"))
        assert result.write_keys == ["arg1"]

    def test_unknown_command_no_args_returns_empty(self) -> None:
        result = parse_redis_access("UNKNOWNCMD", ())
        assert result.read_keys == []
        assert result.write_keys == []


class TestCommandCoverage:
    """Test that our command table covers all important Redis commands."""

    # Every command that appears in the key-spec table should extract at least
    # one key when given a plausible argument list.
    def test_all_core_commands_are_handled(self) -> None:
        """Smoke test: commands with known key-specs should not fall through to fallback."""
        from frontrun._redis_command_data import _COMMAND_KEY_SPECS

        # Commands that use keyword-based begin_search need the keyword in args.
        keyword_args: dict[str, tuple[object, ...]] = {
            "XREAD": ("COUNT", "10", "STREAMS", "s1", "0-0"),
            "XREADGROUP": ("GROUP", "g1", "c1", "COUNT", "10", "STREAMS", "s1", ">"),
            "GEORADIUS": ("mygeo", "15", "37", "100", "km"),
            "GEORADIUSBYMEMBER": ("mygeo", "member", "100", "km"),
            "MIGRATE": ("host", "6379", "key1", "0", "1000"),
        }

        for cmd_name in _COMMAND_KEY_SPECS:
            # For subcommands like "OBJECT ENCODING", split into cmd_name + subcommand arg.
            if " " in cmd_name:
                parts = cmd_name.split(" ", 1)
                actual_cmd = parts[0]
                prefix = (parts[1],)
            else:
                actual_cmd = cmd_name
                prefix = ()

            if cmd_name in keyword_args:
                actual_args = prefix + keyword_args[cmd_name]
            else:
                # Start at 1 so keynum args get numkeys=1 not 0.
                actual_args = prefix + tuple(str(i) for i in range(1, 21))

            result = parse_redis_access(actual_cmd, actual_args)
            assert result.read_keys or result.write_keys or result.is_transaction_control, (
                f"Command {cmd_name} with args {actual_args[:5]}... returned no keys — "
                f"likely hitting the fallback. Add explicit handling."
            )


class TestCompoundSubcommandKeyExtraction:
    """redis-py 4.2+ sends subcommands as compound names like 'XGROUP CREATE'.

    When cmd_name contains a space, the subcommand word is NOT in cmd_args,
    so key-spec indices must be adjusted.
    """

    def test_xgroup_create_compound(self):
        result = parse_redis_access("XGROUP CREATE", ("mystream", "grp", "0"))
        assert "mystream" in result.write_keys

    def test_xgroup_destroy_compound(self):
        result = parse_redis_access("XGROUP DESTROY", ("mystream", "grp"))
        assert "mystream" in result.write_keys

    def test_xinfo_groups_compound(self):
        result = parse_redis_access("XINFO GROUPS", ("mystream",))
        assert "mystream" in result.read_keys

    def test_xinfo_stream_compound(self):
        result = parse_redis_access("XINFO STREAM", ("mystream",))
        assert "mystream" in result.read_keys

    def test_memory_usage_compound(self):
        result = parse_redis_access("MEMORY USAGE", ("mykey",))
        assert "mykey" in result.read_keys

    def test_xgroup_setid_compound(self):
        result = parse_redis_access("XGROUP SETID", ("mystream", "grp", "0"))
        assert "mystream" in result.write_keys


class TestCompoundNoKeyCmds:
    """Compound command names for server/admin commands should not produce false keys."""

    def test_client_setname_no_keys(self):
        result = parse_redis_access("CLIENT SETNAME", ("myname",))
        assert result.read_keys == []
        assert result.write_keys == []

    def test_config_get_no_keys(self):
        result = parse_redis_access("CONFIG GET", ("maxmemory",))
        assert result.read_keys == []
        assert result.write_keys == []

    def test_slowlog_get_no_keys(self):
        result = parse_redis_access("SLOWLOG GET", ("10",))
        assert result.read_keys == []
        assert result.write_keys == []

    def test_acl_getuser_no_keys(self):
        result = parse_redis_access("ACL GETUSER", ("admin",))
        assert result.read_keys == []
        assert result.write_keys == []

    def test_cluster_info_no_keys(self):
        result = parse_redis_access("CLUSTER INFO", ())
        assert result.read_keys == []
        assert result.write_keys == []

    def test_script_exists_no_keys(self):
        result = parse_redis_access("SCRIPT EXISTS", ("abc123",))
        assert result.read_keys == []
        assert result.write_keys == []


class TestReplaySchedulingPoint:
    """Replay must recreate a scheduling point for every command that made one
    during exploration, including keyspace-only commands (FLUSHDB, KEYS, SCAN,
    ...) which carry a keyspace intent-lock but no read/write keys.
    """

    def test_keyspace_only_commands_need_replay_scheduling_point(self):
        from frontrun._redis_client import _replay_needs_scheduling_point

        for cmd, args in [
            ("FLUSHDB", ()),
            ("FLUSHALL", ()),
            ("KEYS", ("*",)),
            ("SCAN", ("0",)),
        ]:
            access = parse_redis_access(cmd, args)
            # Sanity: these carry a keyspace lock but no key-level access.
            assert access.keyspace is not None
            assert not access.read_keys and not access.write_keys
            assert _replay_needs_scheduling_point(access), (
                f"{cmd} carries a keyspace intent-lock and gets a scheduling point "
                f"during exploration, so replay must recreate one too"
            )

    def test_key_commands_still_need_replay_scheduling_point(self):
        from frontrun._redis_client import _replay_needs_scheduling_point

        assert _replay_needs_scheduling_point(parse_redis_access("GET", ("k",)))
        assert _replay_needs_scheduling_point(parse_redis_access("SET", ("k", "v")))

    def test_connection_setup_commands_skip_replay_scheduling_point(self):
        from frontrun._redis_client import _replay_needs_scheduling_point

        # AUTH/SELECT carry no key or keyspace intent-lock.
        access = parse_redis_access("AUTH", ("pw",))
        assert not _replay_needs_scheduling_point(access)
