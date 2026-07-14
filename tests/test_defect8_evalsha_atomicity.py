"""Defect #8: DPOR does not model Redis Lua script (EVALSHA) atomicity.

DPOR reports false positive races for Redis operations protected by Lua
scripts because it treats EVALSHA key accesses as interleaving operations.
Redis Lua scripts execute atomically — no other commands can run between
the script's Redis calls.

The fix: treat EVAL/EVALSHA as one atomic transaction-control command.  Its
declared keys still form a command-level dependency so DPOR explores opposite
whole-script orderings; it must never insert a scheduling point *inside* the
script.
"""

from __future__ import annotations

from frontrun._redis_parsing import parse_redis_access


class TestEvalshaAtomicity:
    """EVAL/EVALSHA should be treated as atomic operations."""

    def test_evalsha_declared_keys_are_atomic_writes(self) -> None:
        result = parse_redis_access("EVALSHA", ("abc123", "2", "key1", "key2", "arg1"))
        assert result.read_keys == [], f"EVALSHA should not report read keys, got {result.read_keys}"
        assert result.write_keys == ["key1", "key2"]
        assert result.is_transaction_control, "EVALSHA should be marked as transaction control (atomic)"

    def test_eval_declared_keys_are_atomic_writes(self) -> None:
        result = parse_redis_access("EVAL", ("return 1", "1", "key1"))
        assert result.read_keys == []
        assert result.write_keys == ["key1"]
        assert result.is_transaction_control

    def test_evalsha_ro_declared_keys_are_atomic_reads(self) -> None:
        result = parse_redis_access("EVALSHA_RO", ("abc123", "1", "key1"))
        assert result.read_keys == ["key1"]
        assert result.write_keys == []
        assert result.is_transaction_control

    def test_eval_ro_declared_keys_are_atomic_reads(self) -> None:
        result = parse_redis_access("EVAL_RO", ("return 1", "1", "key1"))
        assert result.read_keys == ["key1"]
        assert result.write_keys == []
        assert result.is_transaction_control
