"""Registry of Redis patch targets used by the Redis client modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedisTarget:
    """Method patch target for Redis client classes."""

    module_name: str
    class_name: str
    method_name: str


SYNC_REDIS_TARGETS: tuple[RedisTarget, ...] = (
    RedisTarget("redis", "Redis", "execute_command"),
    RedisTarget("redis", "StrictRedis", "execute_command"),
    RedisTarget("redis.client", "Pipeline", "immediate_execute_command"),
    RedisTarget("redis.client", "Pipeline", "execute"),
    RedisTarget("redis.client", "PubSub", "execute_command"),
)

ASYNC_REDIS_TARGETS: tuple[RedisTarget, ...] = (
    RedisTarget("redis.asyncio", "Redis", "execute_command"),
    RedisTarget("redis.asyncio", "Pipeline", "immediate_execute_command"),
    RedisTarget("redis.asyncio", "Pipeline", "execute"),
    RedisTarget("redis.asyncio.client", "Pipeline", "immediate_execute_command"),
    RedisTarget("redis.asyncio.client", "Pipeline", "execute"),
    RedisTarget("redis.asyncio.client", "PubSub", "execute_command"),
    RedisTarget("coredis", "Redis", "execute_command"),
)
