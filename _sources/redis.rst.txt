Redis Conflict Detection -- Technical Details
=============================================

Frontrun's Redis conflict detection intercepts ``execute_command()`` calls on
redis-py clients, classifies each command as a read or write on specific keys,
and reports per-key resource IDs to the DPOR engine.  This replaces the coarse
endpoint-level detection where all Redis traffic to the same ``(host, port)``
collapses to a single conflict point.


How a Redis Command Becomes a DPOR Conflict
--------------------------------------------

Consider two threads operating on the same Redis server.  Thread A executes:

.. code-block:: python

   val = r.get("counter")      # reads key "counter"
   r.set("counter", val + 1)   # writes key "counter"

Thread B executes:

.. code-block:: python

   r.set("user:42:name", "Alice")   # writes key "user:42:name"

Without key-level detection, both threads produce ``send()``/``recv()`` on the
same socket (``localhost:6379``).  DPOR sees a conflict on a single resource ID
and explores O(n!) interleavings --- none of which can actually race, because
the two threads touch completely different keys.

With Redis detection enabled, the data flows through three stages:

**Stage 1 -- Command classification** (``_redis_parsing.py``).
``parse_redis_access(command, args)`` returns ``(read_keys, write_keys,
is_transaction_control)``.  The parser covers all major Redis command groups:

- String commands: ``GET`` (read), ``SET`` (write), ``INCR``/``INCRBY`` (read+write),
  ``GETSET``/``GETDEL`` (read+write), ``MGET`` (read, multiple keys), ``MSET``
  (write, multiple keys), etc.
- Hash commands: ``HGET``/``HMGET``/``HGETALL`` (read), ``HSET``/``HMSET``/``HDEL`` (write).
- List commands: ``LRANGE``/``LINDEX`` (read), ``LPUSH``/``RPUSH``/``LPOP``/``RPOP`` (write),
  ``LMOVE``/``RPOPLPUSH`` (read+write on source, write on destination).
- Set commands: ``SMEMBERS``/``SISMEMBER`` (read), ``SADD``/``SREM`` (write),
  ``SMOVE`` (read+write on source, write on destination).
- Sorted set commands: ``ZRANGE``/``ZSCORE`` (read), ``ZADD``/``ZREM`` (write),
  ``ZPOPMIN``/``ZPOPMAX`` (write), ``ZMPOP``/``BZMPOP`` (write).
- Key commands: ``EXISTS``/``TTL``/``TYPE`` (read), ``DEL``/``EXPIRE``/``PERSIST`` (write),
  ``RENAME``/``RENAMENX`` (read+write on source, write on destination).
- HyperLogLog: ``PFCOUNT`` (read), ``PFADD`` (write), ``PFMERGE`` (read on sources, write on destination).
- Stream commands: ``XRANGE``/``XREAD`` (read), ``XADD``/``XDEL``/``XTRIM`` (write),
  ``XREADGROUP`` (read+write --- advances the consumer group last-delivered ID).
- Pub/Sub: ``PUBLISH`` (write to channel), ``SUBSCRIBE``/``PSUBSCRIBE`` (read from channel).
- Geo commands: ``GEOPOS``/``GEODIST`` (read), ``GEOADD`` (write),
  ``GEORADIUS``/``GEORADIUSBYMEMBER`` (read, or read+write if ``STORE``/``STOREDIST`` is used).
- Script commands: ``EVAL``/``EVALSHA``/``FCALL`` conservatively report all
  key arguments as read+write (scripts may do anything).
- Transaction control: ``MULTI``/``EXEC``/``DISCARD``/``WATCH``/``UNWATCH`` are
  flagged as ``is_transaction_control=True`` with no key sets.
- Server/cluster commands: ``PING``, ``INFO``, ``CONFIG``, ``CLUSTER``, etc. produce
  empty read/write sets (no DPOR conflict).

Over 160 Redis commands are handled explicitly; any unrecognised command falls
back to a conservative read+write classification on all its key arguments.

**Stage 2 -- Monkey-patching** (``_redis_client.py`` for sync,
``_redis_client_async.py`` for async).
``patch_redis()`` / ``patch_redis_async()`` wraps the low-level
``execute_command()`` method that every high-level redis-py method
(``r.get()``, ``r.set()``, ``r.hset()``, etc.) funnels through.  The wrapper:

1. Calls ``parse_redis_access()`` on the command and arguments.
2. Suppresses the coarser endpoint-level socket I/O report for this call
   (sets a thread-local flag that the socket monkey-patch checks).
3. Calls the real ``execute_command()`` to execute the command on Redis.
4. Reports each read key as ``IoKind.Read`` on resource ID
   ``redis:<key>@<host>:<port>`` and each write key as ``IoKind.Write`` via the
   active ``IOReporter``.

The ``@<host>:<port>`` suffix means keys on different Redis servers are always
independent, even if they have the same name.

**Stage 3 -- DPOR engine** (same Rust engine as for bytecode and SQL).
The engine receives ``(thread_id, resource_id, access_kind)`` tuples and tracks
them with vector clocks exactly as it does for Python-level attribute accesses.
Two accesses conflict when they touch the same ``resource_id`` with at least one
``Write``.  At each conflict point the engine inserts a backtrack entry so the
alternative ordering is explored in a future execution.


Scheduling points and TOCTOU races
------------------------------------

Redis interception injects explicit DPOR scheduling points (``scheduler.pause()``)
around each Redis command.  This gives DPOR tight boundaries to explore:

.. code-block:: python

   # Thread A (with DPOR scheduling points shown)
   exists_result = r.exists("resource")   # ← scheduling point before/after EXISTS
   if not exists_result:
       r.set("resource", "initialized")   # ← scheduling point before/after SET

   # Thread B (same code)
   exists_result = r.exists("resource")
   if not exists_result:
       r.set("resource", "initialized")

Without explicit scheduling points, the cooperative scheduler might not switch
threads between the ``EXISTS`` and the ``SET``.  With them, DPOR explores the
interleaving where B's ``EXISTS`` runs after A's ``EXISTS`` but before A's
``SET`` --- exposing the double-initialization race.

asyncio.Lock interactions during connection-pool waits are handled specially:
scheduling is suppressed while waiting for a pool connection, preventing false
DPOR backtrack points from connection management code.


Pipeline support
-----------------

Each command in a redis-py pipeline is intercepted individually as the pipeline
is built.  The commands are batched and sent to Redis in one round trip, but
DPOR sees each command's read/write set separately and can explore interleaving
at the command level.

.. code-block:: python

   pipe = r.pipeline()
   pipe.get("balance")        # reported as Read on "redis:balance@..."
   pipe.set("balance", 200)   # reported as Write on "redis:balance@..."
   pipe.execute()


Transaction support (MULTI/EXEC)
----------------------------------

``MULTI``/``EXEC`` blocks are handled as follows:

- ``MULTI`` is flagged as transaction control; no resource is reported.
- Commands inside the transaction are classified normally and their read/write
  sets are buffered.
- ``EXEC`` flushes the buffered sets atomically to the DPOR engine.
- ``DISCARD`` clears the buffer without reporting.

This mirrors the SQL transaction grouping behaviour (``BEGIN``/``COMMIT``).


Activation
-----------

**Sync** ``explore_dpor``:
Redis patching is active whenever ``detect_io=True`` (the default).  No extra
parameter is needed.

.. code-block:: python

   from frontrun.dpor import explore_dpor

   result = explore_dpor(
       setup=State,
       threads=[thread_a, thread_b],
       invariant=check_invariant,
       detect_io=True,    # default; activates Redis key-level patching
   )

**Async** ``frontrun.explore`` with async workers:
``detect_io=True`` also activates patching for ``redis.asyncio`` and
``coredis`` clients from 0.5 onwards.

.. code-block:: python

   from frontrun import explore

   result = await explore(
       setup=State,
       workers=[task_a, task_b],
       invariant=check_invariant,
       detect_io=True,
   )

When ``detect_io=True`` is set (sync or async), the coarse endpoint-level
socket I/O reports for Redis connections are automatically suppressed.

.. note::

   The async-only ``detect_redis=True`` kwarg is deprecated in 0.5 and
   will be removed in 0.6.  Use ``detect_io=True`` instead.

Without suppression, every ``send()`` (classified as a write on
``socket:<host>:<port>``) and every ``recv()`` (a read on the same resource)
would conflict with every other send/recv to the same server.  Ten Redis
commands in each of two threads would generate roughly 200 spurious conflict
pairs, causing DPOR to explore a combinatorial explosion of interleavings that
cannot actually differ in outcome.  Suppressing the socket-level reports and
replacing them with per-key resource IDs means DPOR only explores orderings
that can genuinely change the result.


Command coverage verification
-------------------------------

The test suite (``tests/test_redis_parsing.py``) includes a
``TestCommandCoverage`` class that verifies all 160+ core Redis commands are
handled explicitly --- none fall through to the conservative fallback.  It also
checks that no command appears in multiple mutually-exclusive classification
sets (e.g. both ``_SINGLE_KEY_READ_CMDS`` and ``_SINGLE_KEY_WRITE_CMDS``).

This coverage test is run as part of ``make test``.


Known limitations
------------------

- **Lua scripts** (``EVAL``, ``EVALSHA``, ``FCALL``): classified conservatively
  as read+write on all declared key arguments.  The actual keys accessed inside
  the script are not inspected.
- **Pub/Sub channels**: treated as independent resources from regular keys.
  A ``PUBLISH`` to ``"events"`` and a ``GET`` on ``"events"`` do not conflict
  (they use different Redis namespaces).
- **Async DPOR excess paths**: when both opcode-level tracing and Redis
  key-level detection are active, shared Python state (e.g. a shared counter
  object) may still produce additional backtrack points beyond those from Redis
  keys alone.  See ``ideas/KNOWN_ISSUES.md`` for details.
- **coredis async client**: supported but less tested than ``redis.asyncio``.
