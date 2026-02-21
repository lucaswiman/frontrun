Postgres Race Conditions
========================

This guide demonstrates two classic database race conditions using frontrun
against a live Postgres 16 instance via psycopg2.  Both races are full
integration tests — no simulation, no mocking.

.. code-block:: text

    python examples/postgres_races.py   # run to reproduce the traces below

The full source is in ``examples/postgres_races.py``.


Race 1: Lost Update (READ COMMITTED data race)
-----------------------------------------------

**What it is**

A *lost update* occurs when two concurrent transactions each read the same row,
compute a new value in application code, and write it back — without holding a
row-level lock.  The second ``UPDATE`` silently overwrites the first, losing one
credit.

**SQL pattern** (READ COMMITTED isolation, Postgres default)::

    -- Transaction A
    BEGIN;
    SELECT balance FROM accounts WHERE name='alice';   -- reads 1000
    -- (application: new_balance = 1000 + 100)
    UPDATE accounts SET balance = 1100 WHERE name='alice';
    COMMIT;

    -- Transaction B (concurrent — both read before either writes)
    BEGIN;
    SELECT balance FROM accounts WHERE name='alice';   -- also reads 1000!
    -- (application: new_balance = 1000 + 200)
    UPDATE accounts SET balance = 1200 WHERE name='alice';   -- overwrites A
    COMMIT;

    -- Final balance: 1200  (should be 1300)

**frontrun tool: ``TraceExecutor`` (trace markers)**

``TraceExecutor`` uses Python's ``sys.settrace`` to watch for inline
``# frontrun: <marker>`` comments on executable statements.  A
``Step(thread, marker)`` in the schedule means *run that thread until it is
about to execute the tagged line, then pause*.  This gives precise control
over when each psycopg2 ``execute()`` call runs inside Postgres.

**Python code**::

    def txn_a() -> None:
        with conn_a.cursor() as cur:
            cur.execute("SELECT balance FROM accounts WHERE name='alice'")  # frontrun: pg_select
            old = cur.fetchone()[0]
            new = old + 100
        with conn_a.cursor() as cur2:
            cur2.execute("UPDATE accounts SET balance=%s WHERE name='alice'", (new,))  # frontrun: pg_update
        conn_a.commit()

    # (txn_b is identical but adds 200)

    schedule = Schedule([
        Step("txn_a", "pg_select"),   # pause Txn A just before its SELECT
        Step("txn_b", "pg_select"),   # pause Txn B just before its SELECT
        Step("txn_a", "pg_update"),   # Txn A runs SELECT (reads 1000), pause before UPDATE
        Step("txn_b", "pg_update"),   # Txn B runs SELECT (reads 1000), pause before UPDATE
        # (schedule exhausted) → both run UPDATE + COMMIT freely
    ])

    executor = TraceExecutor(schedule)
    executor.run("txn_a", txn_a)
    executor.run("txn_b", txn_b)
    executor.wait(timeout=10.0)

**Reproduction trace** (exact program output)::

    ======================================================================
    Race 1: Lost Update  (TraceExecutor + psycopg2 + Postgres)
    ======================================================================

      SQL pattern (READ COMMITTED, no FOR UPDATE):
        Txn A: BEGIN; SELECT balance; -- compute 1000+100=1100
                      UPDATE balance=1100; COMMIT;
        Txn B: BEGIN; SELECT balance; -- compute 1000+200=1200
                      UPDATE balance=1200; COMMIT;  ← overwrites Txn A

      Initial balance : 1000
      Expected result : 1000 + 100 + 200 = 1300

      Initial balance                          : 1000
      Txn B's read (after its own UPDATE)      : 1200
      Actual final balance in DB               : 1200  (expected 1300)

      LOST UPDATE confirmed: balance is 1200, not 1300.
      Txn B overwrote Txn A's update.
      Both transactions read 1000 before either wrote back.

      Reproducibility: 100% — the Schedule deterministically forces
      both SELECTs to run before either UPDATE on every execution.

**Reading the trace**

Txn B's read after its own UPDATE is ``1200``, not ``1300``.  This confirms
that Txn B saw the initial balance (``1000``), added ``200``, and wrote
``1200`` — overwriting Txn A's earlier write of ``1100``.  The Schedule
is the reason it is reproducible: ``Step("txn_a", "pg_select")`` followed
by ``Step("txn_b", "pg_select")`` guarantees that both transactions have
issued their SELECT before either issues its UPDATE, every single run.

**Reproducibility: 100%**

The ``Schedule`` forces the exact interleaving on every execution.

**Fix**

Use ``SELECT … FOR UPDATE`` to hold a row-level lock, or raise isolation::

    -- Fix: acquire a row lock before reading
    BEGIN;
    SELECT balance FROM accounts WHERE name='alice' FOR UPDATE;
    UPDATE accounts SET balance = balance + 100 WHERE name='alice';
    COMMIT;


Race 2: Deadlock (lock-ordering cycle)
-----------------------------------------

**What it is**

A *deadlock* occurs when two transactions each hold one row-level lock and
wait for the other's lock, creating a circular dependency.  Postgres detects
the cycle and aborts one transaction.

**SQL pattern**::

    -- Transaction 1 (alice → bob order)
    BEGIN;
    SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock alice
    SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock bob
    UPDATE accounts SET balance = balance - 100 WHERE name='alice';
    UPDATE accounts SET balance = balance + 100 WHERE name='bob';
    COMMIT;

    -- Transaction 2 (bob → alice order — OPPOSITE!)
    BEGIN;
    SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock bob
    SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock alice ← DEADLOCK!
    ...

**frontrun tool: ``threading.Event`` coordination**

Two ``threading.Event`` objects guarantee the deadlock-causing ordering.
Txn1 signals after locking Alice; Txn2 waits for that signal, locks Bob,
then signals.  Txn1 waits for Txn2's signal before trying Bob — by which
point Txn2 already holds Bob and is trying Alice.  Circular wait confirmed::

    alice_locked = threading.Event()
    bob_locked   = threading.Event()

    def txn1() -> None:
        with conn1.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='alice' FOR UPDATE")
        alice_locked.set()          # signal: alice row-lock held by Txn1
        bob_locked.wait(timeout=5)  # wait:   ensure Txn2 holds bob before we try it
        with conn1.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='bob' FOR UPDATE")  # ← BLOCKED
            ...

    def txn2() -> None:
        alice_locked.wait(timeout=5)  # wait: ensure Txn1 holds alice before we lock bob
        with conn2.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='bob' FOR UPDATE")
        bob_locked.set()            # signal: bob row-lock held by Txn2; Txn1 now tries bob → BLOCKED
        with conn2.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='alice' FOR UPDATE")  # ← DEADLOCK

**Reproduction trace** (exact program output)::

    ======================================================================
    Race 2: Deadlock  (threading.Event coordination + psycopg2 + Postgres)
    ======================================================================

      SQL pattern (FOR UPDATE, opposite lock order):
        Txn1: LOCK alice → LOCK bob   → transfer alice→bob ($100)
        Txn2: LOCK bob   → LOCK alice → transfer bob→alice ($50)

      Interleaving forced by threading.Event coordination:
        Step 1: Txn1 acquires alice row-lock  (SELECT … FOR UPDATE)
        Step 2: Txn2 acquires bob   row-lock  (SELECT … FOR UPDATE)
        Step 3: Txn1 tries bob   row-lock  ← BLOCKED (held by Txn2)
        Step 4: Txn2 tries alice row-lock  ← BLOCKED (held by Txn1) → DEADLOCK

      txn2 received Postgres error:
        deadlock detected
        DETAIL:  Process 19592 waits for ShareLock on transaction 766; blocked by process 19591.
        Process 19591 waits for ShareLock on transaction 767; blocked by process 19592.
        HINT:  See server log for query details.
        CONTEXT:  while locking tuple (0,1) in relation "accounts"

      DEADLOCK confirmed: Postgres aborted one transaction and rolled
      it back, freeing the other transaction to complete.

      Reproducibility: 100% — the threading.Event coordination
      guarantees the deadlock-causing lock-ordering on every run.

**Reading the trace**

Postgres aborted ``txn2`` (the one that tried to lock Alice after Txn1
already held it) and raised ``deadlock detected`` with process IDs and the
specific tuple it was blocked on.  Txn1 then completed normally.  The
``threading.Event`` coordination is what makes this 100% reproducible: the
``alice_locked`` / ``bob_locked`` events enforce the exact lock-acquisition
ordering that creates the cycle.

**Reproducibility: 100%**

The ``threading.Event`` coordination guarantees the deadlock-causing ordering
on every run.  Process IDs and transaction numbers vary between runs; the
deadlock itself is invariant.

**Fix**

Always acquire row locks in a globally consistent order (e.g. sorted by
primary key or name)::

    -- Both transactions lock in alphabetical order: alice first, then bob
    SELECT * FROM accounts WHERE name = ANY(ARRAY['alice','bob'])
    ORDER BY name FOR UPDATE;   -- alice before bob, always


Why not DPOR for psycopg2?
---------------------------

DPOR instruments Python bytecode (``LOAD_ATTR`` / ``STORE_ATTR`` opcodes).
psycopg2 is a C extension that drives libpq directly — its ``execute()`` and
``fetchone()`` calls bypass Python attribute access entirely.  DPOR cannot
observe or control them.

Use ``TraceExecutor`` (line-level ``sys.settrace``) to control *when* psycopg2
calls execute; use ``threading.Event`` to coordinate lock-ordering for deadlock
scenarios.

DPOR remains the right tool when the shared state is Python objects — for
example, an ORM layer or an in-process cache that sits in front of Postgres.


Tool selection summary
-----------------------

+----------------------------+--------------------------+------------------------------------------+
| Bug class                  | Right tool               | Why                                      |
+============================+==========================+==========================================+
| Lost update (data race)    | ``TraceExecutor``        | sys.settrace controls when each          |
| with real DB connection    | (trace markers)          | psycopg2 execute() fires in Postgres     |
+----------------------------+--------------------------+------------------------------------------+
| Deadlock (lock ordering)   | ``threading.Event``      | Events guarantee the exact lock-order    |
| with real DB connection    | coordination             | that creates the circular wait           |
+----------------------------+--------------------------+------------------------------------------+
| Lost update (data race)    | ``explore_dpor``         | DPOR tracks LOAD_ATTR/STORE_ATTR;        |
| on Python objects / ORM    |                          | systematically explores interleavings    |
+----------------------------+--------------------------+------------------------------------------+

Both races are **100% reproducible** once the triggering interleaving is known:

* For the lost update, the ``Schedule`` deterministically forces both SELECTs
  before either UPDATE on every execution.
* For the deadlock, the ``threading.Event`` objects enforce the lock-acquisition
  ordering that creates the cycle.
