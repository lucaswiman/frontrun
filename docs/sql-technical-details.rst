SQL Conflict Detection -- Technical Details
============================================

Frontrun's SQL conflict detection intercepts ``cursor.execute()`` calls,
parses the SQL, and reports per-table (or per-row) resource IDs to the
DPOR engine.  This replaces the coarse endpoint-level detection where
all SQL to the same ``(host, port)`` collapses to a single conflict
point.


How a SQL Statement Becomes a DPOR Conflict
-------------------------------------------

Consider two threads running against the same PostgreSQL database.
Thread A executes:

.. code-block:: sql

   SELECT * FROM accounts WHERE id = 1;

Thread B executes:

.. code-block:: sql

   UPDATE accounts SET balance = 0 WHERE id = 2;

Without SQL-level detection, both statements produce ``send()``/``recv()``
on the same socket (``localhost:5432``).  DPOR sees a write-write conflict
on a single ``ObjectId`` and explores O(n!) interleavings -- none of which
can actually race, because the two statements touch different rows on
different tables.

With SQL detection enabled, the data flows through six stages:

**Stage 1 -- SQL parsing** (``_sql_parsing.py``).
``parse_sql_access(sql)`` returns ``(read_tables, write_tables,
lock_intent, tx_op)``.  A regex fast-path handles ~90% of ORM-generated
SQL (single-table SELECT/INSERT/UPDATE/DELETE).  Complex SQL (CTEs,
subqueries, UNION, MERGE) falls back to sqlglot for full AST analysis.
Transaction control statements (BEGIN, COMMIT, ROLLBACK, SAVEPOINT) are
detected and returned as ``tx_op`` with no table sets.

**Stage 2 -- Parameter resolution** (``_sql_params.py``).
ORM queries are parameterized (``WHERE id = %s``), so sqlglot sees
placeholders, not values.  ``resolve_parameters()`` substitutes the
actual Python values before AST analysis.  All five PEP 249 paramstyles
are supported: qmark (``?``), numeric (``:1``), named (``:name``),
format (``%s``), and pyformat (``%(name)s``).  The resolved SQL is only
used for predicate extraction -- it is never executed.

**Stage 3 -- Predicate extraction** (``_sql_predicates.py``).
``extract_equality_predicates()`` parses the WHERE clause and extracts
equality and IN-list predicates on primary key columns:

.. code-block:: python

   # Input (after parameter resolution):
   "SELECT * FROM accounts WHERE id = 42 AND region = 'us-east'"

   # Output:
   [EqualityPredicate("id", "42"), EqualityPredicate("region", "us-east")]

When both operations have complete PK predicates, disjointness is
checked with ``frozenset.isdisjoint()`` -- O(k) for k PK columns.

**Stage 4 -- ObjectId derivation**.
The resource ID is a string like ``"sql:accounts"`` (table-level) or
``"sql:accounts:(('id','42'),)"`` (row-level).  This string is hashed
via the existing ``_make_object_key()`` to produce a ``u64`` ObjectId
for the Rust DPOR engine.  No Rust changes are needed -- SQL tables are
just I/O objects with table-derived ObjectIds.

**Stage 5 -- Access kind mapping**.
SELECT maps to ``AccessKind::Read``.  INSERT maps to ``Write``.  UPDATE
and DELETE map to both ``Read`` and ``Write`` (conservative).  SELECT
FOR UPDATE maps to ``Write``.  The Rust engine's
``ObjectState::dependent_accesses`` implements the standard conflict
rules: Read-Read is independent, everything else (RW, WR, WW) conflicts.

**Stage 6 -- Endpoint suppression** (``_sql_cursor.py``).
While the original ``cursor.execute()`` runs, a context manager sets
``_sql_suppress = True`` in thread-local storage and adds the OS thread
ID to a shared ``_suppress_tids`` set.  This prevents the coarser
socket-level detection from double-reporting the same operation --
both in Python (``_report_socket_io`` checks TLS) and in the LD_PRELOAD
bridge (``_PreloadBridge.listener`` checks the shared set).


The Refinement Hierarchy
------------------------

.. code-block:: text

   Row-level  <  Table-level  <  Endpoint-level
   (finest)                      (coarsest)

Each layer is a sound refinement of the one below.  Endpoint-level
treats all SQL to the same host:port as conflicting -- sound but
explores many unnecessary interleavings.  Table-level parses SQL to
extract table names; operations on different tables are independent,
and same-table operations use Adya Read/Write classification.
Row-level (when both operations have complete PK predicates) treats
same-table, different-PK operations as independent.

When parsing fails, the system falls back to the next coarser layer.
Parse failure returns empty table sets, so endpoint-level suppression
never fires and socket-level detection remains in effect.  When
parameter resolution fails, row-level falls back to table-level.
Fallback is always in the conservative direction.


Transaction Grouping
--------------------

SQL operations within a ``BEGIN`` ... ``COMMIT`` block are grouped
atomically.  When ``_intercept_execute()`` sees ``BEGIN``, it sets
``_in_transaction = True`` and starts buffering I/O reports in
``_tx_buffer``.  Subsequent SQL operations append to the buffer instead
of reporting immediately.  At ``COMMIT``, the buffer is flushed to the
reporter.  At ``ROLLBACK``, the buffer is discarded.  ``SAVEPOINT`` and
``ROLLBACK TO`` manage partial rollback via buffer truncation.

During a transaction, the DPOR scheduler skips scheduling (does not
yield to other threads), ensuring all SQL operations within the
transaction appear atomic.  This prevents false positives from
intermediate transaction states that would be rolled back on failure.


Concrete Example: SELECT-then-UPDATE Lost Update
-------------------------------------------------

Two threads each increment a counter stored in a database row (see
:doc:`orm_race` for user-facing demos of this scenario).

.. code-block:: python

   # Thread A and Thread B both execute this:
   def handle_login(engine):
       with Session(engine) as session:
           user = session.get(User, 1)                     # SELECT * FROM users WHERE id = 1
           user.login_count = user.login_count + 1          # Python-side increment
           session.commit()                                 # UPDATE users SET login_count = <n> WHERE id = 1; COMMIT

The buggy interleaving:

.. code-block:: text

   Thread A: BEGIN
   Thread A: SELECT * FROM users WHERE id = 1     -> login_count = 0
   Thread B: BEGIN
   Thread B: SELECT * FROM users WHERE id = 1     -> login_count = 0
   Thread A: UPDATE users SET login_count = 1 WHERE id = 1
   Thread A: COMMIT                                -- login_count is now 1
   Thread B: UPDATE users SET login_count = 1 WHERE id = 1
   Thread B: COMMIT                                -- login_count is still 1, not 2!

Following the six-stage pipeline, both SELECTs resolve ``%(pk)s`` to
``1`` and report ``("sql:users:(('id', '1'),)", "read")``.  Read-Read
on the same ObjectId is independent -- no conflict, which is correct.
Thread A's ``session.commit()`` triggers an UPDATE that reports
``("sql:users:(('id', '1'),)", "write")``, creating a Read-Write
conflict with Thread B's prior read.  DPOR explores the alternative
ordering.

The fix -- ``SELECT FOR UPDATE``:

.. code-block:: python

   def handle_login_safe(engine):
       with Session(engine) as session:
           user = session.execute(
               select(User).where(User.id == 1).with_for_update()
           ).scalar_one()                                   # SELECT ... FOR UPDATE
           user.login_count = user.login_count + 1
           session.commit()

.. code-block:: text

   Thread A: BEGIN
   Thread A: SELECT * FROM users WHERE id = 1 FOR UPDATE   -> login_count = 0 (row locked)
   Thread B: BEGIN
   Thread B: SELECT * FROM users WHERE id = 1 FOR UPDATE   -> BLOCKED (waits for row lock)
   Thread A: UPDATE users SET login_count = 1 WHERE id = 1
   Thread A: COMMIT                                         -- login_count = 1, lock released
   Thread B: (unblocked) login_count = 1                    -> reads current value
   Thread B: UPDATE users SET login_count = 2 WHERE id = 1
   Thread B: COMMIT                                         -- login_count = 2, correct!

When ``parse_sql_access()`` sees ``FOR UPDATE``, it returns
``lock_intent="UPDATE"``, so the SELECT is reported as a write:
``("sql:users:(('id', '1'),)", "write")``.  Both threads' initial
statements are now writes on the same ObjectId -- a Write-Write
conflict -- so DPOR explores both orderings.


Anomaly Classification
----------------------

``_sql_anomaly.py`` builds a Dependency Serialization Graph (DSG) from
the SQL trace events and classifies cycles by their edge types:

- Lost update -- two threads both read then write the same resource
- Write skew -- RW-only cycle across different tables
- Dirty read -- WR edge (read of uncommitted data)
- Non-repeatable read -- same resource read twice with intervening write
- Phantom read -- table read twice with intervening INSERT/DELETE
- Write-write -- WW cycle (concurrent uncoordinated writes)


PostgreSQL Isolation Levels
---------------------------

Frontrun does not model PostgreSQL isolation levels (READ COMMITTED,
REPEATABLE READ, SERIALIZABLE).  The DPOR engine controls thread
scheduling and observes shared-resource accesses; it does not simulate
the database's MVCC or SSI internals.  Isolation semantics are enforced
by PostgreSQL itself during each execution.

Because the invariant checks real database state after real SQL
execution, any anomaly frontrun reports actually happened under the
configured isolation level.  The anomaly classifier labels the observed
pattern (lost update, write skew, etc.) in the actual failing
interleaving -- it does not predict what is theoretically possible under
a different level.

To test behavior at a different isolation level, change the database
configuration and re-run.


Formal Verification with TLA+
------------------------------

Three TLA+ specifications in ``specs/`` verify correctness.  All pass
exhaustive model checking via TLC with no invariant violations.

Spec 1: SqlConflictRefinement (3,200 states, 13 invariants)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checks all combinations of 2 operations x 4 statement types x 2 tables
x 2 PK values x {parsed, unparsed} x {params_resolved, unresolved}.

Thirteen invariants cover: the refinement chain (table refines endpoint,
row refines table, transitivity); independence guarantees (read-read,
different tables, different rows); conflict guarantees (same-row write,
same-table write); suppression correctness; conflict symmetry; and
parameter resolution safety (unresolved params fall back conservatively,
resolved params enable row-level precision).

Spec 2: DporSqlScheduling (2,048 states, 7 invariants)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models a database as a map from ``(table, pk) -> value``, applies every
pair of operations in both orders (A;B vs B;A), and checks that when
the detector says "no conflict", both orders produce the same database
state.  Seven invariants verify soundness at both table and row level,
the refinement chain, and parameter resolution safety.

The ``NoMissedBugs`` invariants are the strongest: for any two
operations where execution order matters (different DB states), the
detector reports a conflict, guaranteeing DPOR explores the
interleaving.

Spec 3: SuppressionSafety (32,768 states, 3 invariants)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-deterministically generates arbitrary parser outputs (including
incorrect ones) and verifies that: a correct parser detects ground-truth
conflicts; parse failure prevents suppression so endpoint-level always
conflicts; and empty parses (BEGIN/COMMIT) do not suppress.

Running the specs
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd specs/
   curl -sL -o tla2tools.jar \
     https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

   java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC SqlConflictRefinement -deadlock -nowarning
   java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC DporSqlScheduling -deadlock -nowarning
   java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC SuppressionSafety -deadlock -nowarning


Why Not z3/SMT for Row-Level Conflicts
---------------------------------------

ORM-generated SQL is overwhelmingly equality lookups (``WHERE id = ?``)
and IN-lists (``WHERE id IN (?, ?, ?)``), covering ~95% of real-world
row-level queries.  ``frozenset.isdisjoint()`` handles these in
nanoseconds.  Unhandled predicates (ranges, OR, BETWEEN, subqueries)
fall back to table-level -- conservative but sound.

z3 was considered and rejected for several reasons:

- ``_intercept_execute`` runs on every SQL statement; z3 adds ~50 ms
  per satisfiability check, which would dominate execution time at scale.
- ``z3-solver`` is ~200 MB, disproportionate for a testing library.
- Encoding SQL types into z3 sorts is error-prone, especially NULL
  three-valued logic (``NULL = NULL`` is ``NULL``, not ``TRUE``).

If range predicate support becomes important, lightweight
interval-arithmetic (no z3) could handle numeric bounds with simple
comparison.

The disjointness check is O(k) where k is the number of PK columns:

.. code-block:: python

   def pk_predicates_disjoint(preds_a, preds_b) -> bool:
       a_map = {p.column: values(p) for p in preds_a}
       b_map = {p.column: values(p) for p in preds_b}
       for col in a_map:
           if col in b_map and a_map[col].isdisjoint(b_map[col]):
               return True
       return False


Wire Protocol Parsing (C-Level Drivers)
---------------------------------------

For C-extension drivers like psycopg2/libpq that call ``send()``
directly (bypassing Python's DBAPI layer), the LD_PRELOAD library
includes a Rust-based PostgreSQL wire protocol parser
(``crates/io/src/sql_extract.rs``).

It recognizes two PostgreSQL message types:

- Simple Query (``'Q'``) -- message type byte + i32 length +
  null-terminated SQL string
- Extended Query / Parse (``'P'``) -- message type byte + i32 length +
  statement name + query string + parameter info

When the ``send()`` hook intercepts a buffer matching these patterns,
it extracts the SQL text and writes an enriched event to the pipe.
The Python-side bridge can then parse the SQL and report at table level,
even for C-extension drivers that never go through ``cursor.execute()``.


Indexical INSERT Resource IDs
------------------------------

Autoincrement columns (``SERIAL``, ``IDENTITY``, ``AUTOINCREMENT``)
assign IDs based on execution order.  When DPOR explores different
thread interleavings, the same INSERT produces different concrete IDs.

Frontrun solves this with **indexical resource IDs**: after each INSERT
executes, ``cursor.lastrowid`` is captured and mapped to a logical alias
like ``sql:users:t0_ins0`` ("thread 0's first INSERT into users").
Downstream operations referencing the captured ID are automatically
translated to the same alias, giving DPOR stable conflict detection
across interleavings.

Each INSERT also reports a write to a shared **sequence resource**
``sql:<table>:seq``, ensuring DPOR explores both orderings of concurrent
INSERTs to the same table.

**Fallback:** When ``lastrowid`` capture fails (e.g. psycopg2 without a
``RETURNING`` clause), ``explore()`` and ``explore_random()`` raise
``NondeterministicSQLError`` by default.  Suppress this with
``warn_nondeterministic_sql=False`` if you understand the implications,
or add a ``RETURNING`` clause to your INSERTs.


References
----------

- Adya, Liskov, O'Neil.
  `Generalized Isolation Level Definitions <http://pmg.csail.mit.edu/papers/icde00.pdf>`_.
  ICDE 2000.
- Berenson et al.
  `A Critique of ANSI SQL Isolation Levels <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf>`_.
  SIGMOD 1995.
- Kingsbury, Alvaro.
  `Elle <https://github.com/jepsen-io/elle>`_.
  VLDB 2021.
- Cui et al.
  `IsoRel <https://dl.acm.org/doi/10.1145/3728953>`_.
  ACM 2025.
- Jiang et al.
  `TxCheck <https://www.usenix.org/system/files/osdi23-jiang.pdf>`_.
  OSDI 2023.
- Mao.
  `sqlglot <https://github.com/tobymao/sqlglot>`_.
