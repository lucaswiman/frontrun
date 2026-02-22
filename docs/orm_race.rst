SQLAlchemy Lost-Update Race Condition
=====================================

This walkthrough demonstrates frontrun detecting a real race condition
in SQLAlchemy ORM code running against Postgres.  The full source is in
``examples/orm_race.py``.

The bug
-------

Any ORM that computes new column values in Python is vulnerable to
lost updates.  Two concurrent request handlers both read the same row,
compute a new value from what they read, and write it back.  When their
transactions overlap, the second commit silently overwrites the first:

.. code-block:: text

   Handler A:  SELECT → login_count = 0
   Handler B:  SELECT → login_count = 0
   Handler A:  UPDATE SET login_count = 1, COMMIT   ← correct
   Handler B:  UPDATE SET login_count = 1, COMMIT   ← stale! should be 2

The fix is to push the arithmetic into SQL (``SET login_count =
login_count + 1``) or to serialize access with ``SELECT … FOR UPDATE``.

The model and the handler
-------------------------

A minimal SQLAlchemy model and the buggy handler that increments
``login_count`` in Python:

.. code-block:: python

   from sqlalchemy import String, create_engine
   from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

   class Base(DeclarativeBase):
       pass

   class User(Base):
       __tablename__ = "users"
       id: Mapped[int] = mapped_column(primary_key=True)
       name: Mapped[str] = mapped_column(String(100))
       login_count: Mapped[int] = mapped_column(default=0)

   engine = create_engine("postgresql:///frontrun_test")

The handler that every demo runs — one per thread, each with its own
session:

.. code-block:: python

   def handle_login(engine):
       with Session(engine) as session:
           user = session.get(User, 1)          # SELECT
           user.login_count = user.login_count + 1  # Python-side increment
           session.commit()                      # UPDATE … SET login_count=<n>

Two concurrent calls should leave ``login_count == 2``.  The race makes
it ``1``.


Demo 1 — Trace markers (deterministic)
---------------------------------------

Trace markers (``# frontrun: orm_read`` / ``# frontrun: orm_write``)
on the ``session.get()`` and ``session.commit()`` lines let
``TraceExecutor`` force both SELECTs to run before either COMMIT:

.. code-block:: python

   from frontrun.common import Schedule, Step
   from frontrun.trace_markers import TraceExecutor

   def handler_a():
       with Session(engine) as session:
           user = session.get(User, 1)            # frontrun: orm_read
           user.login_count = user.login_count + 1
           session.commit()                        # frontrun: orm_write

   schedule = Schedule([
       Step("a", "orm_read"),   # Handler A SELECTs (login_count=0)
       Step("b", "orm_read"),   # Handler B SELECTs (login_count=0, stale)
       Step("a", "orm_write"),  # Handler A COMMITs (login_count=1)
       Step("b", "orm_write"),  # Handler B COMMITs (login_count=1, should be 2!)
   ])

   executor = TraceExecutor(schedule)
   executor.run("a", handler_a)
   executor.run("b", handler_b)
   executor.wait(timeout=10.0)

Output:

.. code-block:: text

   ======================================================================
   Demo 1: SQLAlchemy lost update  (TraceExecutor — deterministic)
   ======================================================================

     Scenario:
       Two handlers each read login_count and increment it in Python.
       Expected final login_count: 2

     Final login_count: 1  (expected 2)

     LOST UPDATE confirmed: one increment was silently lost.
     Handler B's commit wrote login_count=1 based on a stale
     read, overwriting handler A's increment.

     Reproducibility: 100% — the Schedule deterministically forces
     both SELECTs to run before either UPDATE on every execution.


Demo 2 — Bytecode exploration (automatic)
-----------------------------------------

``explore_interleavings`` generates random opcode-level schedules,
running both handlers against the real database on each attempt.
No trace markers are needed — the explorer finds the bad interleaving
on its own:

.. code-block:: python

   from frontrun.bytecode import explore_interleavings

   class _State:
       def __init__(self):
           with Session(engine) as session:
               user = session.get(User, 1)
               user.login_count = 0
               session.commit()

   def _thread_fn(_state):
       with Session(engine) as session:
           user = session.get(User, 1)
           user.login_count = user.login_count + 1
           session.commit()

   result = explore_interleavings(
       setup=_State,
       threads=[_thread_fn, _thread_fn],
       invariant=lambda s: _read_count() == 2,
       max_attempts=50,
       seed=42,
       detect_io=True,    # C-level sockets detected via LD_PRELOAD
   )

Output:

.. code-block:: text

   ======================================================================
   Demo 2: SQLAlchemy lost update  (bytecode exploration — automatic)
   ======================================================================

     Generating random opcode-level schedules and running both
     handlers against real Postgres.  Checking whether both
     increments persist after each interleaving.

     property_holds    : False
     attempts_explored : 1
     counterexample found after 1 attempt(s)  (5/5 reproductions)

     Race condition found after 1 interleavings.

       Lost update: threads 0 and 1 both read login_count before either wrote it back.


       Thread 1 | orm_race.py:238           user = session.get(User, 1)  [read .get]
       Thread 0 | orm_race.py:238           user = session.get(User, 1)  [read .get]
       Thread 0 | orm_race.py:240           user.login_count = user.login_count + 1  [read .login_count]
       Thread 1 | orm_race.py:240           user.login_count = user.login_count + 1  [read .login_count]
       Thread 0 | orm_race.py:240           user.login_count = user.login_count + 1  [write .login_count]
       Thread 1 | orm_race.py:240           user.login_count = user.login_count + 1  [write .login_count]
       Thread 1 | orm_race.py:241           session.commit()  [read .commit]
       Thread 0 | orm_race.py:241           session.commit()  [read .commit]

       Reproduced 5/5 times (100%)

     LOST UPDATE confirmed via bytecode exploration.

The explorer found the race on its very first attempt and reproduced it
5 out of 5 times.  The trace shows the exact interleaving: both threads
read ``login_count`` (via ``session.get``) before either writes, so
both compute ``0 + 1 = 1``.


Demo 3 — Naive threading (intermittent)
----------------------------------------

Plain threads against real Postgres, with a random 0–15 ms start offset
modelling realistic request-arrival timing.  The race reproduces only a
fraction of the time — exactly the kind of flaky bug that slips through
CI:

.. code-block:: text

   ======================================================================
   Demo 3: Naive threading + SQLAlchemy  (500 trials)
   ======================================================================

     Running both handlers in plain threads against real Postgres.
     Threads start with a random 0-15ms offset to model realistic
     request arrival timing.  Counting how often the race manifests...

     Trials:   500
     Failures: 23
     Rate:     4.6%

     The race manifested in 23/500 trials (4.6%).
     frontrun makes it 100% reproducible.

With ordinary threads the bug surfaces roughly 5–10 % of the time
(the exact rate varies with system load).  The two frontrun approaches
above both catch it deterministically, every time.


Running the example yourself
----------------------------

.. code-block:: bash

   # Build the virtualenv with SQLAlchemy + psycopg2, and the I/O library
   make build-examples-3.14t build-io   # or build-examples-3.10

   # Create the test database (if it doesn't exist)
   createdb frontrun_test

   # Run via the frontrun CLI for C-level I/O interception
   frontrun .venv-3.14t/bin/python examples/orm_race.py
