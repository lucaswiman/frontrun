Examples
========

Practical examples of using Frontrun to test concurrent code.


Bank Account Transfer (Lost Update)
-------------------------------------

Two concurrent transfers each read the balance before either writes.
The second write overwrites the first, losing one transfer:

.. code-block:: python

   from frontrun.common import Schedule, Step
   from frontrun.trace_markers import TraceExecutor

   class BankAccount:
       def __init__(self, balance=0):
           self.balance = balance

       def transfer(self, amount):
           current = self.balance  # frontrun: read_balance
           new_balance = current + amount
           self.balance = new_balance  # frontrun: write_balance
           return new_balance

   account = BankAccount(balance=100)

   # Both threads read before either writes
   schedule = Schedule([
       Step("transfer1", "read_balance"),    # T1 reads 100
       Step("transfer2", "read_balance"),    # T2 reads 100
       Step("transfer1", "write_balance"),   # T1 writes 150
       Step("transfer2", "write_balance"),   # T2 writes 150
   ])

   executor = TraceExecutor(schedule)
   executor.run({
       "transfer1": lambda: account.transfer(50),
       "transfer2": lambda: account.transfer(50),
   }, timeout=5.0)

   assert account.balance == 150  # should be 200; one transfer lost


Correct Synchronization with Locks
------------------------------------

The same code with a lock eliminates the race. Frontrun's trace markers
still control the scheduling, but the lock serializes the critical section
regardless of which thread arrives first:

.. code-block:: python

   import threading
   from frontrun.common import Schedule, Step
   from frontrun.trace_markers import TraceExecutor

   class SyncedBankAccount:
       def __init__(self, balance=0):
           self.balance = balance
           self.lock = threading.Lock()

       def transfer(self, amount):
           with self.lock:
               current = self.balance  # frontrun: read_balance
               new_balance = current + amount
               self.balance = new_balance  # frontrun: write_balance
           return new_balance

   account = SyncedBankAccount(balance=100)

   schedule = Schedule([
       Step("transfer1", "read_balance"),
       Step("transfer2", "read_balance"),
       Step("transfer1", "write_balance"),
       Step("transfer2", "write_balance"),
   ])

   executor = TraceExecutor(schedule)
   executor.run({
       "transfer1": lambda: account.transfer(50),
       "transfer2": lambda: account.transfer(50),
   }, timeout=5.0)

   assert account.balance == 200  # both transfers applied correctly


Automatic Race Finding with DPOR
----------------------------------

DPOR systematically explores all meaningfully different interleavings.
No markers needed --- it detects shared-memory conflicts automatically:

.. code-block:: python

   from frontrun import explore

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value
           self.value = temp + 1

   result = explore(
       setup=Counter,
       workers=[lambda c: c.increment(), lambda c: c.increment()],
       invariant=lambda c: c.value == 2,
   )

   assert result.property_holds, result.explanation

Output when the race is found:

.. code-block:: text

   Race condition found after 2 interleavings.

     Write-write conflict: threads 0 and 1 both wrote to value.

     Thread 0 | counter.py:7             temp = self.value
              | [read Counter.value]
     Thread 0 | counter.py:8             self.value = temp + 1
              | [write Counter.value]
     Thread 1 | counter.py:7             temp = self.value
              | [read Counter.value]
     Thread 1 | counter.py:8             self.value = temp + 1
              | [write Counter.value]

     Reproduced 10/10 times (100%)


Automatic Race Finding with Bytecode Exploration
--------------------------------------------------

Bytecode exploration generates random opcode-level schedules. It often
finds races very quickly and can catch races invisible to DPOR (e.g. shared
state in C extensions), but the error traces are less interpretable:

.. code-block:: python

   from frontrun import explore_random

   class Counter:
       def __init__(self, value=0):
           self.value = value

       def increment(self):
           temp = self.value
           self.value = temp + 1

   result = explore_random(
       setup=lambda: Counter(value=0),
       threads=[
           lambda c: c.increment(),
           lambda c: c.increment(),
       ],
       invariant=lambda c: c.value == 2,
       max_attempts=200,
       max_ops=200,
       seed=42,
   )

   assert result.property_holds, result.explanation

Output:

.. code-block:: text

   Race condition found after 1 interleavings.

     Lost update: threads 0 and 1 both read value before either wrote it back.

     Thread 1 | counter.py:7             temp = self.value
              | [read value]
     Thread 0 | counter.py:7             temp = self.value
              | [read value]
     Thread 1 | counter.py:8             self.value = temp + 1
              | [write value]
     Thread 0 | counter.py:8             self.value = temp + 1
              | [write value]

     Reproduced 10/10 times (100%)


Async Concurrency Control
---------------------------

Async trace markers let you control interleaving at ``await`` boundaries:

.. code-block:: python

   from frontrun import TraceExecutor
   from frontrun.common import Schedule, Step

   class AsyncBankAccount:
       def __init__(self, balance=0):
           self.balance = balance

       async def get_balance(self):
           return self.balance

       async def set_balance(self, value):
           self.balance = value

       async def transfer(self, amount):
           # frontrun: read_balance
           current = await self.get_balance()
           new_balance = current + amount
           # frontrun: write_balance
           await self.set_balance(new_balance)

   account = AsyncBankAccount(balance=100)

   schedule = Schedule([
       Step("task1", "read_balance"),
       Step("task2", "read_balance"),
       Step("task1", "write_balance"),
       Step("task2", "write_balance"),
   ])

   executor = TraceExecutor(schedule)
   executor.run({
       "task1": lambda: account.transfer(50),
       "task2": lambda: account.transfer(50),
   })

   assert account.balance == 150  # one transfer lost


Interactive HTML Exploration Reports
--------------------------------------

``explore()`` can write a self-contained interactive HTML report that
lets you step through every explored execution, inspect thread switch-points,
and see the conflicting attribute accesses that caused each reordering.

**Generating a report from pytest** --- pass ``--frontrun-report PATH`` to the
``frontrun`` test runner:

.. code-block:: bash

   frontrun pytest tests/ --frontrun-report dpor_report.html

**Generating a report from a script** --- set ``_global_report_path`` before
calling ``explore()``:

.. code-block:: python

   import frontrun._report
   from frontrun import explore

   class Accounts:
       def __init__(self) -> None:
           self.a = 100
           self.b = 100
           self.c = 100

   def transfer_a_to_b(accounts: Accounts) -> None:
       if accounts.a >= 60:
           accounts.a -= 60
           accounts.b += 60

   def transfer_b_to_c(accounts: Accounts) -> None:
       if accounts.b >= 80:
           accounts.b -= 80
           accounts.c += 80

   frontrun._report._global_report_path = "dpor_report.html"
   try:
       result = explore(
           setup=Accounts,
           workers=[transfer_a_to_b, transfer_b_to_c],
           invariant=lambda accs: accs.a + accs.b + accs.c == 300,
           preemption_bound=2,
       )
       print(result.explanation)
   finally:
       frontrun._report._global_report_path = None

The race here is a classic **lost update on account B**: both threads
read-modify-write ``accounts.b`` without any lock.  When one thread's write
overwrites the other's, the total balance drifts away from 300.

The report shows every explored interleaving as a timeline.  Executions where
the invariant holds are shown in green; failing ones in red.  Click any
execution button or use the arrow keys to step through them.  Each switch-point
panel shows the source line and opcode where the scheduler switched threads,
making it easy to pinpoint exactly which access caused the conflict.

**Example reports** (generated at documentation build time):

- `Bank transfer — racy <_static/dpor_bank_transfer.html>`_:
  Both threads share account B without a lock.
- `Bank transfer — locked <_static/dpor_bank_transfer_locked.html>`_:
  A single ``threading.Lock`` makes each transfer atomic; DPOR verifies safety with far fewer paths.
- `SQLite counter — racy <_static/dpor_sqlite_counter.html>`_:
  Two threads each read-modify-write a SQLite counter; DPOR detects the SQL-level conflict.
- `SQLite counter — fixed <_static/dpor_sqlite_counter_fixed.html>`_:
  A single ``UPDATE counter SET value = value + 1`` eliminates the race.
- `Dining philosophers (3) <_static/dpor_dining_philosophers.html>`_:
  Three philosophers always grab the left fork first, creating a circular wait.

Run any example directly to regenerate its report::

    python examples/dpor_bank_transfer.py my_report.html
    python examples/dpor_bank_transfer_locked.py my_report.html
    python examples/dpor_sqlite_counter.py my_report.html
    python examples/dpor_sqlite_counter.py my_report.html fixed
    python examples/dpor_dining_philosophers.py my_report.html


Locking and path reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The locked bank transfer illustrates an important property of DPOR: when
operations are protected by a lock, DPOR explores far fewer interleavings
because the only meaningful ordering question is *which thread acquires the
lock first*.  Compare the racy report against the locked report to see this
directly.

The ``stop_on_first=False`` parameter (used in all the examples above) tells
DPOR to continue exploring after the first failure.  The default
``stop_on_first=True`` stops as soon as a violation is found, which is usually
what you want in a test suite.


Real-World Case Study: SQLAlchemy ORM
---------------------------------------

For a walkthrough of a real lost-update race in SQLAlchemy ORM code running
against PostgreSQL, see :doc:`orm_race`. That case study demonstrates detection
with trace markers, bytecode exploration, and DPOR with C-level I/O
interception via ``LD_PRELOAD``.
