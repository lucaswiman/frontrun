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
   executor.run("transfer1", lambda: account.transfer(50))
   executor.run("transfer2", lambda: account.transfer(50))
   executor.wait(timeout=5.0)

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
   executor.run("transfer1", lambda: account.transfer(50))
   executor.run("transfer2", lambda: account.transfer(50))
   executor.wait(timeout=5.0)

   assert account.balance == 200  # both transfers applied correctly


Automatic Race Finding with DPOR
----------------------------------

DPOR systematically explores all meaningfully different interleavings.
No markers needed --- it detects shared-memory conflicts automatically:

.. code-block:: python

   from frontrun.dpor import explore_dpor

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value
           self.value = temp + 1

   result = explore_dpor(
       setup=Counter,
       threads=[lambda c: c.increment(), lambda c: c.increment()],
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

   from frontrun.bytecode import explore_interleavings

   class Counter:
       def __init__(self, value=0):
           self.value = value

       def increment(self):
           temp = self.value
           self.value = temp + 1

   result = explore_interleavings(
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

   from frontrun.async_trace_markers import AsyncTraceExecutor
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

   executor = AsyncTraceExecutor(schedule)
   executor.run({
       "task1": lambda: account.transfer(50),
       "task2": lambda: account.transfer(50),
   })

   assert account.balance == 150  # one transfer lost


Real-World Case Study: SQLAlchemy ORM
---------------------------------------

For a walkthrough of a real lost-update race in SQLAlchemy ORM code running
against PostgreSQL, see :doc:`orm_race`. That case study demonstrates detection
with trace markers and bytecode exploration, and discusses why DPOR cannot
detect this particular race (the shared state lives in the database, not in
Python memory).
