Examples
========

This section contains practical examples of using Frontrun to test concurrent code.


Example 1: Bank Account Transfer
---------------------------------

A classic example showing how a race condition can cause lost updates:

.. code-block:: python

   from frontrun.trace_markers import Schedule, Step, TraceExecutor

   class BankAccount:
       def __init__(self, balance=0):
           self.balance = balance

       def transfer(self, amount):
           """Transfer funds (has a race condition)."""
           # frontrun: after_read
           current = self.balance

           # Simulate processing time
           new_balance = current + amount

           # frontrun: before_write
           self.balance = new_balance
           return new_balance

   # Create an account with $100
   account = BankAccount(balance=100)

   # Define a schedule that triggers the race condition
   # Both threads read before either writes
   schedule = Schedule([
       Step("transfer1", "after_read"),    # T1 reads 100
       Step("transfer2", "after_read"),    # T2 reads 100
       Step("transfer1", "before_write"),  # T1 writes 150
       Step("transfer2", "before_write"),  # T2 writes 150
   ])

   # Execute with controlled interleaving
   executor = TraceExecutor(schedule)
   executor.run("transfer1", lambda: account.transfer(50))
   executor.run("transfer2", lambda: account.transfer(50))
   executor.wait(timeout=5.0)

   # Race condition detected: both deposits were counted as 150
   # instead of 200 (100 + 50 + 50)
   assert account.balance == 150, f"Expected 150, got {account.balance}"
   print("Race condition: one $50 transfer was lost!")


Example 2: Correct Synchronization with Locks
-----------------------------------------------

Comparing the same code with proper synchronization:

.. code-block:: python

   import threading
   from frontrun.trace_markers import Schedule, Step, TraceExecutor

   class SyncedBankAccount:
       def __init__(self, balance=0):
           self.balance = balance
           self.lock = threading.Lock()

       def transfer(self, amount):
           """Transfer funds with proper synchronization."""
           with self.lock:
               # frontrun: after_read
               current = self.balance
               new_balance = current + amount
               # frontrun: before_write
               self.balance = new_balance
           return new_balance

   # Create an account with $100
   account = SyncedBankAccount(balance=100)

   # Same schedule as before
   schedule = Schedule([
       Step("transfer1", "after_read"),
       Step("transfer2", "after_read"),
       Step("transfer1", "before_write"),
       Step("transfer2", "before_write"),
   ])

   # Execute with controlled interleaving
   executor = TraceExecutor(schedule)
   executor.run("transfer1", lambda: account.transfer(50))
   executor.run("transfer2", lambda: account.transfer(50))
   executor.wait(timeout=5.0)

   # With proper locking, the balance is correct
   assert account.balance == 200, f"Expected 200, got {account.balance}"
   print("Synchronization works: both transfers applied correctly!")


Example 3: Producer-Consumer Pattern
-------------------------------------

Testing a simple producer-consumer queue:

.. code-block:: python

   from frontrun.trace_markers import Schedule, Step, TraceExecutor

   class SimpleQueue:
       def __init__(self):
           self.items = []

       def put(self, item):
           """Add item to queue."""
           # frontrun: before_append
           self.items.append(item)
           # frontrun: after_append

       def get(self):
           """Remove and return first item."""
           # frontrun: before_check
           if not self.items:
               return None
           # frontrun: after_check
           # frontrun: before_pop
           item = self.items.pop(0)
           # frontrun: after_pop
           return item

   queue = SimpleQueue()

   # Producer puts, consumer gets
   schedule = Schedule([
       Step("producer", "before_append"),
       Step("consumer", "before_check"),
       Step("consumer", "after_check"),
       Step("producer", "after_append"),
       Step("consumer", "before_pop"),
       Step("consumer", "after_pop"),
   ])

   executor = TraceExecutor(schedule)
   executor.run("producer", lambda: queue.put("data"))
   executor.run("consumer", lambda: queue.get())
   executor.wait(timeout=5.0)

   print("Producer-consumer executed with controlled interleaving")


Example 4: Async Concurrency Control
-------------------------------------

Testing race conditions in async code using async trace markers:

.. code-block:: python

   import asyncio
   from frontrun.async_trace_markers import AsyncTraceExecutor
   from frontrun.common import Schedule, Step

   class AsyncBankAccount:
       def __init__(self, balance=0):
           self.balance = balance

       async def transfer(self, amount):
           """Transfer funds (has a race condition at await points)."""
           # frontrun: after_read
           current = self.balance
           await asyncio.sleep(0)  # Yield point for marker

           new_balance = current + amount

           # frontrun: before_write
           await asyncio.sleep(0)  # Yield point for marker
           self.balance = new_balance

   account = AsyncBankAccount(balance=100)

   # Define a schedule that triggers the race condition
   schedule = Schedule([
       Step("task1", "after_read"),    # Task1 reads 100
       Step("task2", "after_read"),    # Task2 reads 100
       Step("task1", "before_write"),  # Task1 writes 150
       Step("task2", "before_write"),  # Task2 writes 150
   ])

   executor = AsyncTraceExecutor(schedule)
   executor.run({
       "task1": lambda: account.transfer(50),
       "task2": lambda: account.transfer(50),
   })

   # Race condition: one transfer was lost
   assert account.balance == 150, f"Expected 150, got {account.balance}"
   print("Async race condition detected!")


Example 5: Finding Race Conditions Automatically (Experimental)
---------------------------------------------------------------

Using bytecode instrumentation to automatically explore interleavings:

.. warning::

   Bytecode instrumentation is experimental. Use with caution.

.. code-block:: python

   from frontrun.bytecode import explore_interleavings

   class Counter:
       def __init__(self, value=0):
           self.value = value

       def increment(self):
           temp = self.value
           self.value = temp + 1

   # Automatically explore different interleavings
   result = explore_interleavings(
       setup=lambda: Counter(value=0),
       threads=[
           lambda c: c.increment(),
           lambda c: c.increment(),
       ],
       invariant=lambda c: c.value == 2,  # Should hold if no races
       max_attempts=200,
       max_ops=200,
       seed=42,
   )

   if not result.property_holds:
       print(f"Race condition found!")
       print(f"Explored {result.num_explored} different interleavings")
       print(f"Counterexample: value = {result.counterexample.value}")
   else:
       print("No race conditions found in explored interleavings")
