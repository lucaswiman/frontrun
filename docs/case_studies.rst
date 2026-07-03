================================================================================
Case Studies: Races in Real, Unmodified Libraries
================================================================================

The examples elsewhere in these docs use small, self-contained snippets so the
mechanics are easy to follow. This page is different: every race below was found
by pointing ``frontrun.explore()`` at a **real, released, unmodified** PyPI
package and letting DPOR schedule its threads. No forks, no injected ``sleep()``
calls, no rewritten internals — the code under test is exactly what ``pip
install`` gives you.

These are not exotic bugs. They are ordinary *check-then-act* and
*read-modify-write* patterns — the kind that are easy to write and genuinely
hard to see in review, because the CPython GIL keeps the racy window narrow
enough that they almost always pass under normal testing. frontrun's job is to
make that window deterministic: it either produces the exact interleaving that
breaks an invariant, replayable every time, or it proves the invariant holds
across every meaningfully different schedule. The point of this page is that the
same workflow that finds the lost update in the two-line counter on the front
page works, without modification, on code you already depend on.

Both write-ups include the complete test and the trace frontrun prints, verified
against ``python-http-client`` 3.3.7 and ``python-socketio`` 5.16.3. To reproduce
one yourself, put the test in a file and run it through the ``frontrun`` CLI
(plain ``pytest`` skips ``frontrun.explore()`` tests — see :doc:`installation`):

.. code-block:: bash

   pip install frontrun python-http-client python-socketio
   frontrun pytest test_case_studies.py

.. contents::
   :local:
   :depth: 1


sendgrid: a reused API client corrupts one request's headers with another's
===========================================================================

**Library:** `python_http_client <https://github.com/sendgrid/python-http-client>`_,
the HTTP layer under the official `sendgrid <https://github.com/sendgrid/sendgrid-python>`_
client (~10M downloads/month).

**Category:** correctness / data-integrity.

sendgrid's own documentation builds one ``SendGridAPIClient`` and reuses it for
every request — the module-level singleton is the intended pattern. That client
wraps a ``python_http_client.Client`` which holds a single mutable
``request_headers`` dict, and two methods mutate it with no synchronization.
``_build_client()`` (``client.py:157``) hands the *same* dict object to every
URL-chained child client rather than copying it, and ``_update_headers()`` writes
per-call headers straight into it (``client.py:145``):

.. code-block:: python

   def _update_headers(self, request_headers):
       self.request_headers.update(request_headers)   # mutates the shared dict

So when a threaded server (Flask/Django/FastAPI worker threads) reuses one
client, every concurrent request is calling ``update()`` on one shared dict. If
two requests carry their own per-call headers — an ``Authorization`` override,
an ``On-Behalf-Of`` subuser, a custom ``X-`` header — one thread can build its
outgoing HTTP request with the *other* thread's headers.

The test reproduces exactly that: one shared ``Client``, two workers setting
their own ``X-Thread-Id`` through the real ``_update_headers`` and reading it
back the way the request-builder does. The invariant is that each worker must
see its own header. One extra argument matters here: ``request_headers`` is a
plain dict inside an installed package, which frontrun excludes from tracing by
default, so we widen the filter with ``trace_packages=["python_http_client*"]``
to let DPOR see the conflicting ``dict.update`` accesses inside the library:

.. code-block:: python

   import frontrun
   import python_http_client

   class State:
       def __init__(self):
           # one shared client — the documented singleton pattern
           self.client = python_http_client.Client(
               host="http://localhost:9999",
               request_headers={"Authorization": "Bearer test-key"},
               version=3,
           )
           self.seen = [None, None]

   def make_worker(i):
       def worker(s):
           s.client._update_headers({"X-Thread-Id": f"thread-{i}"})
           s.seen[i] = s.client.request_headers.get("X-Thread-Id")
       return worker

   def each_worker_sees_own_header(s):
       if s.seen[0] is None or s.seen[1] is None:
           return True
       return s.seen[0] == "thread-0" and s.seen[1] == "thread-1"

   def test_shared_request_headers_race():
       result = frontrun.explore(
           setup=State,
           workers=[make_worker(0), make_worker(1)],
           invariant=each_worker_sees_own_header,
           trace_packages=["python_http_client*"],
       )
       result.assert_holds()

DPOR points straight at the shared mutation inside the library:

.. code-block:: text

   Race condition found after 2 interleavings.

     Race condition involving threads 0, 1.

     Thread 0 | client.py:145             self.request_headers.update(request_headers)
              | [read Client.request_headers]
              | [read dict.update]
     Thread 1 | client.py:145             self.request_headers.update(request_headers)
              | [read Client.request_headers]
              | [read dict.update]
     ...
     Reproduced 10/10 times (100%)

Both threads are calling ``update()`` on the same ``request_headers`` dict; when
one thread's write lands between the other's write and its read-back, the reader
sees the wrong header. Copying the dict once — ``request_headers=dict(self.request_headers)``
in ``_build_client()`` — gives each client its own headers and closes the race.

The ``trace_packages`` argument is worth dwelling on. Without it, the shared
state lives where frontrun does not look, and the race only shows up indirectly
(through incidental conflicts in the URL-chaining code). Widening the trace
filter to the library under test is what turns a confusing, second-hand trace
into one that names the exact line — a reminder that *what you trace* is part of
asking the right question.


python-socketio: a lost room registration under ``async_mode='threading'``
==========================================================================

**Library:** `python-socketio <https://github.com/miguelgrinberg/python-socketio>`_
(~5.6M downloads/month) — a Socket.IO server/client implementation.

**Category:** correctness / data-integrity (lost update).

``BaseManager.basic_enter_room()`` lazily creates the nested dict structure that
tracks which clients are in which room (``socketio/base_manager.py:112-121``):

.. code-block:: python

   def basic_enter_room(self, sid, namespace, room, eio_sid=None):
       if eio_sid is None and namespace not in self.rooms:
           raise ValueError('sid is not connected to requested namespace')
       if namespace not in self.rooms:            # CHECK 1
           self.rooms[namespace] = {}             # ACT 1  (fresh dict)
       if room not in self.rooms[namespace]:      # CHECK 2
           self.rooms[namespace][room] = bidict() # ACT 2
       ...
       self.rooms[namespace][room][sid] = eio_sid # final write

There is no lock in ``BaseManager`` or its ``Manager`` subclass — reasonably so,
since ``Manager`` is the in-process manager. But with ``async_mode='threading'``
(a supported configuration) every event handler runs on the same ``Manager``
instance across threads. When two clients enter rooms in the same, not-yet-seen
namespace at the same time, both pass ``namespace not in self.rooms`` before
either assigns ``self.rooms[namespace] = {}``. The second assignment replaces the
first with a brand-new empty dict, discarding whatever the first thread had just
written into it.

The test drives the real ``basic_enter_room`` with two clients entering two rooms
in one namespace, and asserts the obvious invariant — after both calls, both
clients should be registered:

.. code-block:: python

   import frontrun
   from socketio.base_manager import BaseManager

   class State:
       def __init__(self):
           self.mgr = BaseManager()
           self.namespace = "/chat"

   def enter(sid, room, eio_sid):
       def worker(s):
           s.mgr.basic_enter_room(sid, s.namespace, room, eio_sid=eio_sid)
       return worker

   def both_clients_registered(s):
       ns = s.mgr.rooms.get(s.namespace, {})
       return ("room_a" in ns and "sid_a" in ns["room_a"]
               and "room_b" in ns and "sid_b" in ns["room_b"])

   def test_concurrent_enter_room():
       result = frontrun.explore(
           setup=State,
           workers=[enter("sid_a", "room_a", "eio_a"),
                    enter("sid_b", "room_b", "eio_b")],
           invariant=both_clients_registered,
           trace_packages=["socketio*"],
       )
       result.assert_holds()

As with sendgrid, the shared state lives inside an installed package, so
``trace_packages=["socketio*"]`` is required — without it frontrun never looks
inside the library, the workers run without interleaving there, and the test
passes without testing anything.

DPOR reports the write-write conflict on the shared namespace dict:

.. code-block:: text

   Race condition found after 2 interleavings.

     Write-write conflict: threads 0 and 1 both wrote to dict./chat.

     Thread 1 | base_manager.py:116       self.rooms[namespace] = {}
              | [write dict./chat]
     Thread 1 | base_manager.py:117       if room not in self.rooms[namespace]:
              | [read dict./chat]
     Thread 1 | base_manager.py:118       self.rooms[namespace][room] = bidict()
              | [read dict./chat]
     Thread 1 | base_manager.py:121       self.rooms[namespace][room][sid] = eio_sid
              | [read dict./chat]
     Thread 0 | base_manager.py:116       self.rooms[namespace] = {}
              | [write dict./chat]
     ...
     Reproduced 10/10 times (100%)

In production the effect is a client that silently never receives messages: its
room registration was overwritten, so it is absent from every subsequent
broadcast to that room. Adding a ``threading.Lock`` around the enter/leave
methods closes the window.


Reading these traces
====================

Two things are worth noticing across both examples.

First, the ``[read dict.update]`` / ``[write dict./chat]`` annotations under each
source line are the *resource-level* accesses DPOR tracked — the actual shared
memory that conflicted, keyed by object and attribute. That is what lets the
engine decide two schedules are equivalent and skip one, and it is why the
report can name the exact object the two threads fought over rather than just
"something raced."

Second, the ``Reproduced 10/10 times`` line is not a probability — it is the
result of replaying the discovered schedule ten more times to confirm it is
deterministic. A found counterexample is a constructive proof: run these
operations in this order and the invariant fails, every time. That is the
difference between a bug report you can hand to a maintainer and a flaky test
that fails once in a thousand CI runs.
