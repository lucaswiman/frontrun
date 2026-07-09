# Cross-Process Exploration: Deterministic Interleaving of Workers Contending on Shared External State

**Status:** Implemented (2026-07-04) — `frontrun.explore(execution="process")` and
`frontrun.explore_processes()`; see `docs/cross_process.rst`. Original proposal
(2026-06-12) follows; phase status is tracked in
`possible-future-roadmap/index.md`.

## Problem statement

Most production race incidents are not thread-vs-thread in one process — they are
worker-vs-worker: two gunicorn/uwsgi workers, a web request racing a celery task, a cron
job racing a deploy migration, all contending on the same Postgres/Redis. Frontrun
currently simulates this by running the competing code paths as threads in one process,
which works but (a) diverges from production topology (connection pools, transaction
scope, process-local caches), and (b) cannot exercise code that genuinely forks or is
spawned by a framework.

No Python tool does deterministic cross-process interleaving exploration. Frontrun is
unusually close to being able to.

## Key insight: across processes, only external state conflicts

Two processes share no Python memory. The only resources on which their operations can
conflict are external: SQL rows/tables, Redis keys, files, queues. Therefore a
cross-process scheduler does **not** need opcode-level control of the workers — it only
needs to interpose at external-access points, which frontrun already intercepts
in-process:

- SQL: cursor patching reports `sql:<table>[:db=<scope>][:<pred_key>]` resource IDs via
  `_report_sql_access()` (`frontrun/_sql_cursor.py`), with row-level predicates,
  `:seq` insert ordering, and transaction control ops.
- Redis: `_redis_client.py` reports key-level accesses through the same reporter.
- Anything else: the LD_PRELOAD library (`crates/io/`) reports C-level socket/file I/O
  over a pipe (`FRONTRUN_IO_FD`), already attributed per OS thread.

Code *between* external accesses is process-private and can run uncontrolled — it is
independent by construction. So the explored schedule has marker-like granularity (one
event per external access), keeping the search space small, and DPOR applies unchanged:
resource keys are the same strings, each process gets a vector clock, and the Rust engine
sees each process as a "thread" id.

## Architecture

```
┌─────────────────────────────┐
│ coordinator (test process)  │   owns: Rust DPOR engine, RowLockRegistry,
│   frontrun.explore(         │         wait-for graph, schedule trace, invariant
│       processes={...})      │
└──────────┬──────────────────┘
           │ unix socket (length-prefixed msgpack/json)
   ┌───────┴────────┬─────────────────┐
   │ worker A       │ worker B        │   each: spawned under `frontrun` CLI env,
   │ SchedulerProxy │ SchedulerProxy  │   normal frontrun SQL/Redis patches installed,
   └────────────────┘─────────────────┘   reporter routed to the proxy
```

### Worker side: a remote scheduler behind the existing interface

The SQL interception path already takes its scheduler from ambient context:
`_dpor_schedule_and_suppress_sync()` calls `_acquire_pending_row_locks()` and then
`_dpor_ctx[0].report_and_wait(None, _dpor_ctx[1])` (`frontrun/_sql_cursor.py`). That
scheduler object is the entire integration surface. Define a `SchedulerProxy` that
implements the same methods used by the access-tracking layer —

```python
class SchedulerProxy:
    def report_and_wait(self, frame, thread_id) -> bool: ...   # frame always None here
    def acquire_row_locks(self, thread_id, resource_ids) -> None: ...
    def release_row_locks(self, thread_id) -> None: ...
    # plus the io-reporter callable that batches (resource_id, kind) accesses
```

— each method sending a request over the socket and blocking until the coordinator grants
the turn. Workers never install opcode tracing; only the SQL/Redis/preload patches are
active. A worker's "step" = one external access (report accesses → wait for grant →
perform the real driver call → report completion).

### Coordinator side: existing scheduler, remote threads

The coordinator runs the existing DPOR loop with one substitution: instead of in-process
threads blocking on a condition variable in `report_and_wait`, remote workers block on a
socket read. `_schedule_next()` / `engine.schedule(execution)`, access reporting via
`engine.report_io_access(...)`, row locks (`acquire_row_locks` in
`frontrun/_dpor_runtime/scheduler.py` — keyed by resource-id strings, nothing in it is
process-local), and wait-for-graph deadlock detection (`row_lock` nodes already exist in
`frontrun/_deadlock.py`) all transfer as-is. This is a strong argument for finishing the
`_dpor_core/` worker/scheduler abstraction first: the cross-process runner becomes a third
backend beside sync-threads and async-tasks.

### Exploration loop

Per iteration: run `setup()` (reset DB to a known state), launch/instruct workers, drive
the schedule to completion, run `invariant()` against the DB, feed the trace back to the
engine, repeat until the wakeup tree is exhausted. Counterexample = schedule of
(process, external access) steps with the same failure formatting as today
(`frontrun/_dpor_core/failures.py`).

## API sketch

```python
import frontrun

result = frontrun.explore(
    strategy="dpor",
    processes={
        "worker_a": frontrun.Subprocess("myapp.checkout:run", args=(order_id,)),
        "worker_b": frontrun.Subprocess("myapp.checkout:run", args=(order_id,)),
    },
    setup=reset_db,                 # runs in coordinator between iterations
    invariant=stock_never_negative, # reads the DB
)
```

`Subprocess` targets a `module:callable` spawned under the `frontrun` CLI environment
(`FRONTRUN_ACTIVE=1`, `LD_PRELOAD`, plus new `FRONTRUN_COORDINATOR=<socket path>` and
`FRONTRUN_WORKER_ID=<n>`). Registered as a `Strategy` adapter in `frontrun/_strategy.py`.

## Phasing

1. **Plumbing (no DPOR):** two spawned workers, SQLite/Postgres, `SchedulerProxy`
   scheduling at SQL points only, exhaustive or random exploration of the (small)
   access-interleaving space. Proves: socket protocol, per-iteration DB reset, invariant
   over external state, deterministic replay of a given schedule.
2. **DPOR:** route accesses into the Rust engine on the coordinator; row locks +
   wait-for-graph deadlock detection across processes (this detects real cross-worker
   `SELECT FOR UPDATE` deadlocks deterministically). Redis support.
3. **Worker reuse:** persistent workers re-running the target callable per iteration
   (spawn cost dominates otherwise); transaction-rollback or template-DB reset strategies.
4. **Unmodified/non-Python workers (speculative):** schedule by blocking inside the
   LD_PRELOAD hooks themselves until granted — at that point the worker needs no Python
   patches at all. This is where the deferred SQL wire-protocol parsing item
   (`possible-future-roadmap/integrations-and-detection.md`) becomes relevant: transaction
   boundaries and statement classification would come from the wire instead of the cursor.

## Determinism caveats and limits

- **Process-private nondeterminism is fine** (it's outside the explored model), but the
  target callables must be deterministic *in their external effects* given the schedule —
  same caveat as today's `setup`/`threads` contract.
- **Autoincrement IDs:** existing indexical aliases (`sql:users:t0_ins0`,
  `frontrun/_sql_insert_tracker.py`) generalize with worker id substituted for thread id;
  the PostgreSQL RETURNING-injection item (already High Priority) becomes more important.
- **Internal threads inside a worker** are uncontrolled in phases 1–3; document that the
  unit of scheduling is the process. (A worker could in principle run its own nested
  frontrun session, but that is out of scope.)
- **DB reset cost** will dominate iteration time; the per-iteration reset hook must
  support cheap strategies (savepoint rollback, `TRUNCATE`, template DB) rather than full
  re-migration.
- **Socket round-trip per external access** is negligible relative to the DB call it
  brackets.

## Why this beats the status quo

Today the recommended pattern for cross-worker races is "run both code paths as threads"
(see `contrib/django`, `contrib/sqlalchemy`). That stays the fast inner loop. This
proposal adds the topology-faithful mode: real connection-per-process, real pool behavior,
real transaction isolation between actors, and the ability to point frontrun at code that
cannot be hosted in-process. It is also the only one of the current proposals that expands
*what frontrun is for* — from a concurrency-testing library into a deterministic
mini-Jepsen for single-database Python systems.
