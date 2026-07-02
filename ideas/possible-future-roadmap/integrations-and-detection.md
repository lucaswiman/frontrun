# Integrations and Detection: Remaining Work

Last reviewed: 2026-06-12.

This document consolidates remaining unfinished work from SQL conflict detection, Redis, and
stateful resource detection layers. **Already implemented:** SQL cursor patching for
sqlite3/psycopg2/pymysql/aiosqlite/asyncpg, table/row-level conflict detection, wire protocol
parsing, Redis key-level detection, I/O detection layers (sys.setprofile, socket/file patching),
LD_PRELOAD library.

Previously-listed items dropped as not worth doing (superseded by existing cursor/driver
patching, or speculative with no concrete use case): `__class__` reassignment taint
propagation, `gc.get_referrers()` resource discovery, deterministic record/replay of external
state, import-hook known-library registry, one-line decorator annotation, frame-local variable
poisoning. See git history for the original write-ups.

## High Priority

### Autoincrement RETURNING Clause Injection (PostgreSQL)

**What:** For psycopg2 and psycopg3, `lastrowid` is unavailable after INSERT. Inject a RETURNING clause to capture the inserted row's ID explicitly.

**Why:** Currently raises `NondeterministicSQLError` when `warn_nondeterministic_sql=True`. With RETURNING, every INSERT gets an indexical alias like `sql:users:t0_ins0`, mapping concrete row IDs to these aliases for downstream conflict detection.

**Complexity:** Low. Modify `_sql_insert_tracker.py` to wrap INSERT statements lacking RETURNING and inject `RETURNING id` (or the appropriate PK column). Handle edge cases: multi-row inserts, composite PKs, explicit RETURNING clauses already present.

**Location:** `frontrun/_sql_insert_tracker.py`

---

## Medium Priority

### Cross-Table Foreign Key Analysis

**What:** Schema introspection to detect FK dependencies, e.g. `orders.user_id` → `users.id`. Currently `INSERT INTO orders (user_id, ...)` and `DELETE FROM users WHERE id = ?` are marked independent (different tables), but the FK creates a real conflict.

**Why:** More accurate conflict detection. Especially important for referential integrity bugs and cascade-delete scenarios.

**Complexity:** Medium (~150 lines + 25 tests).
- Query `information_schema.referential_constraints` on first connection to PostgreSQL/MySQL
- Build FK dependency graph: `{orders → users, shipments → orders}`
- At conflict detection: if Op1 touches T1 and Op2 touches T2 with T1 → T2 via FK, mark as dependent
- Manual FK registration via `frontrun/_schema.py` already exists; automatic introspection is the remaining piece

**Location:** `frontrun/_schema.py`, `frontrun/_sql_cursor.py`

---

## Low Priority / Long-Term

### Transaction Identity via Driver APIs

**What:** Use `cursor.connection.info.backend_pid` (psycopg2/psycopg3) or `conn.in_transaction` (SQLite) to track connection and transaction boundaries more reliably than wire-protocol parsing.

**Why:** Distinguishes independent connections from shared connections. Handles autocommit, savepoints, and driver-specific transaction semantics without C-level wire parsing. Also a prerequisite-adjacent piece for cross-process exploration (see `ideas/cross_process_exploration.md`), where per-connection identity is what ties an external access to a schedulable client.

**Complexity:** Low. Already accessible from `_sql_cursor.py`. Add to resource_id: `sql:{table}:conn={backend_pid}`. Replaces removed "shared socket" warning with per-connection tracking.

**Location:** `frontrun/_sql_cursor.py`, `frontrun/_io_detection.py`

---

### sys.addaudithook Integration (Layer 0)

**What:** Zero-config safety net using `sys.addaudithook` to intercept `socket.connect` and `open` events from C code before they even reach Python's socket/file layers.

**Why:** Catches I/O from C extensions that bypass Python's socket module (rare but possible), and provides a fallback for detection when other layers are disabled.

**Complexity:** Low (~20 lines). Limitation: granularity is coarse (entire endpoint, not per-table); audit hooks can't be removed (must gate on test-run flag).

**Status:** Verified experimentally (experiment scripts not retained in-repo). Production integration deferred pending need for broader compatibility. Currently `sys.setprofile` + socket/file patching cover the practical cases.

**Location:** Could be added to `frontrun/_io_detection.py` as fallback layer

---

### sys.monitoring CALL Events (Layer 1.5, Python 3.12+)

**What:** Use PEP 669 `CALL` event type to detect calls to `.execute()`, `.send()`, `.write()` etc. without code rewriting.

**Why:** Lower overhead than `sys.settrace`-based detection. Coexists with existing INSTRUCTION events on same tool ID.

**Complexity:** Low (~30 lines). Add `CALL` to event bitmask, check callable name against `RESOURCE_METHOD_NAMES = {"execute", "send", "recv", "read", "write", "commit", "rollback"}`.

**Status:** Verified experimentally on Python 3.13 (INSTRUCTION + CALL events coexist on the same tool). Not yet integrated into production code. Any new hook must go through `frontrun/_opcode_observer.py`, which is the sole owner of `sys.monitoring`.

**Location:** `frontrun/_io_detection.py` (3.12+ only path)

---

## Deferred / Experimental

### SQL Wire-Protocol Parsing (LD_PRELOAD Level)

**What:** Parse PostgreSQL `K` (BackendKeyData) and `Z` (ReadyForQuery) messages in `LD_PRELOAD` recv hooks to extract `backend_pid` and transaction boundaries.

**Why:** Enables transaction tracking and connection identity at C level for non-Python drivers (libpq FFI, etc.).

**Note on non-Python cross-process workers:** parsing the wire protocol *inside an LD_PRELOAD recv hook* is the wrong layer for scheduling unmodified workers — it is one-sided (no quiescence), misses SQLite/static binaries, and can't drive aborts. If unmodified/non-Python worker scheduling is ever pursued, the wire parsing belongs in a **DSN-level proxy** (a frontrun process in front of Postgres/Redis) that has full-duplex visibility and can define step-completion as response-forwarded; see the Implemented / cross-process note in `index.md`. This `sql_extract.rs` PG parser is a reusable seed for that proxy either way.

**Complexity:** Medium. ~10 lines for BackendKeyData (one-time). ~50 lines for ReadyForQuery (requires message framing per fd).

**Status:** Documented. Deferred pending an actual use case for C-level direct libpq access. Python driver APIs (`conn.info.backend_pid`, `cursor.connection`) sufficient for mainstream usage.

---

## Testing & Validation

- **SQL tests:** `tests/test_sql_*.py` (sqlite3, psycopg2, asyncpg, etc.)
- **Integration tests:** `tests/test_integration_*.py` (require Redis, Postgres)

Run via `make test-3.14` or `make test-integration-3.14`.

---

## Layered Detection Summary

For reference, the complete detection stack (highest to lowest precision):

```
Layer 6: User annotations (@frontrun.resource, with accessing)
Layer 4: Known-library plugins (sqlalchemy, redis, psycopg2)
Layer 3: Duck-typing heuristics (sys.monitoring CALL → .execute(), .send())
Layer 2: Taint propagation (__class__ reassignment / proxy + gc.get_referrers)
Layer 1.5: sys.setprofile C_CALL events (already implemented)
Layer 1: Socket/file monkey-patching (already implemented)
Layer 0: sys.addaudithook (zero-config but coarse)
```

Currently deployed: **Layers 1, 1.5, and targeted Layer 2 (via cursor patching)**. The
unimplemented layers above remain documented here only insofar as they have a plausible
trigger; the rest were dropped (see note at top).
