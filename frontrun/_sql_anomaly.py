"""SQL isolation anomaly classification for DPOR failing interleavings.

Given the list of TraceEvents from a failing execution, classifies the SQL-level
concurrency anomaly by building a Dependency Serialization Graph (DSG) and
detecting cycles whose edge types determine the anomaly kind.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from frontrun._trace_format import TraceEvent

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SqlAnomaly:
    """A classified SQL isolation anomaly."""

    kind: str  # "lost_update", "write_skew", "dirty_read", "non_repeatable_read", "write_write"
    summary: str
    tables: frozenset[str]
    threads: frozenset[int]


# ---------------------------------------------------------------------------
# Helper: extract SQL events from a trace
# ---------------------------------------------------------------------------


def _extract_sql_events(events: list[TraceEvent]) -> list[tuple[int, str, str, str]]:
    # Returns (thread_id, resource_id, table_name, access_type) for every SQL IO event.
    # resource_id is the full key (e.g. "accounts" or "accounts:(('id','1'),)") for
    # conflict matching; table_name is the bare table for display.
    result: list[tuple[int, str, str, str]] = []
    for ev in sorted(events, key=lambda e: e.step_index):
        if ev.obj_type_name != "IO":
            continue
        attr = ev.attr_name or ""
        if not attr.startswith("sql:"):
            continue
        access = ev.access_type
        if access is None:
            continue
        remainder = attr[4:]  # everything after "sql:"
        table = remainder.split(":")[0]
        if table:
            result.append((ev.thread_id, remainder, table, access))
    return result


def _resource_to_table(resource: str) -> str:
    return resource.split(":")[0]


# ---------------------------------------------------------------------------
# DSG construction
# ---------------------------------------------------------------------------

# An edge: (from_thread, to_thread, edge_type, resource_id)
_Edge = tuple[int, int, str, str]


def _build_dsg(ops: list[tuple[int, str, str, str]]) -> list[_Edge]:
    # Build DSG edges. ops is (thread_id, resource_id, table, access_type) in order.
    # Conflicts are keyed by resource_id (includes row predicates when present).
    res_thread_ops: dict[str, dict[int, list[tuple[int, str]]]] = {}
    for idx, (tid, resource, _table, access) in enumerate(ops):
        res_thread_ops.setdefault(resource, {}).setdefault(tid, []).append((idx, access))

    edges: list[_Edge] = []
    seen: set[tuple[int, int, str, str]] = set()

    for resource, thread_map in res_thread_ops.items():
        tids = list(thread_map.keys())
        for i, ta in enumerate(tids):
            for tb in tids[i + 1 :]:
                for idx_a, acc_a in thread_map[ta]:
                    for idx_b, acc_b in thread_map[tb]:
                        if idx_a < idx_b:
                            frm, to, first_acc, second_acc = ta, tb, acc_a, acc_b
                        else:
                            frm, to, first_acc, second_acc = tb, ta, acc_b, acc_a

                        if first_acc == "write" and second_acc == "read":
                            etype = "WR"
                        elif first_acc == "read" and second_acc == "write":
                            etype = "RW"
                        elif first_acc == "write" and second_acc == "write":
                            etype = "WW"
                        else:
                            continue

                        key = (frm, to, etype, resource)
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)

    return edges


# ---------------------------------------------------------------------------
# Cycle detection via DFS
# ---------------------------------------------------------------------------


def _find_cycle(graph: dict[int, list[tuple[int, str, str]]]) -> list[tuple[int, int, str, str]] | None:
    visited: set[int] = set()
    rec_stack: list[int] = []
    on_stack: set[int] = set()  # O(1) membership check for back-edge detection
    edge_to: dict[int, tuple[int, int, str, str]] = {}

    def dfs(node: int) -> list[tuple[int, int, str, str]] | None:
        visited.add(node)
        rec_stack.append(node)
        on_stack.add(node)
        for neighbor, etype, resource in graph.get(node, []):
            edge = (node, neighbor, etype, resource)
            if neighbor not in visited:
                edge_to[neighbor] = edge
                result = dfs(neighbor)
                if result is not None:
                    return result
            elif neighbor in on_stack:
                cycle_edges: list[tuple[int, int, str, str]] = [edge]
                idx = rec_stack.index(neighbor)
                for k in range(idx, len(rec_stack) - 1):
                    e = edge_to.get(rec_stack[k + 1])
                    if e is not None and e[0] == rec_stack[k]:
                        cycle_edges.append(e)
                return cycle_edges
        rec_stack.pop()
        on_stack.discard(node)
        return None

    for start in sorted(graph.keys()):
        if start not in visited:
            result = dfs(start)
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# Pre-DSG anomaly checks
# ---------------------------------------------------------------------------


def _iter_table_read_conflicts(
    ops: list[tuple[int, str, str, str]],
) -> Iterator[tuple[int, int, str, set[str]]]:
    """Yield ``(focus_tid, writer_tid, table, writer_accesses)`` for each write by
    another thread that falls between the first and last table-level read of
    ``focus_tid`` on ``table``.  Shared scaffolding for NRR and phantom checks.
    """
    for focus_tid in sorted({tid for tid, _, _, _ in ops}):
        table_read_positions: dict[str, list[int]] = {}
        for i, (tid, _resource, table, access) in enumerate(ops):
            if tid == focus_tid and access == "read":
                table_read_positions.setdefault(table, []).append(i)

        for table, positions in table_read_positions.items():
            if len(positions) < 2:
                continue
            first_read, last_read = positions[0], positions[-1]
            for i, (tid, _res, tbl, access) in enumerate(ops):
                if tid == focus_tid or tbl != table or access != "write":
                    continue
                if not (first_read < i < last_read):
                    continue
                writer_accesses = {acc for t, _r, tb, acc in ops if t == tid and tb == table}
                yield focus_tid, tid, table, writer_accesses


def _check_non_repeatable_read(ops: list[tuple[int, str, str, str]]) -> SqlAnomaly | None:
    # Same thread reads same resource (or same table) twice with another thread's write in between.
    def _nrr(focus_tid: int, writer_tid: int, table: str) -> SqlAnomaly:
        return SqlAnomaly(
            kind="non_repeatable_read",
            summary=(
                f"Non-repeatable read: thread {focus_tid} read table '{table}' twice "
                f"but thread {writer_tid} wrote to it in between, changing the result."
            ),
            tables=frozenset([table]),
            threads=frozenset([focus_tid, writer_tid]),
        )

    # 1. Exact resource-level check (original logic): same exact resource_id read
    #    twice by focus_tid with another thread's write on that same resource_id
    #    between the two reads.  Fires regardless of the writer's other accesses.
    #
    #    The ``<table>:seq`` membership marker is excluded here: only INSERT/DELETE
    #    write it (SELECT/UPDATE read it), so a read-write-read on ``:seq`` is
    #    always a phantom, never a non-repeatable read.  Letting it match here
    #    would shadow _check_phantom_read for every real INSERT/DELETE phantom.
    for focus_tid in sorted({tid for tid, _, _, _ in ops}):
        resource_read_positions: dict[str, list[int]] = {}
        for i, (tid, resource, _table, access) in enumerate(ops):
            if tid == focus_tid and access == "read" and not resource.endswith(":seq"):
                resource_read_positions.setdefault(resource, []).append(i)

        for resource, positions in resource_read_positions.items():
            if len(positions) < 2:
                continue
            first_read, last_read = positions[0], positions[-1]
            for i, (tid, res, _table, access) in enumerate(ops):
                if tid != focus_tid and res == resource and access == "write" and first_read < i < last_read:
                    return _nrr(focus_tid, tid, _resource_to_table(resource))

    # 2. Cross-granularity check: same table, different resource IDs.  Only fires
    #    when the writing thread also reads the table (i.e. an UPDATE).  Write-only
    #    threads (INSERT/DELETE) are phantom reads, handled by _check_phantom_read.
    for focus_tid, writer_tid, table, writer_accesses in _iter_table_read_conflicts(ops):
        if "read" in writer_accesses:
            return _nrr(focus_tid, writer_tid, table)
    return None


def _check_phantom_read(ops: list[tuple[int, str, str, str]]) -> SqlAnomaly | None:
    """Detect phantom reads: a thread reads a table (table-level, no row predicate),
    then another thread inserts/deletes rows in that table, then the first thread
    reads the same table again.

    Phantom reads are distinguished from non-repeatable reads by the nature of
    the change: phantoms involve new or removed rows (INSERT/DELETE between two
    reads), while non-repeatable reads involve changed values (UPDATE between
    two reads of the same row).

    Heuristic: if the writing thread does NOT read this table, the write is
    likely an INSERT/DELETE (no prior read needed) and we classify as phantom.
    """
    for focus_tid, writer_tid, table, writer_accesses in _iter_table_read_conflicts(ops):
        if "read" not in writer_accesses:
            return SqlAnomaly(
                kind="phantom_read",
                summary=(
                    f"Phantom read: thread {focus_tid} read table '{table}' twice "
                    f"but thread {writer_tid} inserted/deleted rows in between, "
                    f"changing the set of rows returned."
                ),
                tables=frozenset([table]),
                threads=frozenset([focus_tid, writer_tid]),
            )
    return None


def _check_lost_update(ops: list[tuple[int, str, str, str]]) -> SqlAnomaly | None:
    # Lost update: two threads both read AND write the same resource, and their
    # operations interleave (each thread's read precedes the other thread's write).
    # Serialized executions (T0 fully completes before T1 starts) are not lost updates.
    resource_thread_ops: dict[str, dict[int, list[tuple[int, str]]]] = {}
    for idx, (tid, resource, _table, access) in enumerate(ops):
        resource_thread_ops.setdefault(resource, {}).setdefault(tid, []).append((idx, access))

    for resource, thread_map in resource_thread_ops.items():
        # Collect threads with both read and write operations, recording
        # the earliest read and latest write index for each.
        rw_threads: dict[int, tuple[int, int]] = {}
        for tid, thread_ops in thread_map.items():
            reads = [i for i, acc in thread_ops if acc == "read"]
            writes = [i for i, acc in thread_ops if acc == "write"]
            if reads and writes:
                rw_threads[tid] = (min(reads), max(writes))

        tids = sorted(rw_threads.keys())
        for i, ta in enumerate(tids):
            for tb in tids[i + 1 :]:
                r0, w0 = rw_threads[ta]
                r1, w1 = rw_threads[tb]
                # A true lost update requires interleaving: each thread reads
                # before the other thread's last write.  If r1 >= w0, T1 read
                # after T0 finished writing (serialized), so T1 saw T0's result
                # and no update is lost.  Symmetric check for the other order.
                if r0 < w1 and r1 < w0:
                    table = _resource_to_table(resource)
                    return SqlAnomaly(
                        kind="lost_update",
                        summary=(
                            f"Lost update on table '{table}': both threads read then wrote the same rows, "
                            f"so one thread's update overwrote the other's."
                        ),
                        tables=frozenset([table]),
                        threads=frozenset([ta, tb]),
                    )
    return None


# ---------------------------------------------------------------------------
# Anomaly classification
# ---------------------------------------------------------------------------


def classify_sql_anomaly(events: list[TraceEvent]) -> SqlAnomaly | None:
    """Classify the SQL isolation anomaly visible in *events*.

    Returns None when no SQL events are present or no cross-thread conflict is found.
    """
    ops = _extract_sql_events(events)
    if not ops:
        return None

    # Need at least two threads to have a conflict
    if len({tid for tid, _, _, _ in ops}) < 2:
        return None

    # Pre-DSG checks for patterns that don't require cycle detection.
    nrr = _check_non_repeatable_read(ops)
    if nrr is not None:
        return nrr

    # Phantom reads are a specialized case: a thread reads a table twice and
    # another thread performs a write-only (INSERT/DELETE, no read) in between.
    # Checked after non-repeatable read since NRR is the more common pattern.
    phantom = _check_phantom_read(ops)
    if phantom is not None:
        return phantom

    lu = _check_lost_update(ops)
    if lu is not None:
        return lu

    edges = _build_dsg(ops)
    if not edges:
        return None

    # Build adjacency list for cycle detection
    graph: dict[int, list[tuple[int, str, str]]] = {}
    for frm, to, etype, resource in edges:
        graph.setdefault(frm, []).append((to, etype, resource))

    cycle = _find_cycle(graph)

    if cycle is not None:
        cycle_etypes = {etype for _, _, etype, _ in cycle}
        cycle_resources = {r for _, _, _, r in cycle}
        cycle_tables = frozenset(_resource_to_table(r) for r in cycle_resources)
        cycle_threads = frozenset(n for frm, to, _, _ in cycle for n in (frm, to))

        # Dirty read: WR edge in cycle
        if "WR" in cycle_etypes:
            tbls = ", ".join(f"'{t}'" for t in sorted(cycle_tables))
            return SqlAnomaly(
                kind="dirty_read",
                summary=f"Dirty read on {tbls}: a thread read data from an uncommitted concurrent write.",
                tables=cycle_tables,
                threads=cycle_threads,
            )

        # Write skew: RW-only cycle (may span one or more tables)
        if cycle_etypes == {"RW"}:
            tbls = ", ".join(f"'{t}'" for t in sorted(cycle_tables))
            return SqlAnomaly(
                kind="write_skew",
                summary=(
                    f"Write skew across tables {tbls}: each thread read a table the other wrote, "
                    f"making decisions based on a collectively inconsistent snapshot."
                ),
                tables=cycle_tables,
                threads=cycle_threads,
            )

        # Write-write cycle
        if cycle_etypes == {"WW"}:
            tbls = ", ".join(f"'{t}'" for t in sorted(cycle_tables))
            return SqlAnomaly(
                kind="write_write",
                summary=f"Write-write conflict on {tbls}: concurrent writes without coordination.",
                tables=cycle_tables,
                threads=cycle_threads,
            )

    # No cycle — check for single-edge anomalies
    for frm, to, etype, resource in edges:
        tbl = _resource_to_table(resource)
        if etype == "WR":
            return SqlAnomaly(
                kind="dirty_read",
                summary=f"Dirty read: thread {to} read from table '{tbl}' written by thread {frm} before it committed.",
                tables=frozenset([tbl]),
                threads=frozenset([frm, to]),
            )

    for frm, to, etype, resource in edges:
        tbl = _resource_to_table(resource)
        if etype == "WW":
            return SqlAnomaly(
                kind="write_write",
                summary=f"Write-write conflict on table '{tbl}': threads {frm} and {to} both wrote concurrently.",
                tables=frozenset([tbl]),
                threads=frozenset([frm, to]),
            )

    return None
