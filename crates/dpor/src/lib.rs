//! DPOR engine for frontrun concurrency testing.
//!
//! Exposes a Python API via PyO3 for driving systematic interleaving exploration.

pub mod access;
pub mod engine;
pub mod object;
pub mod path;
pub mod thread;
pub mod vv;
pub mod wakeup_tree;

use pyo3::prelude::*;

use access::AccessKind;
use engine::{DporEngine, Execution, SyncEvent};
use path::SearchStrategy;

// ---------------------------------------------------------------------------
// PyO3 wrapper types
// ---------------------------------------------------------------------------

/// The DPOR exploration engine, exposed to Python.
///
/// Holds the exploration tree (persisted across executions) and creates
/// per-execution state. Python drives the engine by alternating between
/// scheduling, reporting accesses/syncs, and advancing to the next execution.
#[pyclass]
struct PyDporEngine {
    inner: DporEngine,
}

/// Per-execution state, exposed to Python.
///
/// Holds thread vector clocks, object access tracking, and the schedule trace
/// for the current execution. Created by `PyDporEngine.begin_execution()` and
/// passed back to the engine for scheduling and access reporting.
#[pyclass]
struct PyExecution {
    inner: Execution,
}

#[pymethods]
impl PyDporEngine {
    #[new]
    #[pyo3(signature = (num_threads, preemption_bound=None, max_branches=100_000, max_executions=None, search=None))]
    fn new(
        num_threads: usize,
        preemption_bound: Option<u32>,
        max_branches: usize,
        max_executions: Option<u64>,
        search: Option<&str>,
    ) -> PyResult<Self> {
        let strategy = Self::parse_search_strategy(search)?;
        Ok(Self {
            inner: DporEngine::new(num_threads, preemption_bound, max_branches, max_executions, strategy),
        })
    }

    /// Start a new execution. Returns fresh per-execution state.
    fn begin_execution(&self) -> PyExecution {
        PyExecution {
            inner: self.inner.begin_execution(),
        }
    }

    /// Pick which thread to run next. Returns thread ID or None (deadlock).
    fn schedule(&mut self, execution: &mut PyExecution) -> Option<usize> {
        self.inner.schedule(&mut execution.inner)
    }

    /// Prefer a runnable thread at the next newly-created scheduling point.
    /// Replayed wakeup-tree choices remain authoritative.
    fn prefer_next_thread(&mut self, thread_id: usize) {
        self.inner.prefer_next_thread(thread_id);
    }

    /// Report a shared memory access. `kind` is "read", "write", or "weak_write".
    fn report_access(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        object_id: u64,
        kind: &str,
    ) -> PyResult<()> {
        let access_kind = Self::parse_access_kind(kind)?;
        self.inner.process_access(&mut execution.inner, thread_id, object_id, access_kind);
        Ok(())
    }

    /// Report a shared memory access attributed to an explicit path position.
    fn report_access_at(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        object_id: u64,
        kind: &str,
        path_id: usize,
    ) -> PyResult<()> {
        let access_kind = Self::parse_access_kind(kind)?;
        self.inner.process_access_at(&mut execution.inner, thread_id, object_id, access_kind, path_id);
        Ok(())
    }

    /// Report a first-access shared memory access.  Like `report_access`
    /// but keeps the earliest access per thread rather than the latest.
    /// Used for container-level keys where multiple writes to the same
    /// container should preserve the first write's position for
    /// fine-grained wakeup tree insertion.
    fn report_first_access(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        object_id: u64,
        kind: &str,
    ) -> PyResult<()> {
        let access_kind = Self::parse_access_kind(kind)?;
        self.inner.process_first_access(&mut execution.inner, thread_id, object_id, access_kind);
        Ok(())
    }

    /// Report a first-access shared memory access attributed to an explicit path position.
    fn report_first_access_at(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        object_id: u64,
        kind: &str,
        path_id: usize,
    ) -> PyResult<()> {
        let access_kind = Self::parse_access_kind(kind)?;
        self.inner.process_first_access_at(&mut execution.inner, thread_id, object_id, access_kind, path_id);
        Ok(())
    }

    /// Report a Python-level I/O access (Redis, SQL).  Uses `dpor_vv`
    /// so that lock-based happens-before edges are respected.  These
    /// operations go through Python code, so Python locks tracked via
    /// `report_sync` should order them correctly.
    fn report_synced_io_access(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        object_id: u64,
        kind: &str,
    ) -> PyResult<()> {
        let access_kind = Self::parse_access_kind(kind)?;
        self.inner.process_synced_io_access(&mut execution.inner, thread_id, object_id, access_kind);
        Ok(())
    }

    /// Report an I/O access (file/socket).  Uses a separate vector clock
    /// that ignores lock-based happens-before, so I/O from different
    /// threads is always treated as potentially concurrent.
    fn report_io_access(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        object_id: u64,
        kind: &str,
    ) -> PyResult<()> {
        let access_kind = Self::parse_access_kind(kind)?;
        self.inner.process_io_access(&mut execution.inner, thread_id, object_id, access_kind);
        Ok(())
    }

    /// Report a synchronization event.
    /// event_type: "lock_acquire", "lock_release", "lock_attempt_ok",
    /// "lock_attempt_fail", "thread_join", "thread_spawn"
    /// sync_id: identifier for the sync primitive or thread
    /// path_id: optional path position for lock events (free-threaded fix)
    #[pyo3(signature = (execution, thread_id, event_type, sync_id, path_id=None))]
    fn report_sync(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        event_type: &str,
        sync_id: u64,
        path_id: Option<usize>,
    ) -> PyResult<()> {
        let event = match event_type {
            "lock_acquire" => SyncEvent::LockAcquire { lock_id: sync_id },
            "lock_release" => SyncEvent::LockRelease { lock_id: sync_id },
            "lock_attempt_ok" => SyncEvent::LockAttempt { lock_id: sync_id, acquired: true },
            "lock_attempt_fail" => SyncEvent::LockAttempt { lock_id: sync_id, acquired: false },
            "thread_join" => SyncEvent::ThreadJoin {
                joined_thread: sync_id as usize,
            },
            "thread_spawn" => SyncEvent::ThreadSpawn {
                child_thread: sync_id as usize,
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown event_type: '{event_type}'"),
                ))
            }
        };
        self.inner.process_sync(&mut execution.inner, thread_id, event, path_id);
        Ok(())
    }

    /// Finish current execution and advance to next path. Returns True if more to explore.
    fn next_execution(&mut self) -> bool {
        self.inner.next_execution()
    }

    #[getter]
    fn executions_completed(&self) -> u64 {
        self.inner.executions_completed()
    }

    #[getter]
    fn tree_depth(&self) -> usize {
        self.inner.tree_depth()
    }

    /// Return the current path position (number of scheduling steps so far).
    /// Used by Python to snapshot the position for sync event attribution.
    #[getter]
    fn path_position(&self) -> usize {
        self.inner.path.current_position()
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads()
    }

    /// Register that `object_id` belongs to resource group `group_id`.
    ///
    /// Objects in the same resource group (e.g., SQL resources from the same
    /// table) are related. When a race is detected on an object with a
    /// resource group, inline wakeup insertion is skipped in favor of
    /// deferred notdep processing. This prevents backtrack explosion from
    /// intermediate operations on unrelated tables (Defect #15).
    fn register_resource_group(&mut self, object_id: u64, group_id: u64) {
        self.inner.register_resource_group(object_id, group_id);
    }

    /// Return pending races detected during the current execution.
    /// Each race is (prev_path_id, current_path_id, thread_id, race_object).
    /// Call before next_execution() which consumes them.
    fn pending_races(&self) -> Vec<(usize, usize, usize, Option<u64>)> {
        self.inner
            .pending_races()
            .iter()
            .map(|r| (r.prev_path_id, r.current_path_id, r.thread_id, r.race_object))
            .collect()
    }

    /// Return only attribute-level data races (not container-level or lock-synthetic).
    /// These represent true unsynchronized shared memory accesses suitable for
    /// `error_on_any_race` mode.
    fn attribute_races(&self) -> Vec<(usize, usize, usize, Option<u64>)> {
        self.inner
            .pending_races()
            .iter()
            .filter(|r| r.is_attribute_race)
            .map(|r| (r.prev_path_id, r.current_path_id, r.thread_id, r.race_object))
            .collect()
    }
}

impl PyDporEngine {
    fn parse_access_kind(kind: &str) -> PyResult<AccessKind> {
        match kind {
            "read" => Ok(AccessKind::Read),
            "write" => Ok(AccessKind::Write),
            "weak_write" => Ok(AccessKind::WeakWrite),
            "weak_read" => Ok(AccessKind::WeakRead),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("kind must be 'read', 'write', 'weak_write', or 'weak_read', got '{kind}'"),
            )),
        }
    }

    /// Parse the `search` parameter into a `SearchStrategy`.
    ///
    /// Accepted formats:
    ///   - None / "dfs"                  → DFS (default)
    ///   - "bit-reversal"                → BitReversal with seed=0
    ///   - "bit-reversal:42"             → BitReversal with seed=42
    ///   - "round-robin"                 → RoundRobin with seed=0
    ///   - "round-robin:7"               → RoundRobin with seed=7
    ///   - "stride"                      → Stride with seed=0
    ///   - "stride:3"                    → Stride with seed=3
    ///   - "conflict-first"              → ConflictFirst (deterministic priority by race depth)
    fn parse_search_strategy(search: Option<&str>) -> PyResult<SearchStrategy> {
        match search {
            None | Some("dfs") => Ok(SearchStrategy::Dfs),
            Some(s) if s == "bit-reversal" => Ok(SearchStrategy::BitReversal { seed: 0 }),
            Some(s) if s.starts_with("bit-reversal:") => {
                let seed_str = &s["bit-reversal:".len()..];
                let seed = seed_str.parse::<u64>().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid seed in '{s}': {e}"))
                })?;
                Ok(SearchStrategy::BitReversal { seed })
            }
            Some(s) if s == "round-robin" => Ok(SearchStrategy::RoundRobin { seed: 0 }),
            Some(s) if s.starts_with("round-robin:") => {
                let seed_str = &s["round-robin:".len()..];
                let seed = seed_str.parse::<u64>().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid seed in '{s}': {e}"))
                })?;
                Ok(SearchStrategy::RoundRobin { seed })
            }
            Some(s) if s == "stride" => Ok(SearchStrategy::Stride { seed: 0 }),
            Some(s) if s.starts_with("stride:") => {
                let seed_str = &s["stride:".len()..];
                let seed = seed_str.parse::<u64>().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid seed in '{s}': {e}"))
                })?;
                Ok(SearchStrategy::Stride { seed })
            }
            Some("conflict-first") => Ok(SearchStrategy::ConflictFirst),
            Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown search strategy '{other}'; expected 'dfs', 'bit-reversal[:seed]', \
                 'round-robin[:seed]', 'stride[:seed]', or 'conflict-first'"
            ))),
        }
    }
}

#[pymethods]
impl PyExecution {
    /// Mark a thread as finished.
    fn finish_thread(&mut self, thread_id: usize) {
        self.inner.finish_thread(thread_id);
    }

    /// Mark a thread as blocked.
    fn block_thread(&mut self, thread_id: usize) {
        self.inner.block_thread(thread_id);
    }

    /// Mark a thread as unblocked.
    fn unblock_thread(&mut self, thread_id: usize) {
        self.inner.unblock_thread(thread_id);
    }

    /// Get the list of runnable thread IDs.
    fn runnable_threads(&self) -> Vec<usize> {
        self.inner.runnable_threads()
    }

    /// Drop a just-scheduled transition that had no physical effect.
    fn discard_last_schedule_step(&mut self, thread_id: usize) {
        if self.inner.schedule_trace.last() == Some(&thread_id) {
            self.inner.schedule_trace.pop();
        }
    }

    /// Get the schedule trace (sequence of thread choices).
    #[getter]
    fn schedule_trace(&self) -> Vec<usize> {
        self.inner.schedule_trace.clone()
    }

    /// Check if this execution was aborted.
    #[getter]
    fn aborted(&self) -> bool {
        self.inner.aborted
    }
}

/// Python module definition.
///
/// `gil_used = false` declares this extension supports free-threaded Python
/// (3.13t/3.14t). The DPOR engine state is only mutated through Python calls
/// serialized by our cooperative scheduler, so this is safe.
#[pymodule(gil_used = false)]
fn _dpor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDporEngine>()?;
    m.add_class::<PyExecution>()?;
    Ok(())
}
