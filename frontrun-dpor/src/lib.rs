//! DPOR engine for frontrun concurrency testing.
//!
//! Exposes a Python API via PyO3 for driving systematic interleaving exploration.

pub mod access;
pub mod engine;
pub mod object;
pub mod path;
pub mod thread;
pub mod vv;

use pyo3::prelude::*;

use access::AccessKind;
use engine::{DporEngine, Execution, SyncEvent};

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
    #[pyo3(signature = (num_threads, preemption_bound=None, max_branches=100_000, max_executions=None))]
    fn new(
        num_threads: usize,
        preemption_bound: Option<u32>,
        max_branches: usize,
        max_executions: Option<u64>,
    ) -> Self {
        Self {
            inner: DporEngine::new(num_threads, preemption_bound, max_branches, max_executions),
        }
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

    /// Report a shared memory access. `kind` is "read" or "write".
    fn report_access(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        object_id: u64,
        kind: &str,
    ) -> PyResult<()> {
        let access_kind = match kind {
            "read" => AccessKind::Read,
            "write" => AccessKind::Write,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("kind must be 'read' or 'write', got '{kind}'"),
                ))
            }
        };
        self.inner.process_access(&mut execution.inner, thread_id, object_id, access_kind);
        Ok(())
    }

    /// Report a synchronization event.
    /// event_type: "lock_acquire", "lock_release", "thread_join", "thread_spawn"
    /// sync_id: identifier for the sync primitive or thread
    fn report_sync(
        &mut self,
        execution: &mut PyExecution,
        thread_id: usize,
        event_type: &str,
        sync_id: u64,
    ) -> PyResult<()> {
        let event = match event_type {
            "lock_acquire" => SyncEvent::LockAcquire { lock_id: sync_id },
            "lock_release" => SyncEvent::LockRelease { lock_id: sync_id },
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
        self.inner.process_sync(&mut execution.inner, thread_id, event);
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

    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads()
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
fn frontrun_dpor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDporEngine>()?;
    m.add_class::<PyExecution>()?;
    Ok(())
}
