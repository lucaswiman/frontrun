//! Shared object tracking for DPOR.

use std::collections::HashMap;

use crate::access::{Access, AccessKind};

/// Opaque integer ID for shared objects.
pub type ObjectId = u64;

/// Tracks per-thread accesses to a shared object for DPOR.
///
/// Previous implementation stored only a single `last_access` and
/// `last_write_access`, which meant that with 3+ threads accessing the
/// same object, earlier accesses were overwritten and their conflicts
/// were never explored.
///
/// This version maintains a per-thread map of the most recent access
/// (of any kind) and the most recent write, so that *all* concurrent
/// accesses across threads are checked for conflicts.
#[derive(Clone, Debug)]
pub struct ObjectState {
    /// Per-thread most recent access (any kind).
    per_thread_access: HashMap<usize, Access>,
    /// Per-thread most recent write access.
    per_thread_write: HashMap<usize, Access>,
}

impl ObjectState {
    pub fn new() -> Self {
        Self {
            per_thread_access: HashMap::new(),
            per_thread_write: HashMap::new(),
        }
    }

    /// Returns all accesses that the given `kind` by `current_thread` depends on.
    ///
    /// - A **Read** depends on writes from *other* threads (reads are independent).
    /// - A **Write** depends on any access from *other* threads.
    pub fn dependent_accesses(&self, kind: AccessKind, current_thread: usize) -> Vec<&Access> {
        match kind {
            AccessKind::Read => {
                self.per_thread_write
                    .iter()
                    .filter(|(tid, _)| **tid != current_thread)
                    .map(|(_, access)| access)
                    .collect()
            }
            AccessKind::Write => {
                self.per_thread_access
                    .iter()
                    .filter(|(tid, _)| **tid != current_thread)
                    .map(|(_, access)| access)
                    .collect()
            }
        }
    }

    pub fn record_access(&mut self, access: Access, kind: AccessKind) {
        let thread_id = access.thread_id;
        if kind == AccessKind::Write {
            self.per_thread_write.insert(thread_id, access.clone());
        }
        self.per_thread_access.insert(thread_id, access);
    }
}

impl Default for ObjectState {
    fn default() -> Self {
        Self::new()
    }
}
