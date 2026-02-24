//! Shared object tracking for DPOR.

use std::collections::HashMap;

use crate::access::{Access, AccessKind};

/// Opaque integer ID for shared objects.
pub type ObjectId = u64;

/// Tracks per-thread accesses to a shared object for DPOR.
///
/// Maintains per-thread maps of the most recent read and the most recent
/// write.  A **Write** by another thread depends on *both* the latest
/// read and the latest write from each other thread, because the
/// backtrack points differ: backtracking at a read position allows the
/// adversary to interleave between a read and a subsequent write on the
/// same object (TOCTOU bugs), while backtracking at the write position
/// only reorders complete read-write pairs.
#[derive(Clone, Debug)]
pub struct ObjectState {
    /// Per-thread most recent read access.
    per_thread_read: HashMap<usize, Access>,
    /// Per-thread most recent write access.
    per_thread_write: HashMap<usize, Access>,
}

impl ObjectState {
    pub fn new() -> Self {
        Self {
            per_thread_read: HashMap::new(),
            per_thread_write: HashMap::new(),
        }
    }

    /// Returns all accesses that the given `kind` by `current_thread` depends on.
    ///
    /// - A **Read** depends on writes from *other* threads (reads are independent).
    /// - A **Write** depends on both reads and writes from *other* threads.
    ///   Returning both ensures DPOR creates backtrack points at read
    ///   positions (for TOCTOU detection) and write positions (for
    ///   write-write ordering).
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
                let mut result: Vec<&Access> = Vec::new();
                // Latest read from each other thread
                for (tid, access) in &self.per_thread_read {
                    if *tid != current_thread {
                        result.push(access);
                    }
                }
                // Latest write from each other thread (may duplicate a
                // thread already covered by a read, but at a different
                // path_id — both backtrack targets matter).
                for (tid, access) in &self.per_thread_write {
                    if *tid != current_thread {
                        // Only add if it's a genuinely different backtrack
                        // target from the read we already included.
                        let dominated = self.per_thread_read.get(tid).is_some_and(|r| {
                            r.path_id == access.path_id
                        });
                        if !dominated {
                            result.push(access);
                        }
                    }
                }
                result
            }
        }
    }

    pub fn record_access(&mut self, access: Access, kind: AccessKind) {
        let thread_id = access.thread_id;
        match kind {
            AccessKind::Read => {
                self.per_thread_read.insert(thread_id, access);
            }
            AccessKind::Write => {
                self.per_thread_write.insert(thread_id, access);
            }
        }
    }
}

impl Default for ObjectState {
    fn default() -> Self {
        Self::new()
    }
}
