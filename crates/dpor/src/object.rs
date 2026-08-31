//! Shared object tracking for DPOR.
//!
//! Tracks per-thread access history to each shared object. This is the
//! concrete implementation of the dependency relation used for race detection.
//! Paper: two events are **dependent** when they access the same object and
//! at least one is a write (JACM'17 Section 3.3 p.13-14). Independent events
//! can be reordered without changing the resulting state (Def 3.3).

use std::collections::HashMap;

use crate::access::{Access, AccessKind};

/// Opaque integer ID for shared objects.
pub type ObjectId = u64;

/// Earliest and latest access of one kind by one thread.
///
/// Most recording modes keep a single access (first == last).  Synced-I/O
/// recording keeps both: the earliest access gives the most useful wakeup
/// tree insertion point for read-then-write patterns (e.g. between a SELECT
/// and UPDATE in a transaction), while the latest access is required for
/// soundness — a later read (e.g. a SELECT *after* an UPDATE on the same
/// row) must still race with other threads' writes, otherwise the branch
/// that interleaves between the two statements is silently never explored.
#[derive(Clone, Debug)]
struct AccessSpan {
    first: Access,
    last: Access,
}

impl AccessSpan {
    fn new(access: Access) -> Self {
        Self {
            first: access.clone(),
            last: access,
        }
    }

    /// Iterate the distinct accesses in this span (one when first == last).
    fn entries(&self) -> impl Iterator<Item = &Access> {
        let dup = self.last.path_id == self.first.path_id;
        std::iter::once(&self.first).chain((!dup).then_some(&self.last))
    }

    fn has_path_id(&self, path_id: usize) -> bool {
        self.first.path_id == path_id || self.last.path_id == path_id
    }
}

/// Tracks per-thread accesses to a shared object for DPOR.
///
/// Maintains per-thread spans of read and write accesses.  A **Write** by
/// another thread depends on *both* the reads and the writes from each
/// other thread, because the wakeup tree insertions differ: inserting at a
/// read position allows the scheduler to interleave between a read and a
/// subsequent write on the same object (TOCTOU bugs), while inserting at
/// the write position only reorders complete read-write pairs.
#[derive(Clone, Debug)]
pub struct ObjectState {
    /// Per-thread read access span.
    per_thread_read: HashMap<usize, AccessSpan>,
    /// Per-thread write access span.
    per_thread_write: HashMap<usize, AccessSpan>,
    /// Per-thread weak-write access span.
    per_thread_weak_write: HashMap<usize, AccessSpan>,
    /// Per-thread weak-read access span.
    per_thread_weak_read: HashMap<usize, AccessSpan>,
}

impl ObjectState {
    pub fn new() -> Self {
        Self {
            per_thread_read: HashMap::new(),
            per_thread_write: HashMap::new(),
            per_thread_weak_write: HashMap::new(),
            per_thread_weak_read: HashMap::new(),
        }
    }

    /// Whether *thread* has an access of the given map recorded at *path_id*.
    fn map_has_path_id(map: &HashMap<usize, AccessSpan>, thread: usize, path_id: usize) -> bool {
        map.get(&thread).is_some_and(|span| span.has_path_id(path_id))
    }

    fn map_for_kind(&self, kind: AccessKind) -> &HashMap<usize, AccessSpan> {
        match kind {
            AccessKind::Read => &self.per_thread_read,
            AccessKind::Write => &self.per_thread_write,
            AccessKind::WeakWrite => &self.per_thread_weak_write,
            AccessKind::WeakRead => &self.per_thread_weak_read,
        }
    }

    /// Returns all accesses that the given `kind` by `current_thread` depends on.
    ///
    /// Paper: dependency is defined in JACM'17 Section 3.3 (p.13-14). Two events
    /// on the same object are dependent unless both are reads (reads commute).
    ///
    /// - A **Read** depends on writes from *other* threads (reads are independent).
    /// - A **Write** depends on both reads and writes from *other* threads.
    ///   Returning both ensures DPOR inserts into wakeup trees at read
    ///   positions (for TOCTOU detection) and write positions (for
    ///   write-write ordering).
    ///
    /// Accesses that share a path position with a stronger access by the same
    /// thread (e.g. the read and write halves of a single UPDATE statement)
    /// are *dominated* and skipped so a single statement does not produce
    /// duplicate wakeup insertions.
    pub fn dependent_accesses(&self, kind: AccessKind, current_thread: usize) -> Vec<&Access> {
        let mut result: Vec<&Access> = Vec::new();
        for (index, previous_kind) in AccessKind::ALL.into_iter().enumerate() {
            if !kind.conflicts(previous_kind) {
                continue;
            }
            for (thread, span) in self.map_for_kind(previous_kind) {
                if *thread == current_thread {
                    continue;
                }
                for access in span.entries() {
                    let dominated = AccessKind::ALL[..index].iter().any(|stronger_kind| {
                        kind.conflicts(*stronger_kind)
                            && Self::map_has_path_id(self.map_for_kind(*stronger_kind), *thread, access.path_id)
                    });
                    if !dominated {
                        result.push(access);
                    }
                }
            }
        }
        result
    }

    fn map_for(&mut self, kind: AccessKind) -> &mut HashMap<usize, AccessSpan> {
        match kind {
            AccessKind::Read => &mut self.per_thread_read,
            AccessKind::Write => &mut self.per_thread_write,
            AccessKind::WeakWrite => &mut self.per_thread_weak_write,
            AccessKind::WeakRead => &mut self.per_thread_weak_read,
        }
    }

    /// Record the **latest** access per thread (single-entry semantics):
    /// each new access replaces the previous one.
    pub fn record_access(&mut self, access: Access, kind: AccessKind) {
        let thread_id = access.thread_id;
        self.map_for(kind).insert(thread_id, AccessSpan::new(access));
    }

    /// Like [`record_access`] but keeps the **first** (earliest) access for
    /// each thread rather than overwriting with the latest.  Used for I/O
    /// objects where the earliest position creates the most useful wakeup
    /// tree insertion point (e.g. between a SELECT and UPDATE in a database
    /// transaction).
    pub fn record_io_access(&mut self, access: Access, kind: AccessKind) {
        let thread_id = access.thread_id;
        self.map_for(kind)
            .entry(thread_id)
            .or_insert_with(|| AccessSpan::new(access));
    }

    /// Like [`record_io_access`] but *also* tracks the latest access per
    /// thread.  Keeping only the first access is unsound for synced I/O
    /// (SQL/Redis): a thread that writes a row and later reads it back
    /// (UPDATE ... ; SELECT ...) would drop the later read, so the race
    /// between that read and another thread's write is never detected and
    /// DPOR reports exhaustion without exploring the interleaving where
    /// the other thread's write lands between the two statements.
    pub fn record_synced_io_access(&mut self, access: Access, kind: AccessKind) {
        let thread_id = access.thread_id;
        match self.map_for(kind).entry(thread_id) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().last = access;
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(AccessSpan::new(access));
            }
        }
    }
}

impl Default for ObjectState {
    fn default() -> Self {
        Self::new()
    }
}
