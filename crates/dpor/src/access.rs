//! DPOR access tracking.
//!
//! Records shared memory accesses for race detection. Two accesses form a
//! **race** (JACM'17 Def 3.3 p.13-14) when they are from different threads,
//! at least one is a write, they access the same object, and they are not
//! ordered by the happens-before relation.

use crate::vv::VersionVec;

/// Provenance tag indicating how an access was generated.
///
/// Each access is tagged with its origin so that future merge strategies
/// can apply per-origin policies (e.g., treating I/O accesses differently
/// from Python memory accesses during sleep-set propagation).
///
/// Ordering by "strength": `IoDirect` > `LockSynthetic` > `PythonMemory`.
/// When merging two accesses with different origins, the stronger one wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessOrigin {
    /// Access observed via Python shared-memory tracing (sys.settrace).
    PythonMemory,
    /// Synthetic access generated for lock acquire/release conflict tracking.
    LockSynthetic,
    /// Direct I/O access (file, socket, database) intercepted at the C level.
    IoDirect,
}

impl AccessOrigin {
    /// Merge two origins, keeping the "stronger" one.
    /// Strength order: `IoDirect` > `LockSynthetic` > `PythonMemory`.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            _ if self == other => self,
            (AccessOrigin::IoDirect, _) | (_, AccessOrigin::IoDirect) => AccessOrigin::IoDirect,
            (AccessOrigin::LockSynthetic, _) | (_, AccessOrigin::LockSynthetic) => {
                AccessOrigin::LockSynthetic
            }
            _ => AccessOrigin::PythonMemory,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
    /// Like `Write` but does NOT conflict with other `WeakWrite`s or
    /// `WeakRead`s.
    ///
    /// Used for container-level subscript writes (`STORE_SUBSCR`): these
    /// should conflict with C-method reads (iteration, `len()`) but two
    /// subscript writes on different keys of the same container should
    /// NOT conflict with each other.
    WeakWrite,
    /// Like `Read` but does NOT conflict with `WeakWrite`s or other
    /// `WeakRead`s.
    ///
    /// Used for `LOAD_ATTR` on mutable values: loading a container just
    /// to subscript it shouldn't conflict with subscript writes on
    /// disjoint keys, but should still conflict with C-method writes
    /// (append, clear, etc.).
    WeakRead,
}

impl AccessKind {
    pub const ALL: [Self; 4] = [Self::Read, Self::Write, Self::WeakWrite, Self::WeakRead];

    /// Whether two accesses are dependent when they touch the same object.
    pub fn conflicts(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Write, _)
                | (_, Self::Write)
                | (Self::Read, Self::WeakWrite)
                | (Self::WeakWrite, Self::Read)
        )
    }

    /// Merge two access kinds into a single kind whose conflict set is a
    /// superset of both inputs' conflict sets.  Used by trace caching and
    /// per-branch access recording when the same object is accessed with
    /// different kinds.
    ///
    /// The key improvement over the naive "if different, upgrade to Write"
    /// is that `Read` subsumes `WeakRead`: Read conflicts with {Write,
    /// WeakWrite}, WeakRead conflicts with {Write} — so Read already
    /// covers WeakRead's conflicts.  Upgrading Read+WeakRead to Write
    /// would make the thread appear to *write* the object, causing
    /// spurious sleep-set wakeups when another thread reads it.
    pub fn merge(self, other: Self) -> Self {
        if self == other {
            return self;
        }
        match (self, other) {
            // Read subsumes WeakRead (Read ⊇ WeakRead in conflict sets)
            (AccessKind::Read, AccessKind::WeakRead)
            | (AccessKind::WeakRead, AccessKind::Read) => AccessKind::Read,
            // WeakWrite subsumes WeakRead: WeakWrite conflicts with {Read, Write},
            // WeakRead conflicts with {Write}. WeakWrite's conflict set is a superset.
            (AccessKind::WeakWrite, AccessKind::WeakRead)
            | (AccessKind::WeakRead, AccessKind::WeakWrite) => AccessKind::WeakWrite,
            // Write subsumes everything
            _ => AccessKind::Write,
        }
    }
}

/// Records an access to a shared object for DPOR dependency detection.
/// The `path_id` identifies the scheduling point (position in the execution
/// sequence E), used to determine where to insert into wakeup trees.
/// The `dpor_vv` captures the happens-before state at the time of access,
/// used to check for races: e ⋖_E e' when ¬(e →_E e') (JACM'17 p.13-14).
#[derive(Clone, Debug)]
pub struct Access {
    pub path_id: usize,
    pub dpor_vv: VersionVec,
    pub thread_id: usize,
    pub origin: AccessOrigin,
}

impl Access {
    pub fn new(path_id: usize, dpor_vv: VersionVec, thread_id: usize, origin: AccessOrigin) -> Self {
        Self { path_id, dpor_vv, thread_id, origin }
    }

    /// Check if this access happens-before the given vector clock.
    /// Paper: e →_E e' (Def 3.2, JACM'17 p.12). When this returns false,
    /// the two events are concurrent and form a race (Def 3.3 p.13-14).
    pub fn happens_before(&self, later_vv: &VersionVec) -> bool {
        self.dpor_vv.partial_le(later_vv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weak_write_weak_read_merge() {
        // WeakWrite conflicts with {Read, Write}; WeakRead conflicts with {Write}.
        // WeakWrite's conflict set is a superset of WeakRead's, so WeakWrite
        // subsumes WeakRead — merging should produce WeakWrite, not Write.
        assert_eq!(AccessKind::WeakWrite.merge(AccessKind::WeakRead), AccessKind::WeakWrite);
        assert_eq!(AccessKind::WeakRead.merge(AccessKind::WeakWrite), AccessKind::WeakWrite);
    }

    #[test]
    fn test_merge_same_kind() {
        assert_eq!(AccessKind::Read.merge(AccessKind::Read), AccessKind::Read);
        assert_eq!(AccessKind::Write.merge(AccessKind::Write), AccessKind::Write);
        assert_eq!(AccessKind::WeakRead.merge(AccessKind::WeakRead), AccessKind::WeakRead);
        assert_eq!(AccessKind::WeakWrite.merge(AccessKind::WeakWrite), AccessKind::WeakWrite);
    }

    #[test]
    fn test_read_subsumes_weak_read() {
        assert_eq!(AccessKind::Read.merge(AccessKind::WeakRead), AccessKind::Read);
        assert_eq!(AccessKind::WeakRead.merge(AccessKind::Read), AccessKind::Read);
    }

    #[test]
    fn test_write_subsumes_everything() {
        assert_eq!(AccessKind::Read.merge(AccessKind::Write), AccessKind::Write);
        assert_eq!(AccessKind::Write.merge(AccessKind::Read), AccessKind::Write);
        assert_eq!(AccessKind::WeakWrite.merge(AccessKind::Read), AccessKind::Write);
        assert_eq!(AccessKind::Read.merge(AccessKind::WeakWrite), AccessKind::Write);
    }

    #[test]
    fn test_origin_merge_strength_order() {
        assert_eq!(AccessOrigin::PythonMemory.merge(AccessOrigin::IoDirect), AccessOrigin::IoDirect);
        assert_eq!(AccessOrigin::IoDirect.merge(AccessOrigin::PythonMemory), AccessOrigin::IoDirect);
        assert_eq!(AccessOrigin::PythonMemory.merge(AccessOrigin::LockSynthetic), AccessOrigin::LockSynthetic);
        assert_eq!(AccessOrigin::LockSynthetic.merge(AccessOrigin::IoDirect), AccessOrigin::IoDirect);
    }

    #[test]
    fn test_conflict_matrix_is_symmetric_and_complete() {
        for left in AccessKind::ALL {
            for right in AccessKind::ALL {
                assert_eq!(left.conflicts(right), right.conflicts(left));
                let expected = left == AccessKind::Write
                    || right == AccessKind::Write
                    || matches!(
                        (left, right),
                        (AccessKind::Read, AccessKind::WeakWrite)
                            | (AccessKind::WeakWrite, AccessKind::Read)
                    );
                assert_eq!(left.conflicts(right), expected, "{left:?} vs {right:?}");
            }
        }
    }
}
