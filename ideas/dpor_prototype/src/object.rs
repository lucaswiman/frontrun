//! Shared object tracking for DPOR.
//!
//! Tracks the last access to each shared object so we can detect dependencies
//! (conflicting accesses) between threads.
//!
//! Unlike loom (which tracks atomic cells, mutexes, etc. as distinct types),
//! frontrun tracks Python-level shared state via opaque integer IDs.

use crate::access::{Access, AccessKind};

/// Identifies a shared memory location.
///
/// In the full implementation, this would be a rich enum (Attr, DictKey,
/// ListIndex, etc.). For the prototype, we use opaque integer IDs.
pub type ObjectId = u64;

/// Tracks the last accesses to a shared object for DPOR.
///
/// Determines which pairs of accesses are "dependent" (conflicting):
/// - Read/Read: independent (two reads don't conflict)
/// - Read/Write: dependent
/// - Write/Read: dependent
/// - Write/Write: dependent
#[derive(Clone, Debug)]
pub struct ObjectState {
    /// Last access of any kind (read or write).
    pub last_access: Option<Access>,

    /// Last write access. Read-only accesses don't update this.
    pub last_write_access: Option<Access>,
}

impl ObjectState {
    pub fn new() -> Self {
        Self {
            last_access: None,
            last_write_access: None,
        }
    }

    /// Returns the last dependent access for the given access kind.
    ///
    /// - A Read depends on the last Write (reads are independent of each other).
    /// - A Write depends on any prior access (read or write).
    pub fn last_dependent_access(&self, kind: AccessKind) -> Option<&Access> {
        match kind {
            AccessKind::Read => self.last_write_access.as_ref(),
            AccessKind::Write => self.last_access.as_ref(),
        }
    }

    /// Record a new access, updating tracking state.
    pub fn record_access(&mut self, access: Access, kind: AccessKind) {
        if kind == AccessKind::Write {
            self.last_write_access = Some(access.clone());
        }
        self.last_access = Some(access);
    }
}

impl Default for ObjectState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vv::VersionVec;

    #[test]
    fn test_new_has_no_accesses() {
        let state = ObjectState::new();
        assert!(state.last_access.is_none());
        assert!(state.last_write_access.is_none());
    }

    #[test]
    fn test_read_depends_on_last_write() {
        let mut state = ObjectState::new();
        let vv = VersionVec::new(2);
        let write_access = Access::new(0, vv.clone(), 0);
        state.record_access(write_access, AccessKind::Write);

        // A read should depend on the last write
        assert!(state.last_dependent_access(AccessKind::Read).is_some());
    }

    #[test]
    fn test_read_does_not_depend_on_last_read() {
        let mut state = ObjectState::new();
        let vv = VersionVec::new(2);
        let read_access = Access::new(0, vv.clone(), 0);
        state.record_access(read_access, AccessKind::Read);

        // Another read should not depend on the last read (reads are independent)
        assert!(state.last_dependent_access(AccessKind::Read).is_none());
    }

    #[test]
    fn test_write_depends_on_last_read() {
        let mut state = ObjectState::new();
        let vv = VersionVec::new(2);
        let read_access = Access::new(0, vv.clone(), 0);
        state.record_access(read_access, AccessKind::Read);

        // A write should depend on the last access (even a read)
        assert!(state.last_dependent_access(AccessKind::Write).is_some());
    }

    #[test]
    fn test_write_depends_on_last_write() {
        let mut state = ObjectState::new();
        let vv = VersionVec::new(2);
        let write_access = Access::new(0, vv.clone(), 0);
        state.record_access(write_access, AccessKind::Write);

        // A write should depend on the last write
        assert!(state.last_dependent_access(AccessKind::Write).is_some());
    }
}
