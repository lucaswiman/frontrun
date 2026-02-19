//! DPOR access tracking.
//!
//! Records accesses to shared objects for dependency detection.
//! When thread T accesses object O, we record T's position in the exploration
//! tree and T's DPOR vector clock. When a later thread T' accesses the same
//! object, we compare clocks to determine if the accesses are causally ordered
//! or concurrent.
//!
//! Modeled after loom's `rt::access::Access`.

use crate::vv::VersionVec;

/// The kind of access to a shared object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    /// A read-only access (e.g., `x = obj.attr`).
    Read,
    /// A write access (e.g., `obj.attr = x`).
    Write,
}

/// Records an access to a shared object for DPOR dependency detection.
#[derive(Clone, Debug)]
pub struct Access {
    /// Position in the exploration tree (index into Path::branches) where
    /// this access occurred. Used to locate the branch point for backtracking.
    pub path_id: usize,

    /// DPOR vector clock of the accessing thread at the time of access.
    pub dpor_vv: VersionVec,

    /// The thread that performed this access.
    pub thread_id: usize,
}

impl Access {
    /// Create a new access record.
    pub fn new(path_id: usize, dpor_vv: VersionVec, thread_id: usize) -> Self {
        Self {
            path_id,
            dpor_vv,
            thread_id,
        }
    }

    /// Returns true if this access happens-before the given vector clock.
    /// If true, the accesses are ordered and no backtracking is needed.
    /// If false, the accesses are concurrent and DPOR must add a backtrack point.
    pub fn happens_before(&self, later_vv: &VersionVec) -> bool {
        self.dpor_vv.partial_le(later_vv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_happens_before_ordered() {
        let mut vv1 = VersionVec::new(2);
        vv1.set(0, 1);
        let access = Access::new(0, vv1, 0);

        let mut vv2 = VersionVec::new(2);
        vv2.set(0, 2);
        vv2.set(1, 1);

        assert!(access.happens_before(&vv2));
    }

    #[test]
    fn test_access_happens_before_concurrent() {
        let mut vv1 = VersionVec::new(2);
        vv1.set(0, 2);
        vv1.set(1, 0);
        let access = Access::new(0, vv1, 0);

        let mut vv2 = VersionVec::new(2);
        vv2.set(0, 0);
        vv2.set(1, 2);

        assert!(!access.happens_before(&vv2));
    }
}
