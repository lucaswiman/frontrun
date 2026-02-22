//! DPOR access tracking.

use crate::vv::VersionVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

/// Records an access to a shared object for DPOR dependency detection.
#[derive(Clone, Debug)]
pub struct Access {
    pub path_id: usize,
    pub dpor_vv: VersionVec,
    pub thread_id: usize,
}

impl Access {
    pub fn new(path_id: usize, dpor_vv: VersionVec, thread_id: usize) -> Self {
        Self { path_id, dpor_vv, thread_id }
    }

    pub fn happens_before(&self, later_vv: &VersionVec) -> bool {
        self.dpor_vv.partial_le(later_vv)
    }
}
