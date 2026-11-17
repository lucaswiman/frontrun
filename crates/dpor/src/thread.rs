//! Thread state tracking for DPOR.
//!
//! Each thread maintains vector clocks for happens-before tracking.
//! Paper: happens-before (→_E) is defined in Def 3.2 (JACM'17 p.12-13).
//! Vector clocks implement this efficiently: e →_E e' iff VV(e) ≤ VV(e')
//! (Section 10, JACM'17 p.34-35).

use crate::vv::VersionVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadStatus {
    Disabled,
    Pending,
    Yield,
    Active,
    Visited,
    Blocked,
}

impl ThreadStatus {
    pub fn is_runnable(self) -> bool {
        matches!(
            self,
            ThreadStatus::Pending | ThreadStatus::Yield | ThreadStatus::Active
        )
    }
}

#[derive(Clone, Debug)]
pub struct Thread {
    pub id: usize,
    /// DPOR vector clock: includes lock-based happens-before edges.
    /// Used by `process_access` for standard race detection.
    /// Paper: implements →_E from Def 3.2 (JACM'17 p.12-13).
    pub dpor_vv: VersionVec,
    /// I/O vector clock: tracks scheduling increments and thread
    /// spawn/join but NOT lock-based happens-before.  Used by
    /// `process_io_access` so that file/socket accesses from different
    /// threads always appear potentially concurrent even when they
    /// happen inside separate lock acquisitions of the same lock.
    ///
    /// This is a pragmatic extension not in the paper. The paper's
    /// Algorithms 3-4 (JACM'17 p.27-28) handle locks by tracking
    /// enabled/disabled threads; our approach instead uses a separate
    /// VV without lock edges to force exploration at lock boundaries.
    pub io_vv: VersionVec,
    pub finished: bool,
    pub blocked: bool,
    pub(crate) was_blocked: bool,
}

impl Thread {
    pub fn new(id: usize, num_threads: usize) -> Self {
        Self {
            id,
            dpor_vv: VersionVec::new(num_threads),
            io_vv: VersionVec::new(num_threads),
            finished: false,
            blocked: false,
            was_blocked: false,
        }
    }

    pub fn is_runnable(&self) -> bool {
        !self.finished && !self.blocked
    }
}
