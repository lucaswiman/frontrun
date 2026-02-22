//! Thread state tracking for DPOR.

use crate::vv::VersionVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadStatus {
    Disabled,
    Pending,
    Backtrack,
    Yield,
    Active,
    Visited,
    Blocked,
}

impl ThreadStatus {
    pub fn is_runnable(self) -> bool {
        matches!(
            self,
            ThreadStatus::Pending | ThreadStatus::Backtrack | ThreadStatus::Yield | ThreadStatus::Active
        )
    }
}

#[derive(Clone, Debug)]
pub struct Thread {
    pub id: usize,
    pub causality: VersionVec,
    pub dpor_vv: VersionVec,
    pub finished: bool,
    pub blocked: bool,
}

impl Thread {
    pub fn new(id: usize, num_threads: usize) -> Self {
        Self {
            id,
            causality: VersionVec::new(num_threads),
            dpor_vv: VersionVec::new(num_threads),
            finished: false,
            blocked: false,
        }
    }

    pub fn is_runnable(&self) -> bool {
        !self.finished && !self.blocked
    }
}
