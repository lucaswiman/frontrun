//! Thread state tracking for DPOR.
//!
//! Each thread maintains vector clocks for causality (happens-before)
//! and DPOR scheduling decisions, plus its scheduling status.
//!
//! Modeled after loom's `rt::thread::Thread`.

use crate::vv::VersionVec;

/// Scheduling status for a thread at a branch point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadStatus {
    /// Thread has not been created yet or has finished.
    Disabled,
    /// Thread is runnable but hasn't been explored at this branch.
    Pending,
    /// Thread is marked for exploration by DPOR backtracking.
    Backtrack,
    /// Thread yielded voluntarily.
    Yield,
    /// Thread is currently executing at this branch point.
    Active,
    /// Thread has been explored from this branch point.
    Visited,
    /// Thread is blocked (on a lock, condition, etc.).
    Blocked,
}

impl ThreadStatus {
    /// Returns true if this thread can be scheduled.
    pub fn is_runnable(self) -> bool {
        matches!(
            self,
            ThreadStatus::Pending
                | ThreadStatus::Backtrack
                | ThreadStatus::Yield
                | ThreadStatus::Active
        )
    }
}

/// State of a single thread within one execution.
#[derive(Clone, Debug)]
pub struct Thread {
    /// Thread identifier (dense integer, 0-indexed).
    pub id: usize,

    /// Happens-before vector clock. Updated on synchronization events.
    pub causality: VersionVec,

    /// DPOR-specific vector clock. Updated on scheduling decisions.
    /// Used to determine whether two accesses are causally ordered
    /// for the purpose of backtrack set computation.
    pub dpor_vv: VersionVec,

    /// Whether this thread has finished executing.
    pub finished: bool,

    /// Whether this thread is blocked (waiting on a sync primitive).
    pub blocked: bool,
}

impl Thread {
    /// Create a new thread with the given ID.
    pub fn new(id: usize, num_threads: usize) -> Self {
        Self {
            id,
            causality: VersionVec::new(num_threads),
            dpor_vv: VersionVec::new(num_threads),
            finished: false,
            blocked: false,
        }
    }

    /// Returns the current scheduling status of this thread.
    pub fn status(&self) -> ThreadStatus {
        if self.finished {
            ThreadStatus::Disabled
        } else if self.blocked {
            ThreadStatus::Blocked
        } else {
            ThreadStatus::Pending
        }
    }

    /// Returns true if this thread can be scheduled.
    pub fn is_runnable(&self) -> bool {
        !self.finished && !self.blocked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_thread() {
        let t = Thread::new(0, 2);
        assert_eq!(t.id, 0);
        assert!(!t.finished);
        assert!(!t.blocked);
        assert!(t.is_runnable());
    }

    #[test]
    fn test_finished_thread_not_runnable() {
        let mut t = Thread::new(0, 2);
        t.finished = true;
        assert!(!t.is_runnable());
        assert_eq!(t.status(), ThreadStatus::Disabled);
    }

    #[test]
    fn test_blocked_thread_not_runnable() {
        let mut t = Thread::new(0, 2);
        t.blocked = true;
        assert!(!t.is_runnable());
        assert_eq!(t.status(), ThreadStatus::Blocked);
    }

    #[test]
    fn test_thread_status_runnable() {
        assert!(ThreadStatus::Pending.is_runnable());
        assert!(ThreadStatus::Backtrack.is_runnable());
        assert!(ThreadStatus::Yield.is_runnable());
        assert!(ThreadStatus::Active.is_runnable());
        assert!(!ThreadStatus::Disabled.is_runnable());
        assert!(!ThreadStatus::Visited.is_runnable());
        assert!(!ThreadStatus::Blocked.is_runnable());
    }
}
