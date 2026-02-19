//! Vector clock implementation for happens-before tracking.
//!
//! A vector clock is a vector of logical timestamps, one per thread. It tracks
//! causal ordering: if `a <= b` (component-wise), then event `a` happened before
//! event `b`. If neither dominates, the events are concurrent (and potentially racing).
//!
//! Modeled after loom's `rt::vv::VersionVec`.

/// A vector clock indexed by thread ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionVec {
    /// Clock values indexed by thread ID. Thread IDs are dense integers 0..N.
    clocks: Vec<u32>,
}

impl VersionVec {
    /// Create a zero vector for `num_threads` threads.
    pub fn new(num_threads: usize) -> Self {
        Self {
            clocks: vec![0; num_threads],
        }
    }

    /// Number of threads tracked.
    pub fn len(&self) -> usize {
        self.clocks.len()
    }

    /// Get the clock value for a given thread.
    pub fn get(&self, thread_id: usize) -> u32 {
        self.clocks[thread_id]
    }

    /// Set the clock value for a given thread.
    pub fn set(&mut self, thread_id: usize, value: u32) {
        self.clocks[thread_id] = value;
    }

    /// Increment the clock for `thread_id` (tick on local event).
    pub fn increment(&mut self, thread_id: usize) {
        self.clocks[thread_id] += 1;
    }

    /// Point-wise maximum: self = max(self, other).
    /// Used for acquire synchronization and thread join.
    pub fn join(&mut self, other: &VersionVec) {
        // If other has more threads, extend self with zeros first
        if other.clocks.len() > self.clocks.len() {
            self.clocks.resize(other.clocks.len(), 0);
        }
        for (a, b) in self.clocks.iter_mut().zip(other.clocks.iter()) {
            *a = (*a).max(*b);
        }
    }

    /// Returns true if `self` happens-before-or-equal `other`.
    /// i.e., self[i] <= other[i] for all i.
    pub fn partial_le(&self, other: &VersionVec) -> bool {
        // If self has more components than other, the extra components
        // must all be zero for self <= other to hold.
        for i in 0..self.clocks.len().max(other.clocks.len()) {
            let a = if i < self.clocks.len() {
                self.clocks[i]
            } else {
                0
            };
            let b = if i < other.clocks.len() {
                other.clocks[i]
            } else {
                0
            };
            if a > b {
                return false;
            }
        }
        true
    }

    /// Returns true if `self` and `other` are concurrent
    /// (neither happens-before the other).
    pub fn concurrent_with(&self, other: &VersionVec) -> bool {
        !self.partial_le(other) && !other.partial_le(self)
    }
}

impl std::fmt::Display for VersionVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, c) in self.clocks.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", c)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_zero() {
        let vv = VersionVec::new(3);
        assert_eq!(vv.get(0), 0);
        assert_eq!(vv.get(1), 0);
        assert_eq!(vv.get(2), 0);
    }

    #[test]
    fn test_increment() {
        let mut vv = VersionVec::new(3);
        vv.increment(1);
        assert_eq!(vv.get(0), 0);
        assert_eq!(vv.get(1), 1);
        assert_eq!(vv.get(2), 0);
        vv.increment(1);
        assert_eq!(vv.get(1), 2);
    }

    #[test]
    fn test_join() {
        let mut a = VersionVec::new(3);
        a.set(0, 2);
        a.set(1, 1);
        a.set(2, 0);

        let mut b = VersionVec::new(3);
        b.set(0, 1);
        b.set(1, 3);
        b.set(2, 2);

        a.join(&b);
        assert_eq!(a.get(0), 2);
        assert_eq!(a.get(1), 3);
        assert_eq!(a.get(2), 2);
    }

    #[test]
    fn test_partial_le_equal() {
        let a = VersionVec::new(3);
        assert!(a.partial_le(&a));
    }

    #[test]
    fn test_partial_le_strictly_less() {
        let mut a = VersionVec::new(2);
        a.set(0, 1);
        a.set(1, 2);

        let mut b = VersionVec::new(2);
        b.set(0, 2);
        b.set(1, 3);

        assert!(a.partial_le(&b));
        assert!(!b.partial_le(&a));
    }

    #[test]
    fn test_partial_le_concurrent() {
        let mut a = VersionVec::new(2);
        a.set(0, 2);
        a.set(1, 1);

        let mut b = VersionVec::new(2);
        b.set(0, 1);
        b.set(1, 2);

        assert!(!a.partial_le(&b));
        assert!(!b.partial_le(&a));
    }

    #[test]
    fn test_concurrent_with() {
        let mut a = VersionVec::new(2);
        a.set(0, 2);
        a.set(1, 1);

        let mut b = VersionVec::new(2);
        b.set(0, 1);
        b.set(1, 2);

        assert!(a.concurrent_with(&b));
        assert!(b.concurrent_with(&a));
    }

    #[test]
    fn test_not_concurrent_when_ordered() {
        let mut a = VersionVec::new(2);
        a.set(0, 1);
        a.set(1, 1);

        let mut b = VersionVec::new(2);
        b.set(0, 2);
        b.set(1, 2);

        assert!(!a.concurrent_with(&b));
        assert!(!b.concurrent_with(&a));
    }

    #[test]
    fn test_join_different_sizes() {
        let mut a = VersionVec::new(2);
        a.set(0, 1);
        a.set(1, 2);

        let mut b = VersionVec::new(3);
        b.set(0, 0);
        b.set(1, 3);
        b.set(2, 1);

        a.join(&b);
        assert_eq!(a.len(), 3);
        assert_eq!(a.get(0), 1);
        assert_eq!(a.get(1), 3);
        assert_eq!(a.get(2), 1);
    }

    #[test]
    fn test_display() {
        let mut vv = VersionVec::new(3);
        vv.set(0, 1);
        vv.set(1, 2);
        vv.set(2, 3);
        assert_eq!(format!("{}", vv), "[1, 2, 3]");
    }
}
