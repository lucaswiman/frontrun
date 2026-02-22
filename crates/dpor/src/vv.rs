//! Vector clock implementation for happens-before tracking.

/// A vector clock indexed by thread ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionVec {
    clocks: Vec<u32>,
}

impl VersionVec {
    pub fn new(num_threads: usize) -> Self {
        Self {
            clocks: vec![0; num_threads],
        }
    }

    pub fn len(&self) -> usize {
        self.clocks.len()
    }

    pub fn get(&self, thread_id: usize) -> u32 {
        if thread_id < self.clocks.len() {
            self.clocks[thread_id]
        } else {
            0
        }
    }

    pub fn set(&mut self, thread_id: usize, value: u32) {
        if thread_id >= self.clocks.len() {
            self.clocks.resize(thread_id + 1, 0);
        }
        self.clocks[thread_id] = value;
    }

    pub fn increment(&mut self, thread_id: usize) {
        if thread_id >= self.clocks.len() {
            self.clocks.resize(thread_id + 1, 0);
        }
        self.clocks[thread_id] += 1;
    }

    /// Point-wise maximum: self = max(self, other).
    pub fn join(&mut self, other: &VersionVec) {
        if other.clocks.len() > self.clocks.len() {
            self.clocks.resize(other.clocks.len(), 0);
        }
        for (a, b) in self.clocks.iter_mut().zip(other.clocks.iter()) {
            *a = (*a).max(*b);
        }
    }

    /// Returns true if self <= other (component-wise).
    pub fn partial_le(&self, other: &VersionVec) -> bool {
        let max_len = self.clocks.len().max(other.clocks.len());
        for i in 0..max_len {
            let a = if i < self.clocks.len() { self.clocks[i] } else { 0 };
            let b = if i < other.clocks.len() { other.clocks[i] } else { 0 };
            if a > b {
                return false;
            }
        }
        true
    }

    /// Returns true if neither self <= other nor other <= self.
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
            write!(f, "{c}")?;
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
        assert_eq!(vv.get(1), 1);
        vv.increment(1);
        assert_eq!(vv.get(1), 2);
        assert_eq!(vv.get(0), 0);
    }

    #[test]
    fn test_join() {
        let mut a = VersionVec::new(3);
        a.set(0, 2);
        a.set(1, 1);
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
    fn test_partial_le() {
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
    fn test_concurrent() {
        let mut a = VersionVec::new(2);
        a.set(0, 2);
        a.set(1, 1);
        let mut b = VersionVec::new(2);
        b.set(0, 1);
        b.set(1, 2);
        assert!(a.concurrent_with(&b));
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
    }
}
