//! Wakeup tree data structure for Optimal DPOR.
//!
//! A wakeup tree is an ordered tree of process sequences that controls which
//! interleavings to explore next. Each branch represents a sequence of thread
//! choices that reverses a detected race. The tree ensures that no sleep-set-blocked
//! explorations occur, achieving optimality (one execution per Mazurkiewicz trace).
//!
//! **Paper ref**: Definition 6.1 (JACM'17 p.21-22, POPL'14 p.6-7).
//! A wakeup tree after ⟨E, P⟩ is an ordered tree ⟨B, ≺⟩ satisfying:
//!   Property 1: WI[E](w) ∩ P = ∅ for each leaf w  (no sleeping weak initial)
//!   Property 2: for siblings u.p ≺ u.w where u.w is a leaf, p ∉ WI[E.u](w)
//!               (p will be removed from sleep set by exploring u.w)
//!
//! **Current status**: The tree structure is fully implemented, but the insert
//! operation uses exact prefix matching rather than the paper's equivalence
//! checking (v ∼[E] w, Lemma 6.2 p.22): a sequence is treated as covered only
//! when an existing leaf is a literal prefix of it. This is sound but may
//! insert redundant branches. See `ideas/optimal_dpor.md` Phase 4c.

/// A node in the wakeup tree. Children are ordered (exploration order matters).
/// The ordering ≺ between siblings determines sleep set removal:
/// when u.p is explored before u.q (p ≺ q), p is added to the sleep set,
/// and u.q must remove p from the sleep set (JACM'17 Def 6.1 Property 2).
#[derive(Clone, Debug)]
struct WakeupNode {
    thread_id: usize,
    children: Vec<WakeupNode>,
}

/// Wakeup tree: an ordered tree of thread-id sequences.
///
/// The tree is rooted at an implicit root node. Each path from root to a leaf
/// represents a wakeup sequence — an initial fragment of an execution that
/// reverses a detected race without encountering sleep-set blocking.
#[derive(Clone, Debug)]
pub struct WakeupTree {
    /// Children of the implicit root node, in exploration order.
    children: Vec<WakeupNode>,
}

impl WakeupTree {
    /// Create an empty wakeup tree (no branches to explore).
    pub fn empty() -> Self {
        Self { children: Vec::new() }
    }

    /// Create a wakeup tree with a single branch (one thread to explore).
    pub fn singleton(thread_id: usize) -> Self {
        Self {
            children: vec![WakeupNode {
                thread_id,
                children: Vec::new(),
            }],
        }
    }

    /// Returns true if this tree has no branches to explore.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Returns the thread IDs at the root level (first steps of each branch).
    pub fn root_threads(&self) -> Vec<usize> {
        self.children.iter().map(|n| n.thread_id).collect()
    }

    /// Get the first (leftmost) thread to explore. Returns None if empty.
    pub fn first_thread(&self) -> Option<usize> {
        self.children.first().map(|n| n.thread_id)
    }

    /// Get the minimum thread ID among root-level branches.
    /// This is used for exploration ordering (lowest-index first).
    ///
    /// In Algorithm 2 line 15 (JACM'17 p.24), the paper picks min≺{p ∈ wut(E)}.
    /// We use min thread ID as a deterministic proxy for the ≺ ordering.
    pub fn min_thread(&self) -> Option<usize> {
        self.children.iter().map(|n| n.thread_id).min()
    }

    /// Extract the subtree rooted at the first child with the given thread_id.
    /// This is used when descending into a chosen thread's exploration.
    ///
    /// Corresponds to Algorithm 2 line 17 (JACM'17 p.24):
    ///   WuT' = subtree(wut(E), p)
    /// The subtree guides the next level of exploration, ensuring multi-step
    /// wakeup sequences are followed correctly.
    pub fn subtree(&self, thread_id: usize) -> WakeupTree {
        for child in &self.children {
            if child.thread_id == thread_id {
                return WakeupTree {
                    children: child.children.clone(),
                };
            }
        }
        WakeupTree::empty()
    }

    /// Remove the branch starting with the given thread_id from this tree
    /// (called after exploring that thread). Also returns whether the thread
    /// was found.
    ///
    /// Corresponds to Algorithm 2 line 19 (JACM'17 p.25):
    ///   "remove all sequences of form p.w from wut(E)"
    pub fn remove_branch(&mut self, thread_id: usize) -> bool {
        let len_before = self.children.len();
        self.children.retain(|n| n.thread_id != thread_id);
        self.children.len() < len_before
    }

    /// Insert a wakeup sequence into the tree.
    ///
    /// The sequence is a list of thread IDs representing a path that must be
    /// explored to reverse a detected race. The sequence is inserted respecting
    /// the tree structure: shared prefixes are merged, and new branches are
    /// added at the end (rightmost position) to maintain exploration order.
    ///
    /// Corresponds to Algorithm 2 line 6 (JACM'17 p.24):
    ///   wut(E') := insert[E'](v, wut(E'))
    /// where v = notdep(e, E).e' is the race-reversing sequence.
    ///
    /// **Simplification**: The paper's insert (Def 6.2, JACM'17 p.22) checks
    /// whether an existing branch already covers the sequence via the ∼[E]
    /// equivalence relation (Lemma 6.2). Our implementation only checks exact
    /// prefix matching — sound but may keep redundant branches.
    ///
    /// Returns true if a new leaf was added.
    pub fn insert(&mut self, sequence: &[usize]) -> bool {
        if sequence.is_empty() {
            return false;
        }
        self.insert_at(sequence)
    }

    /// Internal recursive insertion.
    fn insert_at(&mut self, sequence: &[usize]) -> bool {
        if sequence.is_empty() {
            return false;
        }

        let thread_id = sequence[0];
        let rest = &sequence[1..];

        // Check if this thread already exists as a child
        for child in &mut self.children {
            if child.thread_id == thread_id {
                // Merge: descend into the existing child
                if rest.is_empty() {
                    // The sequence ends here; this node already exists
                    return false;
                }
                if child.children.is_empty() {
                    // An existing *leaf* u with u a prefix of the new sequence
                    // w already covers w (Def 6.2: u ⊑[E] w): exploring E.u
                    // runs the scheduler freely from there and the races
                    // found on the way re-derive any needed continuation.
                    // Growing the leaf instead would turn "explore u, then
                    // run free" into "explore u, then force w's tail",
                    // overriding the current-thread preference and
                    // replaying an already-explored trace (N threads on one
                    // lock went from N! to more executions once lock events
                    // were attributed to the step they happen in).
                    return false;
                }
                let mut subtree = WakeupTree {
                    children: std::mem::take(&mut child.children),
                };
                let added = subtree.insert_at(rest);
                child.children = subtree.children;
                return added;
            }
        }

        // Thread not found among children: add new branch.
        // The paper (Def 6.2, JACM'17 p.22) would check if any existing branch
        // "covers" this sequence via v ∼[E] w (Lemma 6.2). We skip that check
        // and always add — sound but not optimal. The sleep set check at the
        // call site (insert_wakeup()) already filters some redundant inserts.
        let mut node = WakeupNode {
            thread_id,
            children: Vec::new(),
        };

        // Build the rest of the chain
        let mut current = &mut node;
        for &tid in rest {
            current.children.push(WakeupNode {
                thread_id: tid,
                children: Vec::new(),
            });
            current = current.children.last_mut().unwrap();
        }

        self.children.push(node);
        true
    }

    /// Check if the tree contains any branch starting with the given thread.
    pub fn contains_thread(&self, thread_id: usize) -> bool {
        self.children.iter().any(|n| n.thread_id == thread_id)
    }

    /// Get all leaf sequences (for debugging/testing).
    pub fn leaf_sequences(&self) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        for child in &self.children {
            let mut path = vec![child.thread_id];
            Self::collect_leaves(child, &mut path, &mut result);
        }
        result
    }

    fn collect_leaves(node: &WakeupNode, path: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if node.children.is_empty() {
            result.push(path.clone());
        } else {
            for child in &node.children {
                path.push(child.thread_id);
                Self::collect_leaves(child, path, result);
                path.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = WakeupTree::empty();
        assert!(tree.is_empty());
        assert_eq!(tree.first_thread(), None);
        assert!(tree.root_threads().is_empty());
    }

    #[test]
    fn test_singleton_tree() {
        let tree = WakeupTree::singleton(0);
        assert!(!tree.is_empty());
        assert_eq!(tree.first_thread(), Some(0));
        assert_eq!(tree.root_threads(), vec![0]);
    }

    #[test]
    fn test_insert_single_sequence() {
        let mut tree = WakeupTree::empty();
        assert!(tree.insert(&[1, 2, 3]));
        assert_eq!(tree.leaf_sequences(), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn test_insert_shared_prefix() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[1, 2, 3]);
        tree.insert(&[1, 2, 4]);
        let leaves = tree.leaf_sequences();
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0], vec![1, 2, 3]);
        assert_eq!(leaves[1], vec![1, 2, 4]);
    }

    #[test]
    fn test_insert_disjoint_branches() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);
        tree.insert(&[1, 0]);
        assert_eq!(tree.root_threads(), vec![0, 1]);
        let leaves = tree.leaf_sequences();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn test_insert_extension_of_leaf_is_covered() {
        let mut tree = WakeupTree::empty();
        assert!(tree.insert(&[2]));
        // [2] is a leaf: the longer sequence starting with it is already
        // covered and must not grow the leaf into a forced path.
        assert!(!tree.insert(&[2, 2, 2, 0]));
        assert_eq!(tree.leaf_sequences(), vec![vec![2]]);
        assert!(tree.subtree(2).is_empty());
        // A genuinely internal node still merges new siblings beneath it.
        assert!(tree.insert(&[1, 0]));
        assert!(tree.insert(&[1, 2]));
        assert_eq!(tree.subtree(1).root_threads(), vec![0, 2]);
    }

    #[test]
    fn test_insert_duplicate_does_not_add() {
        let mut tree = WakeupTree::empty();
        assert!(tree.insert(&[1, 2]));
        assert!(!tree.insert(&[1, 2]));
        assert_eq!(tree.leaf_sequences().len(), 1);
    }

    #[test]
    fn test_subtree() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1, 2]);
        tree.insert(&[0, 3]);
        let sub = tree.subtree(0);
        assert_eq!(sub.root_threads(), vec![1, 3]);
    }

    #[test]
    fn test_remove_branch() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);
        tree.insert(&[1, 0]);
        assert!(tree.remove_branch(0));
        assert_eq!(tree.root_threads(), vec![1]);
        assert!(!tree.remove_branch(0));
    }

    #[test]
    fn test_contains_thread() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);
        tree.insert(&[2, 3]);
        assert!(tree.contains_thread(0));
        assert!(tree.contains_thread(2));
        assert!(!tree.contains_thread(1));
    }

    #[test]
    fn test_complex_tree_structure() {
        // Simulates the wakeup tree from Fig. 7 in the paper:
        // Root has children: p(0), r(2)
        // p(0) has children: q(1), r(2)
        // p.r(0,2) has children: r(2) [i.e., p.r.r]
        // r(2) has children: r(2) [i.e., r.r]
        // r.r(2,2) has children: q(1) [i.e., r.r.q]
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);       // p.q
        tree.insert(&[0, 2, 2]);    // p.r.r
        tree.insert(&[2, 2, 1]);    // r.r.q

        let leaves = tree.leaf_sequences();
        assert_eq!(leaves.len(), 3);
        assert_eq!(leaves[0], vec![0, 1]);
        assert_eq!(leaves[1], vec![0, 2, 2]);
        assert_eq!(leaves[2], vec![2, 2, 1]);
    }
}
