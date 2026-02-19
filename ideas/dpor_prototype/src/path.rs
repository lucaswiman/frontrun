//! Exploration tree for DPOR.
//!
//! The exploration tree is the central data structure driving exhaustive search.
//! Each node represents a scheduling choice point (which thread to run next).
//! DPOR prunes branches by only adding backtrack points where reordering
//! dependent operations would produce different results.
//!
//! Modeled after loom's `rt::path::Path` and `Schedule` branch type.

use crate::thread::ThreadStatus;

/// A single branch point in the exploration tree.
///
/// At each branch, the scheduler chose one thread to run. The `threads` array
/// records which threads were available and their exploration status.
#[derive(Clone, Debug)]
pub struct Branch {
    /// Per-thread scheduling status at this branch point.
    pub threads: Vec<ThreadStatus>,

    /// Which thread was (or will be) selected to run at this branch.
    pub active_thread: usize,

    /// Number of preemptions so far on the path leading to this branch.
    pub preemptions: u32,
}

impl Branch {
    /// Create a new branch with the given thread states and chosen thread.
    pub fn new(threads: Vec<ThreadStatus>, active_thread: usize, preemptions: u32) -> Self {
        Self {
            threads,
            active_thread,
            preemptions,
        }
    }

    /// Returns the set of threads that haven't been fully explored at this branch.
    pub fn unexplored_threads(&self) -> Vec<usize> {
        self.threads
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s, ThreadStatus::Backtrack))
            .map(|(i, _)| i)
            .collect()
    }
}

/// The exploration tree: a sequence of branches representing one execution path,
/// with backtrack information for DFS exploration of alternatives.
///
/// During execution, branches are appended as scheduling decisions are made.
/// After execution completes, `step()` backtracks to find the next unexplored
/// alternative.
pub struct Path {
    /// The sequence of branch points.
    branches: Vec<Branch>,

    /// Current position during replay. Branches up to `pos` are replayed
    /// (using the recorded active_thread), and new branches are appended after.
    pos: usize,

    /// Optional preemption bound.
    preemption_bound: Option<u32>,
}

impl Path {
    pub fn new(preemption_bound: Option<u32>) -> Self {
        Self {
            branches: Vec::new(),
            pos: 0,
            preemption_bound,
        }
    }

    /// Current position in the exploration tree (number of branches so far
    /// in this execution).
    pub fn current_position(&self) -> usize {
        self.pos
    }

    /// Total branches in the current/last path.
    pub fn depth(&self) -> usize {
        self.branches.len()
    }

    /// Called when the scheduler needs to pick a thread at a scheduling point.
    ///
    /// If we're replaying (pos < len), return the previously recorded choice.
    /// Otherwise, pick an enabled thread and record a new branch.
    ///
    /// `runnable` is the list of thread IDs that can be scheduled.
    /// `current_thread` is the currently active thread.
    /// `num_threads` is the total number of threads (for sizing the status vector).
    ///
    /// Returns the chosen thread ID, or None if no thread is runnable (deadlock).
    pub fn schedule(
        &mut self,
        runnable: &[usize],
        current_thread: usize,
        num_threads: usize,
    ) -> Option<usize> {
        if runnable.is_empty() {
            return None;
        }

        if self.pos < self.branches.len() {
            // Replaying: follow the recorded path
            let chosen = self.branches[self.pos].active_thread;
            self.pos += 1;
            return Some(chosen);
        }

        // New branch: prefer the current thread to minimize preemptions
        let chosen = if runnable.contains(&current_thread) {
            current_thread
        } else {
            runnable[0]
        };

        // Compute preemption count
        let is_preemption = chosen != current_thread && runnable.contains(&current_thread);
        let prev_preemptions = if let Some(last) = self.branches.last() {
            last.preemptions
        } else {
            0
        };
        let preemptions = if is_preemption {
            prev_preemptions + 1
        } else {
            prev_preemptions
        };

        let mut threads = vec![ThreadStatus::Disabled; num_threads];
        for &tid in runnable {
            threads[tid] = if tid == chosen {
                ThreadStatus::Active
            } else {
                ThreadStatus::Pending
            };
        }

        self.branches.push(Branch::new(threads, chosen, preemptions));
        self.pos += 1;

        Some(chosen)
    }

    /// Mark `thread_id` for backtracking at branch `path_id`.
    ///
    /// This is the core DPOR operation: when we detect concurrent dependent
    /// accesses, we mark the current thread for exploration at the earlier
    /// branch point where the conflict occurred.
    pub fn backtrack(&mut self, path_id: usize, thread_id: usize) {
        if path_id >= self.branches.len() {
            return;
        }

        let branch = &mut self.branches[path_id];

        match branch.threads.get(thread_id).copied() {
            Some(ThreadStatus::Pending) | Some(ThreadStatus::Yield) => {
                // Check preemption bound
                if let Some(bound) = self.preemption_bound {
                    // Would this backtrack exceed the preemption bound?
                    // A backtrack to a different thread than the currently active
                    // one at this branch counts as a preemption.
                    if branch.active_thread != thread_id && branch.preemptions >= bound {
                        // Try to find an earlier branch where we can add the
                        // backtrack within bounds (conservative fallback).
                        self.add_conservative_backtrack(path_id, thread_id, bound);
                        return;
                    }
                }
                branch.threads[thread_id] = ThreadStatus::Backtrack;
            }
            _ => {
                // Already active, visited, backtracked, or disabled: no change
            }
        }
    }

    /// Advance to the next unexplored execution path.
    ///
    /// Walks backward through branches, looking for a branch with a thread
    /// marked `Backtrack`. When found, truncates the path to that point,
    /// sets the backtrack thread as active, and returns `true`.
    /// Returns `false` when all paths are exhausted.
    pub fn step(&mut self) -> bool {
        while let Some(branch) = self.branches.last_mut() {
            // Mark the current active thread as visited
            let active = branch.active_thread;
            if active < branch.threads.len()
                && branch.threads[active] == ThreadStatus::Active
            {
                branch.threads[active] = ThreadStatus::Visited;
            }

            // Look for a thread marked for backtracking
            if let Some(next) = branch
                .threads
                .iter()
                .position(|s| *s == ThreadStatus::Backtrack)
            {
                branch.threads[next] = ThreadStatus::Active;
                branch.active_thread = next;
                // Reset replay position so the next execution replays the
                // entire prefix from the beginning, including this backtracked
                // branch. Everything after this branch will be created fresh.
                self.pos = 0;
                return true;
            }

            // No more alternatives at this branch: pop and continue backtracking
            self.branches.pop();
        }

        false // All paths exhausted
    }

    /// Conservative backtrack for preemption bounding.
    ///
    /// When a backtrack point would exceed the preemption bound, walk backward
    /// to find an earlier branch where the thread can be explored within the bound.
    fn add_conservative_backtrack(
        &mut self,
        path_id: usize,
        thread_id: usize,
        bound: u32,
    ) {
        // Walk backward from path_id to find a branch where:
        // 1. The thread is available (Pending or Yield)
        // 2. Scheduling it wouldn't exceed the preemption bound
        for i in (0..path_id).rev() {
            let branch = &mut self.branches[i];
            if let Some(status) = branch.threads.get(thread_id) {
                let would_preempt =
                    branch.active_thread != thread_id && status.is_runnable();
                if matches!(status, ThreadStatus::Pending | ThreadStatus::Yield)
                    && (!would_preempt || branch.preemptions < bound)
                {
                    branch.threads[thread_id] = ThreadStatus::Backtrack;
                    return;
                }
            }
        }
        // If no suitable earlier branch found, skip this backtrack point.
        // This is a trade-off: we may miss some interleavings, but we
        // stay within the preemption bound.
    }

    /// Get a reference to the branches (for inspection/testing).
    pub fn branches(&self) -> &[Branch] {
        &self.branches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_path() {
        let path = Path::new(None);
        assert_eq!(path.depth(), 0);
        assert_eq!(path.current_position(), 0);
    }

    #[test]
    fn test_schedule_first_branch() {
        let mut path = Path::new(None);
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(0)); // Prefers current thread
        assert_eq!(path.depth(), 1);
    }

    #[test]
    fn test_schedule_prefers_current_thread() {
        let mut path = Path::new(None);
        let chosen = path.schedule(&[0, 1, 2], 1, 3);
        assert_eq!(chosen, Some(1)); // Current thread is 1
    }

    #[test]
    fn test_schedule_falls_back_to_first_runnable() {
        let mut path = Path::new(None);
        // Current thread 2 is not in the runnable set
        let chosen = path.schedule(&[0, 1], 2, 3);
        assert_eq!(chosen, Some(0));
    }

    #[test]
    fn test_schedule_empty_runnable_is_deadlock() {
        let mut path = Path::new(None);
        let chosen = path.schedule(&[], 0, 2);
        assert_eq!(chosen, None);
    }

    #[test]
    fn test_backtrack_marks_thread() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2); // Branch 0: chose thread 0

        // Mark thread 1 for backtracking at branch 0
        path.backtrack(0, 1);

        assert_eq!(
            path.branches()[0].threads[1],
            ThreadStatus::Backtrack
        );
    }

    #[test]
    fn test_step_finds_backtrack_point() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2); // Branch 0: chose thread 0

        // Mark thread 1 for backtracking at branch 0
        path.backtrack(0, 1);

        // Step should find the backtrack point
        assert!(path.step());
        assert_eq!(path.branches()[0].active_thread, 1);
        assert_eq!(
            path.branches()[0].threads[1],
            ThreadStatus::Active
        );
    }

    #[test]
    fn test_step_exhausted() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2);

        // No backtrack points -> exhausted
        assert!(!path.step());
    }

    #[test]
    fn test_replay_follows_recorded_path() {
        let mut path = Path::new(None);
        // First execution: branch at [0,1], chose 0
        path.schedule(&[0, 1], 0, 2);
        // Mark thread 1 for backtracking
        path.backtrack(0, 1);

        // Step to next execution
        assert!(path.step());

        // Reset pos for replay
        // The path should now replay: at branch 0, choose thread 1
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1)); // Replays the backtracked choice
    }

    #[test]
    fn test_full_exploration_two_threads() {
        // Two threads, one branch point: should explore both orderings
        let mut path = Path::new(None);
        let mut executions = Vec::new();

        // First execution
        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(vec![chosen]);
        path.backtrack(0, 1); // Mark thread 1 for backtracking

        // Step to next execution
        assert!(path.step());

        // Second execution (replaying)
        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(vec![chosen]);

        // No more paths
        assert!(!path.step());

        assert_eq!(executions, vec![vec![0], vec![1]]);
    }

    #[test]
    fn test_preemption_bounding() {
        let mut path = Path::new(Some(0)); // No preemptions allowed
        path.schedule(&[0, 1], 0, 2); // Branch 0: chose thread 0

        // Try to backtrack thread 1 at branch 0 - this would be a preemption
        // since thread 0 (the active thread) != thread 1
        path.backtrack(0, 1);

        // With bound=0, the backtrack should be suppressed
        // (thread 1 shouldn't be marked as Backtrack at branch 0
        // because scheduling it there would exceed the preemption bound)
        // Since there are no earlier branches, it can't do a conservative
        // backtrack either, so step should fail.
        assert!(!path.step());
    }
}
