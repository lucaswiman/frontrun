//! Exploration tree for DPOR.

use crate::thread::ThreadStatus;

#[derive(Clone, Debug)]
pub struct Branch {
    pub threads: Vec<ThreadStatus>,
    pub active_thread: usize,
    pub preemptions: u32,
}

impl Branch {
    pub fn new(threads: Vec<ThreadStatus>, active_thread: usize, preemptions: u32) -> Self {
        Self { threads, active_thread, preemptions }
    }
}

/// The exploration tree: manages DFS over scheduling decisions.
pub struct Path {
    branches: Vec<Branch>,
    pos: usize,
    preemption_bound: Option<u32>,
}

impl Path {
    pub fn new(preemption_bound: Option<u32>) -> Self {
        Self { branches: Vec::new(), pos: 0, preemption_bound }
    }

    pub fn current_position(&self) -> usize {
        self.pos
    }

    pub fn depth(&self) -> usize {
        self.branches.len()
    }

    /// Pick which thread to run at the current scheduling point.
    /// During replay, follows the recorded path; otherwise creates a new branch.
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
            let chosen = self.branches[self.pos].active_thread;
            self.pos += 1;
            return Some(chosen);
        }

        // New branch: prefer current thread to minimize preemptions
        let chosen = if runnable.contains(&current_thread) {
            current_thread
        } else {
            runnable[0]
        };

        let is_preemption = chosen != current_thread && runnable.contains(&current_thread);
        let prev_preemptions = self.branches.last().map_or(0, |b| b.preemptions);
        let preemptions = if is_preemption { prev_preemptions + 1 } else { prev_preemptions };

        let mut threads = vec![ThreadStatus::Disabled; num_threads];
        for &tid in runnable {
            threads[tid] = if tid == chosen { ThreadStatus::Active } else { ThreadStatus::Pending };
        }

        self.branches.push(Branch::new(threads, chosen, preemptions));
        self.pos += 1;
        Some(chosen)
    }

    /// Mark thread_id for backtracking at branch path_id.
    pub fn backtrack(&mut self, path_id: usize, thread_id: usize) {
        if path_id >= self.branches.len() {
            return;
        }
        let branch = &mut self.branches[path_id];
        match branch.threads.get(thread_id).copied() {
            Some(ThreadStatus::Pending) | Some(ThreadStatus::Yield) => {
                if let Some(bound) = self.preemption_bound {
                    if branch.active_thread != thread_id && branch.preemptions >= bound {
                        self.add_conservative_backtrack(path_id, thread_id, bound);
                        return;
                    }
                }
                branch.threads[thread_id] = ThreadStatus::Backtrack;
            }
            _ => {}
        }
    }

    /// Advance to the next unexplored execution path.
    pub fn step(&mut self) -> bool {
        while let Some(branch) = self.branches.last_mut() {
            let active = branch.active_thread;
            if active < branch.threads.len() && branch.threads[active] == ThreadStatus::Active {
                branch.threads[active] = ThreadStatus::Visited;
            }

            if let Some(next) = branch.threads.iter().position(|s| *s == ThreadStatus::Backtrack) {
                branch.threads[next] = ThreadStatus::Active;
                branch.active_thread = next;
                self.pos = 0;
                return true;
            }

            self.branches.pop();
        }
        false
    }

    fn add_conservative_backtrack(&mut self, path_id: usize, thread_id: usize, bound: u32) {
        for i in (0..path_id).rev() {
            let branch = &mut self.branches[i];
            if let Some(status) = branch.threads.get(thread_id) {
                let would_preempt = branch.active_thread != thread_id && status.is_runnable();
                if matches!(status, ThreadStatus::Pending | ThreadStatus::Yield)
                    && (!would_preempt || branch.preemptions < bound)
                {
                    branch.threads[thread_id] = ThreadStatus::Backtrack;
                    return;
                }
            }
        }
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
    fn test_schedule_prefers_current_thread() {
        let mut path = Path::new(None);
        assert_eq!(path.schedule(&[0, 1], 0, 2), Some(0));
        assert_eq!(path.depth(), 1);
    }

    #[test]
    fn test_backtrack_and_step() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2);
        path.backtrack(0, 1);
        assert!(path.step());
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
    }

    #[test]
    fn test_step_exhausted() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2);
        assert!(!path.step());
    }

    #[test]
    fn test_full_exploration_two_threads() {
        let mut path = Path::new(None);
        let mut executions = Vec::new();

        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(chosen);
        path.backtrack(0, 1);

        assert!(path.step());
        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(chosen);

        assert!(!path.step());
        assert_eq!(executions, vec![0, 1]);
    }

    #[test]
    fn test_preemption_bounding_zero() {
        let mut path = Path::new(Some(0));
        path.schedule(&[0, 1], 0, 2);
        path.backtrack(0, 1);
        // With bound=0, backtrack to thread 1 is a preemption and should be suppressed
        assert!(!path.step());
    }
}
