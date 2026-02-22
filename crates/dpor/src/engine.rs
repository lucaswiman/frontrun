//! The DPOR engine: orchestrates systematic exploration of interleavings.

use std::collections::HashMap;

use crate::access::{Access, AccessKind};
use crate::object::{ObjectId, ObjectState};
use crate::path::Path;
use crate::thread::Thread;
use crate::vv::VersionVec;

/// Synchronization events that affect the happens-before relation.
#[derive(Clone, Debug)]
pub enum SyncEvent {
    LockAcquire { lock_id: u64 },
    LockRelease { lock_id: u64 },
    ThreadJoin { joined_thread: usize },
    ThreadSpawn { child_thread: usize },
}

/// Per-execution state. Reset at the start of each execution.
pub struct Execution {
    pub threads: Vec<Thread>,
    pub objects: HashMap<ObjectId, ObjectState>,
    pub active_thread: usize,
    pub lock_release_vv: HashMap<u64, VersionVec>,
    pub aborted: bool,
    pub schedule_trace: Vec<usize>,
}

impl Execution {
    pub fn new(num_threads: usize) -> Self {
        Self {
            threads: (0..num_threads).map(|i| Thread::new(i, num_threads)).collect(),
            objects: HashMap::new(),
            active_thread: 0,
            lock_release_vv: HashMap::new(),
            aborted: false,
            schedule_trace: Vec::new(),
        }
    }

    pub fn finish_thread(&mut self, thread_id: usize) {
        self.threads[thread_id].finished = true;
    }

    pub fn block_thread(&mut self, thread_id: usize) {
        self.threads[thread_id].blocked = true;
    }

    pub fn unblock_thread(&mut self, thread_id: usize) {
        self.threads[thread_id].blocked = false;
    }

    pub fn runnable_threads(&self) -> Vec<usize> {
        self.threads.iter().filter(|t| t.is_runnable()).map(|t| t.id).collect()
    }
}

/// The main DPOR engine.
pub struct DporEngine {
    path: Path,
    num_threads: usize,
    max_executions: Option<u64>,
    executions_completed: u64,
    max_branches: usize,
}

impl DporEngine {
    pub fn new(
        num_threads: usize,
        preemption_bound: Option<u32>,
        max_branches: usize,
        max_executions: Option<u64>,
    ) -> Self {
        Self {
            path: Path::new(preemption_bound),
            num_threads,
            max_executions,
            executions_completed: 0,
            max_branches,
        }
    }

    pub fn begin_execution(&self) -> Execution {
        Execution::new(self.num_threads)
    }

    pub fn schedule(&mut self, execution: &mut Execution) -> Option<usize> {
        let runnable = execution.runnable_threads();
        if runnable.is_empty() {
            execution.aborted = true;
            return None;
        }
        if self.path.current_position() >= self.max_branches {
            execution.aborted = true;
            return None;
        }
        let chosen = self.path.schedule(&runnable, execution.active_thread, self.num_threads)?;
        execution.threads[chosen].dpor_vv.increment(chosen);
        execution.active_thread = chosen;
        execution.schedule_trace.push(chosen);
        Some(chosen)
    }

    pub fn process_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        let current_dpor_vv = execution.threads[thread_id].dpor_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        // Check ALL dependent accesses from other threads (not just the last one).
        // This ensures 3+ thread scenarios are handled correctly.
        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_dpor_vv) {
                self.path.backtrack(prev_access.path_id, thread_id);
            }
        }

        let access = Access::new(current_path_id, current_dpor_vv, thread_id);
        object_state.record_access(access, kind);
    }

    pub fn process_sync(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        event: SyncEvent,
    ) {
        match event {
            SyncEvent::LockAcquire { lock_id } => {
                if let Some(release_vv) = execution.lock_release_vv.get(&lock_id) {
                    let release_vv = release_vv.clone();
                    execution.threads[thread_id].causality.join(&release_vv);
                    execution.threads[thread_id].dpor_vv.join(&release_vv);
                }
            }
            SyncEvent::LockRelease { lock_id } => {
                let vv = execution.threads[thread_id].causality.clone();
                execution.lock_release_vv.insert(lock_id, vv);
            }
            SyncEvent::ThreadJoin { joined_thread } => {
                let joined_causality = execution.threads[joined_thread].causality.clone();
                let joined_dpor_vv = execution.threads[joined_thread].dpor_vv.clone();
                execution.threads[thread_id].causality.join(&joined_causality);
                execution.threads[thread_id].dpor_vv.join(&joined_dpor_vv);
            }
            SyncEvent::ThreadSpawn { child_thread } => {
                let parent_causality = execution.threads[thread_id].causality.clone();
                let parent_dpor_vv = execution.threads[thread_id].dpor_vv.clone();
                execution.threads[child_thread].causality.join(&parent_causality);
                execution.threads[child_thread].dpor_vv.join(&parent_dpor_vv);
            }
        }
    }

    pub fn next_execution(&mut self) -> bool {
        self.executions_completed += 1;
        if let Some(max) = self.max_executions {
            if self.executions_completed >= max {
                return false;
            }
        }
        self.path.step()
    }

    pub fn executions_completed(&self) -> u64 {
        self.executions_completed
    }

    pub fn tree_depth(&self) -> usize {
        self.path.depth()
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_threads_no_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut execution = engine.begin_execution();

        let t0 = engine.schedule(&mut execution).unwrap();
        assert_eq!(t0, 0);
        engine.process_access(&mut execution, 0, 1, AccessKind::Write);
        execution.finish_thread(0);

        let t1 = engine.schedule(&mut execution).unwrap();
        assert_eq!(t1, 1);
        engine.process_access(&mut execution, 1, 2, AccessKind::Write);
        execution.finish_thread(1);

        assert!(!engine.next_execution());
        assert_eq!(engine.executions_completed(), 1);
    }

    #[test]
    fn test_two_threads_write_write_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();
            let first = engine.schedule(&mut execution).unwrap();
            engine.process_access(&mut execution, first, 1, AccessKind::Write);
            execution.finish_thread(first);

            let second = engine.schedule(&mut execution).unwrap();
            engine.process_access(&mut execution, second, 1, AccessKind::Write);
            execution.finish_thread(second);

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        assert_eq!(exec_count, 2);
    }

    #[test]
    fn test_read_read_no_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut execution = engine.begin_execution();

        let first = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, first, 1, AccessKind::Read);
        execution.finish_thread(first);

        let second = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, second, 1, AccessKind::Read);
        execution.finish_thread(second);

        assert!(!engine.next_execution());
    }

    #[test]
    fn test_counter_lost_update() {
        #[derive(Clone, Debug)]
        struct State {
            counter: i64,
            local: [i64; 2],
        }

        let mut engine = DporEngine::new(2, None, 100_000, None);
        let thread_ops = vec![
            vec![(0u64, AccessKind::Read), (0, AccessKind::Write)],
            vec![(0u64, AccessKind::Read), (0, AccessKind::Write)],
        ];
        let mut found_bug = false;

        loop {
            let mut execution = engine.begin_execution();
            let mut state = State { counter: 0, local: [0, 0] };
            let mut pcs = vec![0usize; 2];

            loop {
                for i in 0..2 {
                    if pcs[i] >= thread_ops[i].len() {
                        execution.finish_thread(i);
                    }
                }
                if execution.runnable_threads().is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                let pc = pcs[chosen];
                if pc >= thread_ops[chosen].len() {
                    break;
                }
                let (obj_id, kind) = thread_ops[chosen][pc];
                engine.process_access(&mut execution, chosen, obj_id, kind);

                // Apply the step
                match (chosen, pc) {
                    (0, 0) => state.local[0] = state.counter,
                    (0, 1) => state.counter = state.local[0] + 1,
                    (1, 0) => state.local[1] = state.counter,
                    (1, 1) => state.counter = state.local[1] + 1,
                    _ => unreachable!(),
                }
                pcs[chosen] += 1;
            }

            if state.counter != 2 {
                found_bug = true;
            }

            if !engine.next_execution() {
                break;
            }
        }

        assert!(found_bug);
    }

    #[test]
    fn test_independent_threads_one_execution() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut execution = engine.begin_execution();

        let t0 = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, t0, 0, AccessKind::Write);
        execution.finish_thread(t0);

        let t1 = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, t1, 1, AccessKind::Write);
        execution.finish_thread(t1);

        assert!(!engine.next_execution());
        assert_eq!(engine.executions_completed(), 1);
    }

    #[test]
    fn test_three_threads_write_conflict() {
        // Regression test: with the old single-last-access tracking,
        // Thread A's access would be overwritten by Thread B's, so the
        // conflict between A and C was never explored. The per-thread
        // map ensures all conflicts are detected.
        let mut engine = DporEngine::new(3, None, 1000, None);
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();

            // Each thread writes to the same object (id=1)
            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(&mut execution, chosen, 1, AccessKind::Write);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // 3 threads all writing to the same object: 3! = 6 orderings
        assert_eq!(exec_count, 6);
    }

    #[test]
    fn test_three_threads_read_write_conflict() {
        // Thread A reads, Thread B reads, Thread C writes — all same object.
        // With the old implementation, Thread A's read could be lost.
        let mut engine = DporEngine::new(3, None, 1000, None);
        let mut exec_count = 0;
        let thread_kinds = [AccessKind::Read, AccessKind::Read, AccessKind::Write];

        loop {
            let mut execution = engine.begin_execution();

            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(&mut execution, chosen, 1, thread_kinds[chosen]);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Read-Read is independent, but Read-Write and Write-Read conflict.
        // C(write) conflicts with both A(read) and B(read), so C must be
        // explored in all positions relative to both A and B.
        // Expected: 3 orderings (C before both, C between, C after both)
        assert!(exec_count >= 3, "expected at least 3 executions, got {exec_count}");
    }
}
