//! The DPOR engine: orchestrates systematic exploration of interleavings.
//!
//! Implements race detection and happens-before tracking for DPOR.
//!
//! **Paper ref**: Abdulla et al., "Source Sets: A Foundation for Optimal
//! Dynamic Partial Order Reduction", JACM 2017.
//!
//! **Current algorithm**: Race detection is performed during execution
//! (Algorithm 1 style, JACM'17 p.16 lines 5-9), not deferred to maximal
//! executions (Algorithm 2 style, p.24 lines 1-6). Happens-before is tracked
//! via vector clocks (Section 10, JACM'17 p.34-35).

use std::collections::HashMap;

use crate::access::{Access, AccessKind, AccessOrigin};
use crate::object::{ObjectId, ObjectState};
use crate::path::{Path, PendingRace, SearchStrategy};
use crate::thread::Thread;
use crate::vv::VersionVec;

/// Synchronization events that affect the happens-before relation.
/// Paper: happens-before is defined in Definition 3.2 (JACM'17 p.12-13).
/// Lock acquire/release create happens-before edges; thread spawn/join
/// establish causal ordering between parent and child.
#[derive(Clone, Debug)]
pub enum SyncEvent {
    LockAcquire { lock_id: u64 },
    LockRelease { lock_id: u64 },
    /// A non-blocking (`blocking=False`) acquire *attempt*, successful or not.
    ///
    /// Unlike a blocking acquire, a failed trylock is observable behavior:
    /// the caller takes a different branch instead of waiting.  The engine
    /// must therefore explore "attempt while held" vs "attempt after
    /// release" orderings (see `process_deferred_lock_releases`).
    LockAttempt { lock_id: u64, acquired: bool },
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
///
/// Uses deferred race detection (Algorithm 2 style, JACM'17 p.24 lines 1-6):
/// races are collected during execution as `PendingRace` entries and processed
/// at maximal executions (in `next_execution()`), where the full trace is
/// available for computing notdep sequences.

/// A lock release event recorded during execution for deferred backtracking.
#[derive(Clone, Debug)]
struct DeferredLockRelease {
    thread_id: usize,
    path_id: usize,
    lock_id: u64,
}

/// A lock acquire event recorded during execution for deferred backtracking.
#[derive(Clone, Debug)]
struct DeferredLockAcquire {
    thread_id: usize,
    path_id: usize,
}

/// A non-blocking acquire attempt (trylock) recorded during execution.
///
/// Used by `process_deferred_lock_releases` to insert backtracks at the
/// release position of the same lock: reversing (release, attempt) lets the
/// attempting thread run while the lock is still held, so its trylock FAILS
/// and the failure branch (e.g. "assume the holder will drain the queue")
/// gets explored.  Without this, lost-wakeup races behind
/// `acquire(blocking=False)` are invisible to DPOR.
#[derive(Clone, Debug)]
struct DeferredLockAttempt {
    thread_id: usize,
    path_id: usize,
    lock_id: u64,
    acquired: bool,
}

pub struct DporEngine {
    pub path: Path,
    num_threads: usize,
    max_executions: Option<u64>,
    executions_completed: u64,
    max_branches: usize,
    /// Races detected during the current execution, deferred for processing.
    /// Paper ref: Algorithm 2 lines 1-6 (JACM'17 p.24) — races are collected
    /// during execution and processed at maximal executions where the full
    /// trace is available for computing notdep sequences.
    pending_races: Vec<PendingRace>,
    /// Lock releases collected during the current execution.
    /// Processed in `next_execution()` for inter-critical-section backtracking.
    deferred_lock_releases: Vec<DeferredLockRelease>,
    /// Lock acquires collected during the current execution.
    /// Used to determine whether a release should trigger backtracking.
    deferred_lock_acquires: Vec<DeferredLockAcquire>,
    /// Non-blocking acquire attempts collected during the current execution.
    /// Used by release backtracking to explore trylock-failure branches.
    deferred_lock_attempts: Vec<DeferredLockAttempt>,
    /// Resource group mapping: object_id → group_id.
    ///
    /// Objects in the same resource group (e.g., same SQL table) are related;
    /// objects in different groups are independent. When races are detected on
    /// objects with resource groups, inline wakeup insertion is skipped in
    /// favor of deferred notdep processing. This prevents backtrack explosion
    /// from intermediate operations on unrelated tables (Defect #15).
    resource_groups: HashMap<ObjectId, u64>,
    /// One-shot preference for the next newly-created scheduling branch.
    preferred_thread: Option<usize>,
}

impl DporEngine {
    /// XOR mask to derive virtual object IDs for lock acquire conflict tracking.
    const LOCK_OBJECT_XOR: u64 = 0x4C4F_434B_4C4F_434B; // "LOCKLOCK"

    pub fn new(
        num_threads: usize,
        preemption_bound: Option<u32>,
        max_branches: usize,
        max_executions: Option<u64>,
        search_strategy: SearchStrategy,
    ) -> Self {
        Self {
            path: Path::new(preemption_bound, search_strategy),
            num_threads,
            max_executions,
            executions_completed: 0,
            max_branches,
            pending_races: Vec::new(),
            deferred_lock_releases: Vec::new(),
            deferred_lock_acquires: Vec::new(),
            deferred_lock_attempts: Vec::new(),
            resource_groups: HashMap::new(),
            preferred_thread: None,
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
        let preferred = self.preferred_thread;
        let replaying = self.path.current_position() < self.path.depth();
        let chosen = self.path.schedule_preferred(&runnable, execution.active_thread, self.num_threads, preferred)?;
        if !replaying && preferred == Some(chosen) {
            self.preferred_thread = None;
        }
        execution.threads[chosen].dpor_vv.increment(chosen);
        execution.threads[chosen].io_vv.increment(chosen);
        execution.active_thread = chosen;
        execution.schedule_trace.push(chosen);
        Some(chosen)
    }

    pub fn prefer_next_thread(&mut self, thread_id: usize) {
        self.preferred_thread = Some(thread_id);
    }

    /// Report a shared memory access and detect races.
    ///
    /// Uses a **hybrid** approach: immediate inline wakeup tree insertion
    /// (Algorithm 1 style, JACM'17 p.16 lines 5-9) ensures all races add
    /// the racing thread to the wakeup tree, while deferred race collection
    /// (Algorithm 2 style, JACM'17 p.24 lines 1-6) enables notdep sequence
    /// optimization at maximal executions.
    ///
    /// The inline insertion ensures correctness (no races are dropped due to
    /// notdep feasibility issues). The deferred notdep processing adds multi-step
    /// wakeup sequences that can reduce redundant exploration by guiding through
    /// independent intermediate events.
    pub fn process_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        self.process_access_at(execution, thread_id, object_id, kind, current_path_id);
    }

    /// Like [`process_access`] but attributes the access to an explicit
    /// scheduling step. Async DPOR schedules the next task before the current
    /// await-delimited segment executes, so using the live path position would
    /// attach the segment's accesses to the following scheduling choice.
    pub fn process_access_at(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
        current_path_id: usize,
    ) {
        let current_dpor_vv = execution.threads[thread_id].dpor_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_dpor_vv) {
                // Inline wakeup insertion: immediate single-thread insertion
                // ensures the racing thread is always added (Algorithm 1 style).
                self.path.insert_wakeup(prev_access.path_id, thread_id, Some(object_id));
                // Also collect for deferred notdep processing (Algorithm 2 style).
                // The notdep sequence may provide a better multi-step wakeup
                // sequence that guides through independent intermediates.
                self.pending_races.push(PendingRace {
                    prev_path_id: prev_access.path_id,
                    current_path_id,
                    thread_id,
                    race_object: Some(object_id),
                    inline_skipped: false,
                    is_attribute_race: true,
                });
            }
        }

        let access = Access::new(current_path_id, current_dpor_vv, thread_id, AccessOrigin::PythonMemory);
        object_state.record_access(access, kind);

        // Record access for sleep set independence checks.
        self.path.record_access(current_path_id, object_id, kind, AccessOrigin::PythonMemory);
    }

    /// Like [`process_access`] but uses span recording semantics (keeps the
    /// earliest *and* latest access per thread rather than only the latest).
    /// Uses the regular `dpor_vv` for happens-before computation, so
    /// lock-based synchronization is still respected.
    ///
    /// This is useful for container-level access keys (`__cmethods__`) and
    /// for names a thread loads repeatedly (`LOAD_GLOBAL`, `LOAD_DEREF`,
    /// `FOR_ITER`).  The first position lets DPOR insert into the wakeup tree
    /// at the earliest point, enabling exploration of interleavings where
    /// another thread runs between the first and subsequent accesses; the
    /// last position keeps the trailing access racing with other threads'
    /// writes, which keeping only the first would silently drop.
    pub fn process_first_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        self.process_first_access_at(execution, thread_id, object_id, kind, current_path_id);
    }

    /// Like [`process_first_access`] but attributes the access to an explicit
    /// scheduling step.
    pub fn process_first_access_at(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
        current_path_id: usize,
    ) {
        let current_dpor_vv = execution.threads[thread_id].dpor_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_dpor_vv) {
                self.path.insert_wakeup(prev_access.path_id, thread_id, Some(object_id));
                self.pending_races.push(PendingRace {
                    prev_path_id: prev_access.path_id,
                    current_path_id,
                    thread_id,
                    race_object: Some(object_id),
                    inline_skipped: false,
                    is_attribute_race: false,
                });
            }
        }

        let access = Access::new(current_path_id, current_dpor_vv, thread_id, AccessOrigin::PythonMemory);
        object_state.record_access_span(access, kind);

        self.path.record_access(current_path_id, object_id, kind, AccessOrigin::PythonMemory);
    }

    /// Like [`process_access`] but uses `dpor_vv` (lock-aware) with
    /// `record_access_span` recording semantics.  Intended for I/O
    /// operations that go through Python-level code (Redis, SQL) where
    /// Python locks are tracked by `report_sync` and should be respected.
    ///
    /// When the object has a registered resource group (via
    /// [`register_resource_group`]), inline wakeup insertion is skipped.
    /// Races are still collected as `PendingRace` entries for deferred
    /// notdep processing at the end of the execution. This prevents
    /// backtrack explosion from intermediate operations on unrelated
    /// tables (Defect #15).
    pub fn process_synced_io_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        let current_dpor_vv = execution.threads[thread_id].dpor_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        // Check if races on this object involve intermediate accesses
        // from different resource groups. When true, we skip inline wakeup
        // and rely on deferred notdep processing to provide multi-step
        // guidance. This avoids backtrack explosion from intermediate
        // operations on unrelated tables (Defect #15).
        let has_group = self.has_resource_group(object_id);

        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_dpor_vv) {
                // Check if there are intermediate accesses from different
                // resource groups between this race pair. If so, skip inline
                // wakeup to avoid explosion. If not (same group or no
                // intermediates from other groups), do inline as normal.
                let skip_inline = has_group
                    && self.path.has_cross_group_intermediates(
                        prev_access.path_id,
                        current_path_id,
                        &self.resource_groups,
                    );
                if !skip_inline {
                    self.path.insert_wakeup(prev_access.path_id, thread_id, Some(object_id));
                }
                self.pending_races.push(PendingRace {
                    prev_path_id: prev_access.path_id,
                    current_path_id,
                    thread_id,
                    race_object: Some(object_id),
                    inline_skipped: skip_inline,
                    is_attribute_race: false,
                });
            }
        }

        let access = Access::new(current_path_id, current_dpor_vv, thread_id, AccessOrigin::IoDirect);
        object_state.record_access_span(access, kind);

        self.path.record_access(current_path_id, object_id, kind, AccessOrigin::IoDirect);
    }

    /// Like [`process_access`] but uses the thread's `io_vv` instead of
    /// `dpor_vv`.  Because `io_vv` does not include lock-based
    /// happens-before edges, I/O accesses from different threads always
    /// appear potentially concurrent — even when they occur inside
    /// separate lock acquisitions.  This lets DPOR explore interleavings
    /// around file/socket operations and catch TOCTOU races.
    pub fn process_io_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        self.process_io_access_at(execution, thread_id, object_id, kind, current_path_id, AccessOrigin::IoDirect);
    }

    /// Like [`process_io_access`] but uses a specific `path_id` and explicit
    /// `origin` instead of the current path position.  Used by [`process_sync`]
    /// for lock events which must be attributed to the thread's last scheduling
    /// point, not the live path position (which may have been advanced by
    /// another thread on free-threaded Python).
    fn process_io_access_at(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
        current_path_id: usize,
        origin: AccessOrigin,
    ) {
        let current_io_vv = execution.threads[thread_id].io_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_io_vv) {
                self.path.insert_wakeup(prev_access.path_id, thread_id, Some(object_id));
                self.pending_races.push(PendingRace {
                    prev_path_id: prev_access.path_id,
                    current_path_id,
                    thread_id,
                    race_object: Some(object_id),
                    inline_skipped: false,
                    is_attribute_race: false,
                });
            }
        }

        let access = Access::new(current_path_id, current_io_vv, thread_id, origin);
        object_state.record_access_span(access, kind);

        self.path.record_access(current_path_id, object_id, kind, origin);
    }

    /// Process a synchronization event (lock acquire/release, thread spawn/join).
    ///
    /// Updates vector clocks to establish happens-before edges.
    /// Paper: Definition 3.2 properties 1-7 (JACM'17 p.12-13) define valid
    /// happens-before assignments. Lock acquire/release create edges via
    /// vector clock joins; thread spawn/join do the same.
    ///
    /// **Lock handling**: The paper's Algorithms 3-4 (JACM'17 p.27-28) handle
    /// locks by relaxing Assumption 3.1 (processes can disable each other).
    /// Our approach is different: we use a separate `io_vv` vector clock that
    /// omits lock HB edges, making lock operations always appear concurrent
    /// for wakeup tree insertion. This is conservative (may over-explore)
    /// but catches multi-lock races that pure HB-based tracking would miss.
    /// Process a synchronization event with an explicit `path_id` for lock events.
    ///
    /// The `path_id` parameter pins lock-related I/O accesses to a specific
    /// scheduling step.  On free-threaded Python, the thread reporting a lock
    /// event may no longer be the active thread — another thread may have
    /// advanced `path.pos` via `schedule()` concurrently.  Passing the saved
    /// `path_id` from the thread's last scheduling point ensures lock events
    /// land at the correct position regardless of concurrent `pos` advances.
    ///
    /// When `path_id` is `None`, the current `path.current_position()` is used
    /// (the original behavior, correct on GIL-protected Python).
    pub fn process_sync(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        event: SyncEvent,
        path_id: Option<usize>,
    ) {
        match event {
            SyncEvent::LockAcquire { lock_id } => {
                if let Some(release_vv) = execution.lock_release_vv.get(&lock_id) {
                    let release_vv = release_vv.clone();
                    execution.threads[thread_id].dpor_vv.join(&release_vv);
                }
                let lock_obj_id = lock_id ^ Self::LOCK_OBJECT_XOR;
                let pid = path_id.unwrap_or_else(|| self.path.current_position().saturating_sub(1));
                self.process_io_access_at(
                    execution, thread_id, lock_obj_id, AccessKind::Write, pid, AccessOrigin::LockSynthetic,
                );
                // Record for deferred lock release backtracking.
                self.deferred_lock_acquires.push(DeferredLockAcquire {
                    thread_id,
                    path_id: pid,
                });
            }
            SyncEvent::LockAttempt { lock_id, acquired } => {
                let pid = path_id.unwrap_or_else(|| self.path.current_position().saturating_sub(1));
                if !acquired {
                    // A failed trylock is still an operation on the lock:
                    // give it a synthetic Write access so it races with
                    // other threads' acquires (exploring the success branch
                    // via the ordinary race-reversal machinery).  Successful
                    // attempts already get this access from the accompanying
                    // LockAcquire event.
                    let lock_obj_id = lock_id ^ Self::LOCK_OBJECT_XOR;
                    self.process_io_access_at(
                        execution, thread_id, lock_obj_id, AccessKind::Write, pid, AccessOrigin::LockSynthetic,
                    );
                }
                // Record for deferred release→attempt backtracking (both
                // outcomes): a later release of this lock must explore the
                // ordering where the attempt happens while the lock is held.
                self.deferred_lock_attempts.push(DeferredLockAttempt {
                    thread_id,
                    path_id: pid,
                    lock_id,
                    acquired,
                });
            }
            SyncEvent::LockRelease { lock_id } => {
                let vv = execution.threads[thread_id].dpor_vv.clone();
                execution.lock_release_vv.insert(lock_id, vv);
                // Publish the pre-release clock, then advance the releaser's
                // local epoch.  An acquirer joins only the published prefix;
                // accesses performed by the releaser after this release (but
                // before its next scheduler hand-off) must remain concurrent
                // with the acquirer's post-wake accesses.
                execution.threads[thread_id].dpor_vv.increment(thread_id);
                // Lock-aware DPOR: release does NOT create an io_vv access.
                //
                // Previous approach: separate virtual objects for acquire/release,
                // both as Write via io_vv.  This created TWO independent race
                // dimensions per lock, causing overcounting (e.g. 3 traces
                // instead of 2 for N=2 threads with a single lock).
                //
                // New approach: acquire-acquire races (via LOCK_OBJECT_XOR) handle
                // lock ordering.  For multi-lock bugs (where thread T releases
                // lock_a then acquires lock_b), we use DEFERRED release
                // backtracking: at the end of the execution, we check if any
                // thread released a lock then later acquired another.  If so, we
                // insert backtracks at the release position, enabling another
                // thread to interleave between the two critical sections.
                //
                // This is precise: single-lock tests get no release backtracks
                // (thread doesn't acquire after releasing), while multi-lock
                // tests get exactly the backtracks needed for bug detection.
                let pid = path_id.unwrap_or_else(|| self.path.current_position().saturating_sub(1));
                self.deferred_lock_releases.push(DeferredLockRelease {
                    thread_id,
                    path_id: pid,
                    lock_id,
                });
            }
            SyncEvent::ThreadJoin { joined_thread } => {
                let joined_dpor_vv = execution.threads[joined_thread].dpor_vv.clone();
                let joined_io_vv = execution.threads[joined_thread].io_vv.clone();
                execution.threads[thread_id].dpor_vv.join(&joined_dpor_vv);
                execution.threads[thread_id].io_vv.join(&joined_io_vv);
            }
            SyncEvent::ThreadSpawn { child_thread } => {
                let parent_dpor_vv = execution.threads[thread_id].dpor_vv.clone();
                let parent_io_vv = execution.threads[thread_id].io_vv.clone();
                execution.threads[child_thread].dpor_vv.join(&parent_dpor_vv);
                execution.threads[child_thread].io_vv.join(&parent_io_vv);
            }
        }
    }

    /// Advance to the next execution.
    ///
    /// Processes deferred races from the just-completed execution by computing
    /// notdep sequences and inserting them into wakeup trees, then steps the
    /// exploration tree to find the next unexplored path.
    ///
    /// Paper ref: Algorithm 2 (JACM'17 p.24):
    ///   Lines 1-6: process races at maximal execution → compute notdep → insert wut
    ///   Lines 14-20: step the exploration tree (pick next from wakeup tree)
    pub fn next_execution(&mut self) -> bool {
        self.executions_completed += 1;
        if let Some(max) = self.max_executions {
            if self.executions_completed >= max {
                return false;
            }
        }
        // Process deferred races: compute notdep sequences and insert into
        // wakeup trees. This must happen before step() so that the wakeup
        // trees are populated with the correct sequences.
        let races = std::mem::take(&mut self.pending_races);
        self.path.process_deferred_races(&races);

        // Lock-aware DPOR: deferred release backtracking.
        //
        // For each lock release where the releasing thread later acquires
        // another lock, insert backtrack opportunities for other runnable
        // threads at the release position.  This is the key mechanism for
        // detecting multi-lock atomicity bugs: it allows another thread to
        // run between a thread's two critical sections.
        //
        // This is DEFERRED (not inline during execution) so that we only
        // insert backtracks when we can confirm the thread continued to
        // another lock acquire.  For single-lock programs, no thread does
        // acquire-after-release, so no backtracks are inserted — giving
        // exact trace counts.
        self.process_deferred_lock_releases();

        self.path.step()
    }

    /// Process deferred lock releases for inter-critical-section backtracking.
    ///
    /// For each lock release by thread T, check if T later acquired any lock
    /// in this execution.  If so, insert backtrack opportunities at the release
    /// position for all other runnable threads, enabling exploration of
    /// interleavings between T's critical sections.
    fn process_deferred_lock_releases(&mut self) {
        let releases = std::mem::take(&mut self.deferred_lock_releases);
        let acquires = std::mem::take(&mut self.deferred_lock_acquires);
        let attempts = std::mem::take(&mut self.deferred_lock_attempts);

        for release in &releases {
            let has_later_acquire = acquires.iter().any(|acq| {
                acq.thread_id == release.thread_id && acq.path_id > release.path_id
            });
            if has_later_acquire {
                // Insert backtrack for each other runnable thread at the
                // release position.
                let pid = release.path_id;
                for tid in 0..self.num_threads {
                    if tid != release.thread_id {
                        self.path.insert_wakeup(pid, tid, None);
                    }
                }
            }

            // Trylock-aware backtracking: if another thread performed a
            // non-blocking acquire attempt of the SAME lock after this
            // release, reverse the (release, attempt) pair — schedule the
            // attempting thread at the release position, where its trylock
            // fails and the failure branch executes.  A blocking acquire
            // would merely wait there (pointless to explore, hence the
            // acquire-only rule above), but a trylock failure is observable
            // behavior: e.g. the lost-wakeup pattern
            //   q.append(x); if not lock.acquire(blocking=False): return
            // silently drops x when the attempt lands inside the holder's
            // critical section after its final queue-empty check.
            for attempt in &attempts {
                if attempt.lock_id == release.lock_id
                    && attempt.thread_id != release.thread_id
                {
                    if attempt.path_id > release.path_id {
                        // The release event is attributed to the scheduling
                        // position created *after* the holder's last traced
                        // opcode preceding the release; the release itself
                        // executes within the holder's preceding quantum.  To
                        // preempt BEFORE the release actually happens, the
                        // attempting thread must be scheduled one position
                        // earlier (instead of the holder's final in-lock
                        // opcode).
                        let pid = release.path_id.saturating_sub(1);
                        self.path.insert_wakeup(pid, attempt.thread_id, None);
                    } else if !attempt.acquired {
                        // Explored-clock timeouts are represented as a failed
                        // attempt by the clock actor.  When the timeout ran
                        // first, seed the reverse order at that exact branch:
                        // let the holder release before the deadline fires.
                        self.path.insert_wakeup(attempt.path_id, release.thread_id, None);
                    }
                }
            }
        }
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

    /// Return a snapshot of the pending races detected during the current execution.
    /// Call before `next_execution()` which consumes them.
    pub fn pending_races(&self) -> &[PendingRace] {
        &self.pending_races
    }

    /// Register that `object_id` belongs to resource group `group_id`.
    ///
    /// Objects in the same resource group (e.g., SQL resources from the same
    /// table) are related; objects in different groups are independent. When
    /// a race is detected on an object with a resource group, inline wakeup
    /// insertion is skipped in favor of deferred notdep processing. This
    /// prevents backtrack explosion from intermediate operations on unrelated
    /// tables (Defect #15: Operation Coalescing for Unrelated Tables).
    pub fn register_resource_group(&mut self, object_id: ObjectId, group_id: u64) {
        self.resource_groups.insert(object_id, group_id);
    }

    /// Check if an object has a registered resource group.
    fn has_resource_group(&self, object_id: ObjectId) -> bool {
        self.resource_groups.contains_key(&object_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preferred_thread_survives_replayed_prefix() {
        let mut engine = DporEngine::new(3, None, 100, None, SearchStrategy::Dfs);
        let mut first = engine.begin_execution();

        assert_eq!(engine.schedule(&mut first), Some(0));
        engine.process_access(&mut first, 0, 123, AccessKind::Write);
        first.block_thread(0);
        assert_eq!(engine.schedule(&mut first), Some(1));
        engine.process_access(&mut first, 1, 123, AccessKind::Write);

        assert!(engine.next_execution());
        let mut replay = engine.begin_execution();
        engine.prefer_next_thread(2);

        assert_eq!(engine.schedule(&mut replay), Some(1), "the retained prefix remains authoritative");
        assert_eq!(
            engine.schedule(&mut replay),
            Some(2),
            "the preference must apply at the first newly-created scheduling point"
        );
    }

    #[test]
    fn test_two_threads_no_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None, SearchStrategy::Dfs);
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
        let mut engine = DporEngine::new(2, None, 1000, None, SearchStrategy::Dfs);
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
        let mut engine = DporEngine::new(2, None, 1000, None, SearchStrategy::Dfs);
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

        let mut engine = DporEngine::new(2, None, 100_000, None, SearchStrategy::Dfs);
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
        let mut engine = DporEngine::new(2, None, 1000, None, SearchStrategy::Dfs);
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
    fn test_post_release_access_races_with_post_acquire_access() {
        let mut engine = DporEngine::new(2, None, 1000, None, SearchStrategy::Dfs);
        let mut execution = engine.begin_execution();
        const LOCK: u64 = 10;
        const OBJECT: u64 = 20;

        assert_eq!(engine.schedule(&mut execution), Some(0));
        engine.process_sync(&mut execution, 0, SyncEvent::LockRelease { lock_id: LOCK }, None);
        engine.process_access(&mut execution, 0, OBJECT, AccessKind::Write);
        execution.finish_thread(0);

        assert_eq!(engine.schedule(&mut execution), Some(1));
        engine.process_sync(&mut execution, 1, SyncEvent::LockAcquire { lock_id: LOCK }, None);
        engine.process_access(&mut execution, 1, OBJECT, AccessKind::Write);

        assert_eq!(engine.pending_races.len(), 1, "post-release work must not be inherited by the acquirer");
    }

    #[test]
    fn test_pre_release_access_remains_ordered_before_post_acquire_access() {
        let mut engine = DporEngine::new(2, None, 1000, None, SearchStrategy::Dfs);
        let mut execution = engine.begin_execution();
        const LOCK: u64 = 10;
        const OBJECT: u64 = 20;

        assert_eq!(engine.schedule(&mut execution), Some(0));
        engine.process_access(&mut execution, 0, OBJECT, AccessKind::Write);
        engine.process_sync(&mut execution, 0, SyncEvent::LockRelease { lock_id: LOCK }, None);
        execution.finish_thread(0);

        assert_eq!(engine.schedule(&mut execution), Some(1));
        engine.process_sync(&mut execution, 1, SyncEvent::LockAcquire { lock_id: LOCK }, None);
        engine.process_access(&mut execution, 1, OBJECT, AccessKind::Read);

        assert!(engine.pending_races.is_empty(), "the release/acquire edge must still order the published prefix");
    }

    #[test]
    fn test_three_threads_write_conflict() {
        // Regression test: with the old single-last-access tracking,
        // Thread A's access would be overwritten by Thread B's, so the
        // conflict between A and C was never explored. The per-thread
        // map ensures all conflicts are detected.
        let mut engine = DporEngine::new(3, None, 1000, None, SearchStrategy::Dfs);
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

    /// Model of the multi-lock race (defect #11):
    ///
    /// Thread A: lock(cL), write(count), unlock(cL), lock(sL), write(sum), unlock(sL)
    /// Thread B: lock(cL), read(count), unlock(cL), lock(sL), read(sum), unlock(sL)
    ///
    /// The bug: B can interleave between A's two critical sections, seeing
    /// count=1 but sum=0.  DPOR must explore the ordering where A goes first
    /// on count_lock but B goes first on sum_lock.
    #[test]
    fn test_multi_lock_race_detected() {
        let mut engine = DporEngine::new(2, None, 100_000, None, SearchStrategy::Dfs);

        // Object IDs
        const COUNT: u64 = 1;
        const SUM: u64 = 2;
        const CL: u64 = 100;  // count_lock
        const SL: u64 = 200;  // sum_lock

        // Thread operations: (object_id, access_kind, is_lock_acquire, is_lock_release, lock_id)
        // We model lock acquire/release as sync events + io accesses.
        #[derive(Clone)]
        enum Op {
            LockAcquire(u64),
            Access(u64, AccessKind),
            LockRelease(u64),
        }

        let thread_a = vec![
            Op::LockAcquire(CL),
            Op::Access(COUNT, AccessKind::Write),
            Op::LockRelease(CL),
            Op::LockAcquire(SL),
            Op::Access(SUM, AccessKind::Write),
            Op::LockRelease(SL),
        ];
        let thread_b = vec![
            Op::LockAcquire(CL),
            Op::Access(COUNT, AccessKind::Read),
            Op::LockRelease(CL),
            Op::LockAcquire(SL),
            Op::Access(SUM, AccessKind::Read),
            Op::LockRelease(SL),
        ];
        let thread_ops = [thread_a, thread_b];

        let mut found_bug = false;
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();
            let mut count = 0i64;
            let mut sum = 0i64;
            let mut observed_count: Option<i64> = None;
            let mut observed_sum: Option<i64> = None;
            let mut pcs = [0usize; 2];
            let mut locks_held: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();

            loop {
                // Mark finished threads
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

                match &thread_ops[chosen][pc] {
                    Op::LockAcquire(lock_id) => {
                        if let Some(&holder) = locks_held.get(lock_id) {
                            if holder != chosen {
                                // Lock is held by other thread — block
                                execution.block_thread(chosen);
                                continue; // don't advance PC
                            }
                        }
                        locks_held.insert(*lock_id, chosen);
                        engine.process_sync(
                            &mut execution, chosen,
                            SyncEvent::LockAcquire { lock_id: *lock_id },
                            None,
                        );
                    }
                    Op::Access(obj_id, kind) => {
                        engine.process_access(&mut execution, chosen, *obj_id, *kind);
                        // Apply state changes
                        match (chosen, *obj_id, *kind) {
                            (0, 1, AccessKind::Write) => count = 1,
                            (0, 2, AccessKind::Write) => sum = 10,
                            (1, 1, AccessKind::Read) => observed_count = Some(count),
                            (1, 2, AccessKind::Read) => observed_sum = Some(sum),
                            _ => {}
                        }
                    }
                    Op::LockRelease(lock_id) => {
                        locks_held.remove(lock_id);
                        engine.process_sync(
                            &mut execution, chosen,
                            SyncEvent::LockRelease { lock_id: *lock_id },
                            None,
                        );
                        // Unblock any thread waiting for this lock
                        for tid in 0..2 {
                            if tid != chosen && execution.threads[tid].blocked {
                                if pcs[tid] < thread_ops[tid].len() {
                                    if let Op::LockAcquire(lid) = &thread_ops[tid][pcs[tid]] {
                                        if lid == lock_id {
                                            execution.unblock_thread(tid);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                pcs[chosen] += 1;
            }

            // Check invariant: if B observed count, it should be consistent
            if let (Some(c), Some(s)) = (observed_count, observed_sum) {
                if c > 0 && s != c * 10 {
                    found_bug = true;
                }
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        assert!(
            found_bug,
            "DPOR should detect the multi-lock race (count=1, sum=0 interleaving). \
             Explored {exec_count} executions without finding it."
        );
    }

    #[test]
    fn test_three_threads_read_write_conflict() {
        // Thread A reads, Thread B reads, Thread C writes — all same object.
        // With the old implementation, Thread A's read could be lost.
        let mut engine = DporEngine::new(3, None, 1000, None, SearchStrategy::Dfs);
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

    /// Four threads each doing one write to the same object.
    /// Verifies that the wakeup tree correctly handles 4-way write conflicts.
    /// All 4! = 24 orderings are distinct traces and must be explored.
    #[test]
    fn test_four_threads_write_conflict() {
        let mut engine = DporEngine::new(4, None, 10000, None, SearchStrategy::Dfs);
        let mut exec_count = 0;

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
                engine.process_access(&mut execution, chosen, 1, AccessKind::Write);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // 4 threads all writing to the same object: 4! = 24 orderings
        assert_eq!(exec_count, 24);
    }

    /// Two independent pairs: T0/T1 write X, T2/T3 write Y.
    /// Should explore 2 * 2 = 4 orderings (independent pairs don't cross).
    #[test]
    fn test_two_independent_pairs() {
        let mut engine = DporEngine::new(4, None, 10000, None, SearchStrategy::Dfs);
        let mut exec_count = 0;
        // T0 and T1 write object 1; T2 and T3 write object 2
        let thread_objects = [1u64, 1, 2, 2];

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
                engine.process_access(
                    &mut execution,
                    chosen,
                    thread_objects[chosen],
                    AccessKind::Write,
                );
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Two independent pairs: 2! * 2! = 4 orderings
        assert_eq!(exec_count, 4);
    }

    /// Model of dining philosophers (simplified):
    /// 2 threads, each locks two resources in opposite order.
    /// Thread 0: write(A), write(B)
    /// Thread 1: write(B), write(A)
    /// Should find all interleavings including the deadlock-prone one.
    #[test]
    fn test_two_philosophers_all_orderings() {
        let mut engine = DporEngine::new(2, None, 10000, None, SearchStrategy::Dfs);
        let thread_ops: Vec<Vec<(u64, AccessKind)>> = vec![
            vec![(1, AccessKind::Write), (2, AccessKind::Write)],
            vec![(2, AccessKind::Write), (1, AccessKind::Write)],
        ];
        let mut exec_count = 0;
        let mut traces = Vec::new();

        loop {
            let mut execution = engine.begin_execution();
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
                pcs[chosen] += 1;
            }

            traces.push(execution.schedule_trace.clone());
            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Without sleep set propagation, classic DPOR explores 6 orderings.
        // With propagation, write(A)♦write(B) (different objects) allows
        // commuting across thread boundaries, reducing to 3 Mazurkiewicz
        // traces (the feasible orderings of A-writes and B-writes):
        //   1. T0A < T1A and T0B < T1B  →  [0,0,1,1]
        //   2. T0A < T1A and T1B < T0B  →  [0,1,1,0] ≡ [1,0,0,1] etc.
        //   3. T1A < T0A and T1B < T0B  →  [1,1,0,0]
        // (The 4th combination T1A<T0A, T0B<T1B creates a cycle and is infeasible.)
        assert!(
            exec_count >= 3,
            "expected at least 3 executions for 2-philosopher, got {exec_count}"
        );
    }

    /// Writer-Readers: T0 writes x, T1 reads x, T2 reads x.
    ///
    /// There are exactly 4 distinct Mazurkiewicz traces (modulo the
    /// independence of read-read on the same object):
    ///
    ///   1. W-R1-R2  (≡ W-R2-R1 since R1♦R2)
    ///   2. R1-W-R2
    ///   3. R2-W-R1
    ///   4. R1-R2-W  (≡ R2-R1-W since R1♦R2)
    ///
    /// Without sleep set propagation, classic DPOR explores 5+ executions
    /// because it doesn't recognize that e.g. R1-R2-W and R2-R1-W are
    /// equivalent traces (the reads commute).
    ///
    /// With sleep set propagation (Algorithm 2 line 16, JACM'17 p.24:
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    /// ), after exploring R2-W-R1 with T0 visited/sleeping at position 0,
    /// when we explore T2 at position 0 (T2-...), T1 is also sleeping and
    /// independent of T2 (read♦read), so T1 stays asleep. This prevents
    /// exploring both T2-T0-T1 and T2-T1-T0, collapsing them into one.
    ///
    /// Paper ref: JACM'17 Section 11 Table 1 (p.36): the "readers" benchmark
    /// with n=3 has 4 source-set traces vs more for classic DPOR.
    #[test]
    fn test_writer_readers_sleep_propagation() {
        let mut engine = DporEngine::new(3, None, 10000, None, SearchStrategy::Dfs);
        let thread_kinds = [AccessKind::Write, AccessKind::Read, AccessKind::Read];
        let mut exec_count = 0;

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

        // With trace caching (Phase 2b), sleep set propagation extends to new
        // branches. The cached per-thread access unions allow the independence
        // check to work at new positions: read♦read → readers stay asleep.
        // This collapses equivalent reader orderings, giving exactly 4 traces
        // (one per write position: before both, between R1-R2, between R2-R1, after both).
        //
        // Paper ref: JACM'17 Table 1 (p.36) — readers(2) with source sets = 4.
        // Previously 5 with replay-only propagation (one redundant reader ordering).
        assert_eq!(
            exec_count, 4,
            "writer-readers (1W + 2R) should explore exactly 4 traces with trace caching, got {exec_count}"
        );
    }

    /// Five threads: T0 writes x, T1-T4 all read x.
    ///
    /// The distinct Mazurkiewicz traces are determined by where the write
    /// appears relative to the reads. Since all reads are mutually
    /// independent (read♦read), the optimal number of traces is 5
    /// (one per position of W in the sequence).
    ///
    /// With Phase 1 sleep set propagation alone, we get 16 traces
    /// (= sum of C(4,k) for k=0..4 — the binomial coefficients counting
    /// how many readers appear before W). Full optimality (5 traces)
    /// requires source set filtering (Phase 2, JACM'17 Def 4.3 p.15):
    /// adding only one thread per race prevents the combinatorial blowup
    /// of reader orderings.
    ///
    /// Paper ref: JACM'17 Table 1 (p.36), "readers" benchmark.
    #[test]
    fn test_writer_four_readers_sleep_propagation() {
        let mut engine = DporEngine::new(5, None, 100_000, None, SearchStrategy::Dfs);
        let thread_kinds = [
            AccessKind::Write,
            AccessKind::Read,
            AccessKind::Read,
            AccessKind::Read,
            AccessKind::Read,
        ];
        let mut exec_count = 0;

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

        // With trace caching (Phase 2b), sleep set propagation extends to new
        // branches. For readers(N), the optimal count with source sets is 2^N
        // (JACM'17 Table 1 p.36): each subset of {R1,...,RN} that appears
        // before the writer gives a distinct Mazurkiewicz trace. For N=4: 2^4 = 16.
        //
        // Previously ~65 with replay-only propagation. Trace caching enables
        // reader-reader independence at new branches, collapsing equivalent
        // reader orderings within each "before/after" partition.
        assert_eq!(
            exec_count, 16,
            "writer-readers (1W + 4R) should explore exactly 16 traces with trace caching, got {exec_count}"
        );
    }

    /// Sleep set propagation must not break independent-pair reduction.
    /// T0/T1 write X, T2/T3 write Y — two independent pairs should still
    /// explore exactly 2! × 2! = 4 orderings with propagation enabled.
    #[test]
    fn test_independent_pairs_with_propagation() {
        let mut engine = DporEngine::new(4, None, 10000, None, SearchStrategy::Dfs);
        let thread_objects = [1u64, 1, 2, 2];
        let mut exec_count = 0;

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
                engine.process_access(
                    &mut execution,
                    chosen,
                    thread_objects[chosen],
                    AccessKind::Write,
                );
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Independent pairs: 2! × 2! = 4 orderings — same as without propagation
        assert_eq!(exec_count, 4);
    }

    /// Lastzero(3): models the lastzero benchmark from POPL'14 Fig.4 (p.11).
    ///
    /// Variables: array[0..3] = {0, 0, 0, 0}
    /// Thread 0: searches backward for the last zero:
    ///   for (i := 3; array[i] != 0; i--)
    /// Thread j (j ∈ 1..3): array[j] := array[j-1] + 1
    ///
    /// Thread 0 has data-dependent control flow: each read determines
    /// whether it continues the loop or stops. This creates races between
    /// thread 0's reads and the other threads' writes.
    ///
    /// The distinct traces depend on which array elements thread 0 sees
    /// as zero vs non-zero, which depends on interleaving with writers.
    ///
    /// Paper ref: POPL'14 Table 2 (p.11), Figure 4 (p.11).
    /// For lastzero(5): classic=241, source=79, optimal=64.
    /// lastzero(3) should be significantly smaller.
    ///
    /// This test verifies that DPOR explores a bounded number of traces
    /// for this data-dependent benchmark.
    #[test]
    fn test_lastzero_three() {
        let mut engine = DporEngine::new(4, None, 100_000, None, SearchStrategy::Dfs);
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();

            // Thread state: array[0..3], thread 0's loop variable i
            let mut array = [0i64; 4];
            let mut pcs = [0usize; 4]; // program counter per thread
            // Thread 0: reads array[3], array[2], array[1], array[0] in sequence
            //   (stops when it finds a zero)
            // Thread j (1..3): reads array[j-1] (pc=0), writes array[j] (pc=1)
            let mut thread0_i: usize = 3; // loop variable for thread 0
            let mut thread0_done = false;

            loop {
                // Mark finished threads
                for j in 1..4 {
                    if pcs[j] >= 2 {
                        execution.finish_thread(j);
                    }
                }
                if thread0_done {
                    execution.finish_thread(0);
                }
                if execution.runnable_threads().is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };

                if chosen == 0 {
                    // Thread 0: read array[thread0_i]
                    let obj_id = thread0_i as u64;
                    engine.process_access(&mut execution, 0, obj_id, AccessKind::Read);

                    if array[thread0_i] != 0 {
                        // Non-zero: continue loop (decrement i)
                        if thread0_i > 0 {
                            thread0_i -= 1;
                        } else {
                            thread0_done = true;
                        }
                    } else {
                        // Found zero: stop loop
                        thread0_done = true;
                    }
                } else {
                    // Thread j (1..3): two operations
                    let j = chosen;
                    match pcs[j] {
                        0 => {
                            // Read array[j-1]
                            let obj_id = (j - 1) as u64;
                            engine.process_access(&mut execution, j, obj_id, AccessKind::Read);
                            // Store the read value for the write step
                            let val = array[j - 1];
                            array[j] = val + 1; // Actually, the write happens at pc=1
                            // Undo: we should separate read and write
                            array[j] = 0; // Reset, write happens at next step
                            pcs[j] += 1;
                        }
                        1 => {
                            // Write array[j] := array[j-1] + 1
                            let obj_id = j as u64;
                            engine.process_access(&mut execution, j, obj_id, AccessKind::Write);
                            array[j] = array[j - 1] + 1;
                            pcs[j] += 1;
                        }
                        _ => {}
                    }
                }
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // With trace caching + full sleep propagation, lastzero(3) explores ~43 traces.
        // Data-dependent control flow in thread 0 prevents full optimal reduction.
        // The union-based trace cache is conservative for data-dependent programs.
        assert!(
            exec_count <= 60,
            "lastzero(3) should explore at most 60 traces, got {exec_count}"
        );
    }

    // --- Deferred race detection tests (Phase 3) ---

    /// Three threads: T0 and T2 race on X, T1 is independent (writes Y).
    /// With deferred race detection and notdep sequences, DPOR computes
    /// notdep(e0, E).e2 = [T1, T2] (T1 is independent of T0's write to X).
    ///
    /// The notdep sequence [T1, T2] guides exploration to replay T1 first
    /// (independent), then run T2 (the racing thread) before T0's write.
    /// Since T1 is fully independent, sleep sets collapse it: the two
    /// Mazurkiewicz traces are {T0<T2, T2<T0} on X, with T1 anywhere.
    #[test]
    fn test_notdep_three_threads_independent_intermediate() {
        let mut engine = DporEngine::new(3, None, 1000, None, SearchStrategy::Dfs);
        // T0 writes X, T1 writes Y, T2 writes X
        let thread_objects = [1u64, 2, 1];
        let mut exec_count = 0;

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
                engine.process_access(
                    &mut execution,
                    chosen,
                    thread_objects[chosen],
                    AccessKind::Write,
                );
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // T1 is fully independent (writes Y, not X). The two distinct
        // Mazurkiewicz traces are determined by the relative order of
        // T0 and T2's writes to X. T1 can go anywhere without changing
        // the trace equivalence class. With pure deferred+notdep, this
        // gives exactly 2. With the hybrid approach (inline wakeup +
        // deferred notdep), the inline insertions may add single-thread
        // entries that create a few extra explorations, but the count
        // should remain small.
        assert!(
            exec_count <= 4,
            "3 threads (T0/T2 race on X, T1 independent on Y): ≤4 traces, got {exec_count}"
        );
        assert!(
            exec_count >= 2,
            "3 threads (T0/T2 race on X, T1 independent on Y): ≥2 traces, got {exec_count}"
        );
    }

    /// Basic deferred race: 2 threads with write-write conflict.
    /// Should still explore exactly 2 executions with deferred detection.
    #[test]
    fn test_deferred_two_threads_write_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None, SearchStrategy::Dfs);
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

        assert_eq!(
            exec_count, 2,
            "2 threads write-write conflict with deferred detection: 2 executions, got {exec_count}"
        );
    }

    /// Notdep should skip dependent intermediate: T0/T2 write X, T1 also writes X.
    /// T1 is DEPENDENT on T0 (same object, write-write), so notdep = [T2] only.
    /// All 3 threads conflict on X → 3! = 6 orderings, but with sleep sets
    /// this may be reduced. Should be ≤ 24 at most.
    #[test]
    fn test_notdep_skips_dependent_intermediate() {
        let mut engine = DporEngine::new(3, None, 1000, None, SearchStrategy::Dfs);
        let mut exec_count = 0;

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
                // All 3 threads write to the same object
                engine.process_access(&mut execution, chosen, 1, AccessKind::Write);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        assert!(
            exec_count <= 24,
            "3 threads all writing same object: should explore ≤ 24 traces, got {exec_count}"
        );
        // With full 3-way conflict, we expect 6 (3! orderings)
        assert!(
            exec_count >= 6,
            "3 threads all writing same object: should explore ≥ 6 traces, got {exec_count}"
        );
    }

    /// Run a dining-philosopher model until a lock-order deadlock is found
    /// or exploration finishes.  `thread_ops[t]` is the op sequence for thread `t`;
    /// each op is `("lock_acquire"|"lock_release"|"write", object_id)`.
    /// Returns `(executions_explored_including_the_deadlocked_one, deadlock_found)`.
    fn run_philosopher_until_deadlock(
        engine: &mut DporEngine,
        thread_ops: &[Vec<(&str, u64)>],
    ) -> (u64, bool) {
        let num_threads = thread_ops.len();
        let mut exec_count: u64 = 0;
        loop {
            let mut execution = engine.begin_execution();
            let mut pcs = vec![0usize; num_threads];
            let mut locks_held: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
            loop {
                for i in 0..num_threads {
                    if pcs[i] >= thread_ops[i].len() {
                        execution.finish_thread(i);
                    }
                }
                if execution.runnable_threads().is_empty() {
                    break;
                }
                let Some(chosen) = engine.schedule(&mut execution) else { break };
                let pc = pcs[chosen];
                if pc >= thread_ops[chosen].len() {
                    break;
                }
                let (op_type, obj_id) = thread_ops[chosen][pc];
                match op_type {
                    "lock_acquire" => {
                        if let Some(&holder) = locks_held.get(&obj_id) {
                            if holder != chosen {
                                execution.block_thread(chosen);
                                continue;
                            }
                        }
                        locks_held.insert(obj_id, chosen);
                        engine.process_sync(&mut execution, chosen, SyncEvent::LockAcquire { lock_id: obj_id }, None);
                    }
                    "lock_release" => {
                        locks_held.remove(&obj_id);
                        engine.process_sync(&mut execution, chosen, SyncEvent::LockRelease { lock_id: obj_id }, None);
                        for tid in 0..num_threads {
                            if tid != chosen && execution.threads[tid].blocked && pcs[tid] < thread_ops[tid].len() {
                                let (t_op, t_id) = thread_ops[tid][pcs[tid]];
                                if t_op == "lock_acquire" && t_id == obj_id {
                                    execution.unblock_thread(tid);
                                }
                            }
                        }
                    }
                    "write" => engine.process_access(&mut execution, chosen, obj_id, AccessKind::Write),
                    _ => panic!("Unknown op: {op_type}"),
                }
                pcs[chosen] += 1;
            }
            exec_count += 1;
            let all_done = pcs.iter().enumerate().all(|(i, pc)| *pc >= thread_ops[i].len());
            if !all_done {
                return (exec_count, true);
            }
            if !engine.next_execution() {
                return (exec_count, false);
            }
        }
    }

    /// Dining philosophers model tests.
    #[test]
    fn test_four_philosophers_lock_only() {
        // Pure lock-ordering deadlock: acquire left, acquire right, release right, release left
        let mut engine = DporEngine::new(4, Some(2), 100_000, Some(50000), SearchStrategy::Dfs);
        let num_philosophers = 4;

        let thread_ops: Vec<Vec<(&str, u64)>> = (0..num_philosophers)
            .map(|i| {
                let left = i as u64;
                let right = ((i + 1) % num_philosophers) as u64;
                vec![
                    ("lock_acquire", left),
                    ("lock_acquire", right),
                    ("lock_release", right),
                    ("lock_release", left),
                ]
            })
            .collect();

        let (exec_count, deadlock_found) = run_philosopher_until_deadlock(&mut engine, &thread_ops);

        assert!(
            deadlock_found,
            "4-philosopher lock-only model should find a deadlock within {exec_count} executions"
        );
        assert!(
            exec_count <= 5000,
            "4-philosopher lock-only model should not explode: got {exec_count} executions"
        );
    }

    #[test]
    fn test_three_philosophers_with_write() {
        // Lock + shared write: acquire left, write shared, acquire right, release right, release left
        let mut engine = DporEngine::new(3, Some(2), 100_000, Some(50000), SearchStrategy::Dfs);
        let num_philosophers = 3;

        let thread_ops: Vec<Vec<(&str, u64)>> = (0..num_philosophers)
            .map(|i| {
                let left = i as u64;
                let right = ((i + 1) % num_philosophers) as u64;
                vec![
                    ("lock_acquire", left),
                    ("write", 100),
                    ("lock_acquire", right),
                    ("lock_release", right),
                    ("lock_release", left),
                ]
            })
            .collect();

        let (exec_count, deadlock_found) = run_philosopher_until_deadlock(&mut engine, &thread_ops);

        assert!(
            deadlock_found,
            "3-philosopher+write model should find a deadlock within {exec_count} executions"
        );
    }
}
