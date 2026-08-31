//! Exploration tree for Source Sets DPOR.
//!
//! Implements a hybrid of Algorithms 1 and 2 from Abdulla et al., "Source Sets:
//! A Foundation for Optimal Dynamic Partial Order Reduction", JACM 2017.
//!
//! **Current algorithm**: Uses wakeup trees (Algorithm 2, JACM'17 p.24-25) as
//! the exploration structure, with race detection during execution (Algorithm 1
//! style, JACM'17 p.16) as well as deferred notdep processing (Algorithm 2).
//! Sleep sets are propagated across scheduling points using trace caching.
//! Source set filtering is disabled (all racing threads are inserted into
//! wakeup trees).
//!
//! Each scheduling point (Branch) maintains:
//!
//! - A **wakeup tree** of thread sequences to explore — replaces classic
//!   Backtrack status with ordered, structured exploration.
//!   (Algorithm 2 lines 14-20, JACM'17 p.24-25)
//! - A **sleep set** of threads that should not be re-explored because an
//!   equivalent execution has already been covered.
//!   (Algorithm 2 lines 13, 20, JACM'17 p.24-25)
//! - **Object tracking** per thread to enable independence-based sleep set
//!   propagation across scheduling points (needed for Algorithm 2 line 16:
//!   `Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}`).

use std::collections::HashMap;

use crate::access::{AccessKind, AccessOrigin};
use crate::thread::ThreadStatus;
use crate::wakeup_tree::WakeupTree;

/// Controls the order in which wakeup tree branches are explored.
///
/// All strategies visit exactly the same set of Mazurkiewicz trace equivalence
/// classes (soundness and completeness are preserved). Only the *order* differs,
/// which affects how quickly bugs are found with `stop_on_first=True`.
#[derive(Clone, Debug)]
pub enum SearchStrategy {
    /// Classic DFS: always pick `min_thread()` (lowest thread ID).
    /// Deterministic, no seed. This is the paper's Algorithm 2 default.
    Dfs,
    /// Bit-reversal permutation ordering: at each wakeup tree node, visit
    /// children in bit-reversal order of their position among siblings.
    /// This maximally spreads exploration across distinct conflict points
    /// early (low-discrepancy sequence over the tree).
    ///
    /// The `seed` rotates the permutation, giving different deterministic
    /// orderings that all cover the full tree exactly once.
    BitReversal { seed: u64 },
    /// Round-robin across thread IDs: cycles through available threads in a
    /// rotating order. Each call advances the offset, so consecutive picks
    /// at the same tree level choose different threads.
    ///
    /// The `seed` sets the initial offset.
    RoundRobin { seed: u64 },
    /// Stride-based: picks every s-th sibling (where s is derived from the
    /// seed and chosen coprime to the branching factor). Different seeds
    /// produce different deterministic traversals that each cover the tree
    /// exactly once.
    Stride { seed: u64 },
    /// Conflict-first: prioritize threads involved in deeper wakeup sequences
    /// (i.e., those that reverse races at shallower depths get explored first).
    /// Falls back to max_thread() — the reverse of DFS's min_thread().
    ConflictFirst,
}

/// A pending race collected during execution for deferred processing.
///
/// Paper ref: Algorithm 2 lines 1-6 (JACM'17 p.24). In Optimal-DPOR,
/// race detection is deferred to maximal executions (when all threads are
/// finished or blocked). Each race is stored as a PendingRace and
/// processed in `next_execution()` where the full execution trace is
/// available for computing notdep sequences.
#[derive(Clone, Debug)]
pub struct PendingRace {
    /// Position of event e (the first racing access).
    pub prev_path_id: usize,
    /// Position of event e' (the second racing access).
    pub current_path_id: usize,
    /// Thread that performed e' (the thread to reverse to).
    pub thread_id: usize,
    /// The shared object involved in the race.
    pub race_object: Option<u64>,
    /// Whether inline wakeup insertion was skipped for this race.
    /// When true, deferred processing must handle single-thread notdep
    /// sequences (len == 1) that would normally be covered by inline
    /// insertion. Set for races on objects with resource groups (Defect #15).
    pub inline_skipped: bool,
    /// Whether this is an attribute-level race (from `process_access` /
    /// `process_first_access`) vs a container-level, I/O, or synthetic race.
    /// Only attribute-level races are "true" data races for the purpose of
    /// `error_on_any_race` mode.
    pub is_attribute_race: bool,
}

/// Check if two sets of (object → access_kind) are independent.
///
/// Two sets are independent if, for every object that appears in both sets,
/// the access kinds are non-conflicting (e.g., both reads).
///
/// Paper ref: JACM'17 Def 3.3 (p.13) — independence is the negation of
/// dependence. We approximate the paper's E ⊢ p♦q (Def 3.3) using access-kind
/// compatibility as a sufficient condition for independence.
fn accesses_are_independent(
    a: &HashMap<u64, (AccessKind, AccessOrigin)>,
    b: &HashMap<u64, (AccessKind, AccessOrigin)>,
) -> bool {
    // Iterate over the smaller map for efficiency
    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    for (obj, (kind_a, origin_a)) in smaller {
        if let Some((kind_b, origin_b)) = larger.get(obj) {
            // Origins available for future per-origin conflict policies (Fix 6 phase 2).
            let _ = (origin_a, origin_b);
            if kind_a.conflicts(*kind_b) {
                return false;
            }
        }
    }
    true
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub threads: Vec<ThreadStatus>,
    pub active_thread: usize,
    pub preemptions: u32,
    /// Wakeup tree: threads/sequences to explore at this scheduling point.
    /// Paper: wut(E) in Algorithm 2 (JACM'17 p.24).
    pub wakeup: WakeupTree,
    /// Sleep set: threads excluded from wakeup tree insertion because
    /// an equivalent execution starting with them has already been explored.
    /// Indexed by thread ID; `true` = sleeping.
    /// Paper: sleep(E) in Algorithm 2 (JACM'17 p.24). Tracks Visited threads
    /// locally at each position. Cross-branch propagation is handled by
    /// `propagated_sleep_accesses` below.
    pub sleep: Vec<bool>,
    /// Accesses (object → kind) performed by the active thread at this step.
    /// Used for sleep set independence checks during propagation.
    ///
    /// Tracks the AccessKind so we can distinguish read-read (independent)
    /// from read-write (dependent) on the same object.
    ///
    /// Paper: needed to compute E ⊢ p♦q (independence, JACM'17 Def 3.3 p.13)
    /// — two events are independent if their accesses don't conflict.
    pub active_accesses: HashMap<u64, (AccessKind, AccessOrigin)>,
    /// For each previously-explored (Visited) thread at this position,
    /// its accesses (object → (kind, origin)). Used for sleep set propagation:
    /// a sleeping thread stays asleep if its accesses are independent of
    /// the active thread's accesses.
    ///
    /// Paper: approximates the independence check E ⊢ p♦q (JACM'17 p.13).
    pub explored_accesses: HashMap<usize, HashMap<u64, (AccessKind, AccessOrigin)>>,
    /// Sleep set entries propagated from the previous scheduling point.
    ///
    /// When thread p is chosen at position i, each sleeping thread q at
    /// position i whose next action is independent of p's action (p♦q)
    /// stays asleep at position i+1. This map carries both the sleeping
    /// status AND the thread's access info, enabling multi-hop propagation.
    ///
    /// Paper ref: Algorithm 2 line 16 (JACM'17 p.24):
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    /// In the paper's recursive formulation, Sleep' is passed to the
    /// recursive Explore(E.p, WuT', Sleep') call. In our iterative
    /// implementation, we compute it during replay and store it here.
    pub propagated_sleep_accesses: HashMap<usize, HashMap<u64, (AccessKind, AccessOrigin)>>,
    // Note: pending_race_objects was removed — source set filtering at the
    // race-object level requires sleep sets for soundness.
    // Paper: source set filtering (Def 4.3, JACM'17 p.15) allows adding only
    // one thread per race, but this requires sleep sets to prevent re-exploration
    // of equivalent traces. See ideas/optimal_dpor.md Phase 2.
}

impl Branch {
    pub fn new(threads: Vec<ThreadStatus>, active_thread: usize, preemptions: u32) -> Self {
        let num_threads = threads.len();
        Self {
            threads,
            active_thread,
            preemptions,
            wakeup: WakeupTree::empty(),
            sleep: vec![false; num_threads],
            active_accesses: HashMap::new(),
            explored_accesses: HashMap::new(),
            propagated_sleep_accesses: HashMap::new(),
        }
    }
}

/// The exploration tree: manages DFS over scheduling decisions using
/// wakeup trees and sleep sets (Source Sets DPOR).
pub struct Path {
    branches: Vec<Branch>,
    pos: usize,
    preemption_bound: Option<u32>,
    search_strategy: SearchStrategy,
    /// Counter incremented each time we pick a thread from a wakeup tree
    /// (in step() or schedule()). Used by BitReversal to index into the
    /// bit-reversal permutation.
    selection_counter: u64,
    /// Per-step access cache from the most recently completed execution.
    ///
    /// `prev_thread_steps[tid][k]` = thread `tid`'s accesses at exactly its
    /// k-th scheduling point (NOT merged with later steps). Used for per-step
    /// independence checking which avoids false wakeups from merge escalation.
    ///
    /// Example: if thread T does WeakWrite(X) at step 0 and Read(X) at step 1,
    /// a merged suffix is {X: Write} (merge(WeakWrite, Read) = Write),
    /// but the per-step data preserves {X: WeakWrite} and {X: Read} separately.
    /// This allows checking each step individually against the active thread's
    /// accesses, avoiding false conflicts from merge escalation.
    prev_thread_steps: HashMap<usize, Vec<HashMap<u64, (AccessKind, AccessOrigin)>>>,
    /// Wakeup subtree carried from step() to schedule() for multi-step
    /// wakeup sequence guidance.
    ///
    /// When step() picks a thread from the wakeup tree, the subtree of
    /// that thread is stored here. At the first NEW branch after replay,
    /// schedule() consumes it to guide the thread choice and provides
    /// further subtree guidance for subsequent positions.
    ///
    /// Paper ref: Algorithm 2 line 17 (JACM'17 p.24-25):
    ///   WuT' = subtree(wut(E), p)
    /// The subtree guides the recursive Explore(E.p, WuT', Sleep') call.
    /// In our iterative implementation, it cascades through new positions
    /// via this field.
    pending_wakeup_subtree: Option<WakeupTree>,
}

impl Path {
    pub fn new(preemption_bound: Option<u32>, search_strategy: SearchStrategy) -> Self {
        Self {
            branches: Vec::new(),
            pos: 0,
            preemption_bound,
            search_strategy,
            selection_counter: 0,
            prev_thread_steps: HashMap::new(),
            pending_wakeup_subtree: None,
        }
    }


    pub fn current_position(&self) -> usize {
        self.pos
    }

    pub fn depth(&self) -> usize {
        self.branches.len()
    }

    /// Look up a thread's future accesses after `completed_steps` of its own
    /// scheduling steps in the current execution.
    ///
    /// Returns an empty map at the exact end of the previous trace and `None`
    /// once the current execution has progressed beyond that boundary.  Extra
    /// progress proves divergence and must wake the actor conservatively.
    fn future_accesses_for(
        &self,
        tid: usize,
        completed_steps: usize,
    ) -> Option<HashMap<u64, (AccessKind, AccessOrigin)>> {
        let steps = self.prev_thread_steps.get(&tid)?;
        if completed_steps < steps.len() {
            // Build only the suffix requested by sleep propagation.  The old
            // cache eagerly cloned every suffix map, retaining quadratic data
            // even for actors that never slept.
            let mut accesses: HashMap<u64, (AccessKind, AccessOrigin)> = HashMap::new();
            for step in &steps[completed_steps..] {
                for (obj_id, (kind, origin)) in step {
                    accesses
                        .entry(*obj_id)
                        .and_modify(|(existing_kind, existing_origin)| {
                            *existing_kind = existing_kind.merge(*kind);
                            *existing_origin = existing_origin.merge(*origin);
                        })
                        .or_insert((*kind, *origin));
                }
            }
            Some(accesses)
        } else if completed_steps == steps.len() {
            // Matching the prior trace exactly means the next cached action is
            // termination.  Only progress *past* this boundary proves that the
            // current execution has diverged and must wake conservatively.
            Some(HashMap::new())
        } else {
            None
        }
    }

    /// Check whether a sleeping thread's remaining individual steps are ALL
    /// independent of the active thread's accesses, using the per-step cache.
    ///
    /// Unlike `future_accesses_for()` which returns a conservative merged union,
    /// this checks each step individually. This avoids false wakeups from
    /// merge escalation (e.g., WeakWrite + Read → Write in the merged form,
    /// but each step individually may be independent of the active accesses).
    ///
    /// Returns `Some(true)` if ALL remaining steps are independent,
    /// `Some(false)` if any step conflicts, `None` if no per-step cache.
    fn future_steps_independent_of(
        &self,
        tid: usize,
        completed_steps: usize,
        active: &HashMap<u64, (AccessKind, AccessOrigin)>,
    ) -> Option<bool> {
        let steps = self.prev_thread_steps.get(&tid)?;
        if completed_steps > steps.len() {
            return None;
        }
        if completed_steps == steps.len() {
            return Some(true);
        }
        // Check each remaining step individually
        for step_idx in completed_steps..steps.len() {
            if !accesses_are_independent(&steps[step_idx], active) {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Record that an object was accessed at the given scheduling step.
    /// Called by the engine after each process_access/process_io_access.
    /// Populates `active_accesses` for future independence checks
    /// (E ⊢ p♦q, JACM'17 Def 3.3 p.13).
    ///
    /// When the same object is accessed multiple times at one scheduling
    /// point with different AccessKinds, we use AccessKind::merge() to
    /// combine them: Read + WeakRead → Read (since Read already covers
    /// WeakRead's conflicts), while other mixed pairs → Write (conservative).
    pub fn record_access(&mut self, path_id: usize, object_id: u64, kind: AccessKind, origin: AccessOrigin) {
        if let Some(branch) = self.branches.get_mut(path_id) {
            branch.active_accesses
                .entry(object_id)
                .and_modify(|(existing_kind, existing_origin)| {
                    // Use merge() to combine access kinds correctly.
                    // Read + WeakRead → Read (Read subsumes WeakRead).
                    // All other mixed pairs → Write (conservative).
                    *existing_kind = existing_kind.merge(kind);
                    // Merge origin: keep the "stronger" one.
                    *existing_origin = existing_origin.merge(origin);
                })
                .or_insert((kind, origin));
        }
    }

    /// Pick which thread to run at the current scheduling point.
    /// During replay, follows the recorded path; otherwise creates a new branch.
    ///
    /// In Algorithm 2 (JACM'17 p.24), this corresponds to lines 8-12 (choosing
    /// which process to explore) and line 18 (recursive Explore call with
    /// Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}).
    ///
    /// **Sleep set propagation**: At each position, we propagate sleeping
    /// threads from the previous position. A thread q sleeping at position
    /// i-1 stays asleep at position i if q's recorded accesses are
    /// independent of the chosen thread p's accesses at position i-1.
    /// This implements Algorithm 2 line 16 (JACM'17 p.24).
    pub fn schedule(
        &mut self,
        runnable: &[usize],
        current_thread: usize,
        num_threads: usize,
    ) -> Option<usize> {
        self.schedule_preferred(runnable, current_thread, num_threads, None)
    }

    /// Schedule like [`Self::schedule`], but prefer *preferred* when creating
    /// a new branch. Replayed branches remain authoritative.
    pub fn schedule_preferred(
        &mut self,
        runnable: &[usize],
        current_thread: usize,
        num_threads: usize,
        preferred: Option<usize>,
    ) -> Option<usize> {
        if runnable.is_empty() {
            return None;
        }

        if self.pos < self.branches.len() {
            // Replay: propagate sleep set from previous position, then
            // clear active_accesses for fresh recording.
            self.propagate_sleep(self.pos);
            self.branches[self.pos].active_accesses.clear();
            let chosen = self.branches[self.pos].active_thread;
            self.pos += 1;
            return Some(chosen);
        }

        // New branch: check wakeup subtree guidance first (Algorithm 2
        // line 17, JACM'17 p.24-25: WuT' = subtree(wut(E), p)), then
        // fall back to preferring the current thread to minimize preemptions.
        // Within a wakeup subtree (multi-step sequence), use min_thread()
        // to follow the sequence as constructed. The search strategy only
        // affects the initial thread choice in step(), not subsequent guided
        // choices within a multi-step wakeup sequence.
        let (chosen, next_subtree) = if let Some(ref subtree) = self.pending_wakeup_subtree {
            if let Some(guided) = subtree.min_thread() {
                if runnable.contains(&guided) {
                    let sub = subtree.subtree(guided);
                    (guided, if sub.is_empty() { None } else { Some(sub) })
                } else {
                    let c = if runnable.contains(&current_thread) { current_thread } else { runnable[0] };
                    (c, None)
                }
            } else {
                let c = if runnable.contains(&current_thread) { current_thread } else { runnable[0] };
                (c, None)
            }
        } else if let Some(preferred) = preferred.filter(|tid| runnable.contains(tid)) {
            (preferred, None)
        } else {
            let c = if runnable.contains(&current_thread) { current_thread } else { runnable[0] };
            (c, None)
        };
        self.pending_wakeup_subtree = next_subtree;

        let is_preemption = chosen != current_thread && runnable.contains(&current_thread);
        let prev_preemptions = self.branches.last().map_or(0, |b| b.preemptions);
        let preemptions = if is_preemption { prev_preemptions + 1 } else { prev_preemptions };

        let mut threads = vec![ThreadStatus::Disabled; num_threads];
        for &tid in runnable {
            threads[tid] = if tid == chosen { ThreadStatus::Active } else { ThreadStatus::Pending };
        }

        let branch = Branch::new(threads, chosen, preemptions);
        self.branches.push(branch);

        // Propagate sleep set to new branches using position-sensitive
        // future access cache.  The cache provides each sleeping thread's
        // FUTURE accesses (from this position onward), enabling precise
        // independence checks at new positions beyond the replay prefix.
        self.propagate_sleep(self.pos);

        self.pos += 1;
        Some(chosen)
    }

    /// Propagate the sleep set from position `pos-1` to position `pos`.
    ///
    /// For each thread q that is sleeping at position pos-1 (either locally
    /// Visited or propagated from an earlier position), check if q's
    /// recorded accesses are independent of the active thread's accesses
    /// at pos-1. If independent, q stays asleep at pos.
    ///
    /// This implements Algorithm 2 line 16 (JACM'17 p.24):
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    ///
    /// The independence check E ⊢ p♦q is approximated by access-kind
    /// compatibility: two accesses to the same object are independent if
    /// they are both reads (or other non-conflicting combinations like
    /// WeakWrite+WeakRead). This is a sufficient condition — if the object
    /// sets are disjoint or all overlapping accesses are compatible, the
    /// actions are truly independent in any execution context.
    ///
    /// **Approach (c) from the plan**: We propagate through the replayed
    /// prefix and into new positions. For locally-sleeping threads, we use
    /// `explored_accesses` (recorded when the thread was the active thread
    /// at this position). For propagated threads, we carry their access
    /// info forward. Threads without access info are woken up (conservative).
    fn propagate_sleep(&mut self, pos: usize) {
        if pos == 0 {
            // Position 0 has no predecessor to propagate from.
            return;
        }

        let prev = pos - 1;
        // Compute every actor's progress once.  Recounting the whole prefix
        // separately for each sleeping actor (and again for each cache helper)
        // made propagation O(threads * depth) at every path position.
        let mut completed_steps: HashMap<usize, usize> = HashMap::new();
        for branch in &self.branches[..pos] {
            *completed_steps.entry(branch.active_thread).or_default() += 1;
        }
        // Collect the active thread's accesses at the previous position.
        // These are the accesses of the thread that was chosen at pos-1.
        let prev_active_accesses = self.branches[prev].active_accesses.clone();

        // Collect all sleeping threads at pos-1 and their access info.
        // A thread is sleeping if it's locally Visited (sleep[q]=true) or
        // propagated from an earlier position (in propagated_sleep_accesses).
        let mut sleeping_threads: HashMap<usize, HashMap<u64, (AccessKind, AccessOrigin)>> = HashMap::new();

        // 1. Locally-sleeping threads (Visited at pos-1)
        //
        // Use the position-sensitive future access cache when available.
        // Unlike the old full-union cache, this only includes accesses
        // the thread will perform at positions >= pos in the previous
        // execution.  This prevents false wakeups when a thread has
        // already finished its conflicting work at earlier positions.
        //
        // Fallback to `explored_accesses` when no trace cache is available
        // (e.g., during the first execution).
        let num_threads = self.branches[prev].sleep.len();
        for tid in 0..num_threads {
            if self.branches[prev].sleep.get(tid).copied().unwrap_or(false) {
                let progress = completed_steps.get(&tid).copied().unwrap_or(0);
                if let Some(accesses) = self.future_accesses_for(tid, progress) {
                    sleeping_threads.insert(tid, accesses);
                } else if !self.prev_thread_steps.contains_key(&tid) {
                    if let Some(accesses) = self.branches[prev].explored_accesses.get(&tid) {
                        sleeping_threads.insert(tid, accesses.clone());
                    }
                }
                // A missing or exhausted cache cannot certify the still-live
                // actor's future, so omission wakes it conservatively.
            }
        }

        // 2. Propagated-sleeping threads (from even earlier positions)
        //
        // Use position-sensitive cache for propagated threads too, since
        // their carried accesses may only reflect a single-position
        // snapshot from when they were first put to sleep.
        for (tid, accesses) in &self.branches[prev].propagated_sleep_accesses {
            let progress = completed_steps.get(tid).copied().unwrap_or(0);
            if let Some(cached) = self.future_accesses_for(*tid, progress) {
                sleeping_threads.insert(*tid, cached);
            } else if !self.prev_thread_steps.contains_key(tid) {
                sleeping_threads.insert(*tid, accesses.clone());
            }
        }

        // Compute which sleeping threads stay asleep at pos.
        //
        // Per-step independence check (Fix 6): instead of checking against the
        // merged suffix union (which can escalate e.g. WeakWrite+Read → Write),
        // check each remaining step individually. This avoids false wakeups
        // when individual steps are all independent but the merge is not.
        //
        // Falls back to the merged suffix union when no per-step cache is available
        // (e.g., first execution).
        let mut propagated: HashMap<usize, HashMap<u64, (AccessKind, AccessOrigin)>> = HashMap::new();
        for (tid, tid_accesses) in sleeping_threads {
            let progress = completed_steps.get(&tid).copied().unwrap_or(0);
            let independent = match self.future_steps_independent_of(tid, progress, &prev_active_accesses) {
                Some(result) => result,
                None => accesses_are_independent(&tid_accesses, &prev_active_accesses),
            };
            if independent {
                propagated.insert(tid, tid_accesses);
            }
            // If dependent, the thread wakes up — it must be available for
            // wakeup tree insertion at this position since its equivalent
            // trace may not have been explored.
        }

        self.branches[pos].propagated_sleep_accesses = propagated;
    }

    /// Insert thread_id into the wakeup tree at branch path_id.
    /// Filters against sleep sets and preemption bounds before insertion.
    ///
    /// This is called during execution (Algorithm 1 style, JACM'17 p.16 lines
    /// 5-9) to ensure all racing threads are added to the wakeup tree inline.
    /// Deferred notdep processing (Algorithm 2, p.24 lines 2-6) may later
    /// insert multi-step sequences for the same races.
    ///
    /// `_race_object`: the object involved in the race. Reserved for future
    /// source set filtering (Def 4.3, JACM'17 p.15; requires sleep sets).
    pub fn insert_wakeup(&mut self, path_id: usize, thread_id: usize, _race_object: Option<u64>) {
        if path_id >= self.branches.len() {
            return;
        }

        let branch = &self.branches[path_id];

        // Only runnable threads can be added to the wakeup tree
        match branch.threads.get(thread_id).copied() {
            Some(ThreadStatus::Pending) | Some(ThreadStatus::Yield) => {}
            _ => return,
        }

        // Sleep set check: if thread is sleeping (locally Visited or
        // propagated from an earlier position), skip.
        //
        // Paper: Algorithm 2 line 5 (JACM'17 p.24) checks
        //   sleep(E') ∩ WI[E'](v) = ∅
        // before inserting. Our version checks both the local sleep set
        // (threads Visited at this position) and the propagated sleep set
        // (threads whose next action is independent of all chosen threads
        // between their home position and this position).
        if branch.sleep.get(thread_id).copied().unwrap_or(false) {
            return;
        }
        if branch.propagated_sleep_accesses.contains_key(&thread_id) {
            return;
        }

        // Already in wakeup tree?
        if branch.wakeup.contains_thread(thread_id) {
            return;
        }

        // Preemption bound check
        if let Some(bound) = self.preemption_bound {
            let branch = &self.branches[path_id];
            if branch.active_thread != thread_id && branch.preemptions >= bound {
                self.add_conservative_wakeup(path_id, thread_id, bound);
                return;
            }
        }

        // Insert into wakeup tree.
        // Paper: Algorithm 2 line 6 (JACM'17 p.24):
        //   wut(E') := insert[E'](v, wut(E'))
        // We insert single-thread sequences [q] rather than full notdep
        // sequences v = notdep(e, E).e'. See Phase 3b for multi-step.
        self.branches[path_id].wakeup.insert(&[thread_id]);
    }

    /// Advance to the next unexplored execution path.
    /// Picks the next thread from the wakeup tree at the deepest branch.
    ///
    /// Implements the while loop of Algorithm 2 lines 14-20 (JACM'17 p.24-25):
    ///   while ∃p ∈ wut(E):
    ///     pick min≺{p}           → min_thread()
    ///     explore E.p            → set active_thread, reset pos
    ///     remove p.w from wut(E) → remove_branch()
    ///     add p to sleep(E)      → sleep[active] = true
    pub fn step(&mut self) -> bool {
        // Build per-thread exact-step data from the completed execution.
        let mut per_thread_accesses: HashMap<usize, Vec<&HashMap<u64, (AccessKind, AccessOrigin)>>> = HashMap::new();
        for branch in &self.branches {
            per_thread_accesses
                .entry(branch.active_thread)
                .or_default()
                .push(&branch.active_accesses);
        }
        // Clone each exact step once. Suffix unions are derived lazily only
        // when propagation needs one, rather than prebuilding every suffix.
        let mut per_step: HashMap<usize, Vec<HashMap<u64, (AccessKind, AccessOrigin)>>> = HashMap::new();
        for (tid, steps) in &per_thread_accesses {
            per_step.insert(*tid, steps.iter().map(|s| (*s).clone()).collect());
        }
        self.prev_thread_steps = per_step;

        while let Some(branch) = self.branches.last_mut() {
            let active = branch.active_thread;
            if active < branch.threads.len() && branch.threads[active] == ThreadStatus::Active {
                branch.threads[active] = ThreadStatus::Visited;
                // Save explored accesses for future sleep set propagation.
                let accesses = branch.active_accesses.clone();
                branch.explored_accesses.insert(active, accesses);
                // Add to sleep set: this thread has been explored at this position.
                if active < branch.sleep.len() {
                    branch.sleep[active] = true;
                }
                branch.active_accesses.clear();
            }

            // Remove current thread from wakeup tree (already explored).
            branch.wakeup.remove_branch(active);

            // Find next thread to explore from wakeup tree.
            // The search strategy controls the sibling ordering ≺.
            // DFS uses min_thread() (lowest ID first). Other strategies
            // use permutations over children for exploration diversity.
            //
            // Note: changing the ≺ ordering affects sleep set effectiveness
            // (Def 6.1 Property 2 depends on exploration order) but preserves
            // soundness — every explored trace is a valid Mazurkiewicz class.
            // The total trace count may differ from DFS (some redundant classes
            // explored, some sleep set optimizations lost). This is the right
            // trade-off for stop_on_first=True where finding the first bug fast
            // matters more than minimizing total exploration.
            let next_thread = {
                let threads = branch.wakeup.root_threads();
                if threads.is_empty() {
                    None
                } else {
                    let mut sorted = threads;
                    sorted.sort_unstable();
                    let n = sorted.len();
                    Some(match &self.search_strategy {
                        SearchStrategy::Dfs => sorted[0],
                        SearchStrategy::BitReversal { seed } => {
                            let idx = bit_reversal_index(self.selection_counter, *seed, n);
                            self.selection_counter += 1;
                            sorted[idx]
                        }
                        SearchStrategy::RoundRobin { seed } => {
                            let idx = ((self.selection_counter.wrapping_add(*seed)) % n as u64) as usize;
                            self.selection_counter += 1;
                            sorted[idx]
                        }
                        SearchStrategy::Stride { seed } => {
                            let stride = coprime_stride(*seed, n);
                            let idx = ((self.selection_counter.wrapping_mul(stride as u64)) % n as u64) as usize;
                            self.selection_counter += 1;
                            sorted[idx]
                        }
                        SearchStrategy::ConflictFirst => sorted[n - 1],
                    })
                }
            };
            if let Some(next) = next_thread {
                let subtree = branch.wakeup.subtree(next);
                self.pending_wakeup_subtree =
                    if subtree.is_empty() { None } else { Some(subtree) };

                branch.threads[next] = ThreadStatus::Active;
                branch.active_thread = next;
                // Replacing the active thread changes whether this branch's
                // decision is a preemption, so the count cached in Branch::new
                // (for the original thread) is now stale. Recompute it against
                // the previous branch's active thread (finding 3). Deeper
                // branches built afterward inherit from this corrected value
                // via schedule().
                let idx = self.branches.len() - 1;
                self.recompute_preemptions(idx);
                self.pos = 0;
                return true;
            }

            self.branches.pop();
        }
        false
    }

    /// Recompute `self.branches[idx].preemptions` after its active thread has
    /// been replaced (finding 3).
    ///
    /// Mirrors the formula in `schedule()`: the decision at `idx` is a
    /// preemption iff the new active thread differs from the previous branch's
    /// active thread AND that previous thread was still schedulable at this
    /// branch (i.e. not Disabled/Blocked here — Visited counts because it was
    /// runnable when scheduled). The base is the previous branch's (already
    /// corrected) preemption count, or 0 at the root.
    fn recompute_preemptions(&mut self, idx: usize) {
        let active = self.branches[idx].active_thread;
        let (prev_preemptions, prev_active) = if idx == 0 {
            (0, None)
        } else {
            (self.branches[idx - 1].preemptions, Some(self.branches[idx - 1].active_thread))
        };
        let is_preemption = match prev_active {
            Some(prev) => {
                prev != active
                    && self.branches[idx]
                        .threads
                        .get(prev)
                        .is_some_and(|s| !matches!(s, ThreadStatus::Disabled | ThreadStatus::Blocked))
            }
            None => false,
        };
        self.branches[idx].preemptions = prev_preemptions + u32::from(is_preemption);
    }

    /// Check if there are intermediate scheduling steps between `e_pos` and
    /// `e_prime_pos` that access objects from a different resource group than
    /// the racing object's group. This indicates cross-table interference
    /// that would cause backtrack explosion if handled with inline wakeup.
    pub fn has_cross_group_intermediates(
        &self,
        e_pos: usize,
        e_prime_pos: usize,
        resource_groups: &HashMap<u64, u64>,
    ) -> bool {
        if e_pos >= self.branches.len() {
            return false;
        }
        // Collect groups of the racing event's accesses
        let e_groups: std::collections::HashSet<u64> = self.branches[e_pos]
            .active_accesses
            .keys()
            .filter_map(|obj| resource_groups.get(obj).copied())
            .collect();
        if e_groups.is_empty() {
            return false;
        }

        for pos in (e_pos + 1)..e_prime_pos {
            if pos >= self.branches.len() {
                break;
            }
            let step = &self.branches[pos];
            // Skip same-thread events
            if step.active_thread == self.branches[e_pos].active_thread {
                continue;
            }
            for obj_id in step.active_accesses.keys() {
                if let Some(&group) = resource_groups.get(obj_id) {
                    if !e_groups.contains(&group) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Compute the notdep sequence for a race between events at positions
    /// `e_pos` and `e_prime_pos`, where `e_prime_thread` performed e'.
    ///
    /// Paper ref: Algorithm 2 line 4 (JACM'17 p.24):
    ///   v = notdep(e, E).e'
    /// The notdep sequence contains threads of events between e and e' that
    /// are **independent** of e, followed by the racing thread e'.
    ///
    /// An event at position i is excluded from notdep (i.e., is dependent on e) if:
    ///   1. It is by the same thread as e (JACM'17 Def 3.3 p.13: events by the
    ///      same process are ALWAYS dependent regardless of object accesses), OR
    ///   2. Its accesses conflict with e's accesses on a shared object
    ///      (write-write, read-write, etc.)
    ///
    /// The resulting sequence, when inserted into the wakeup tree at e's position,
    /// guides exploration to reverse the race: first replay the independent events,
    /// then execute e' before e.
    pub fn compute_notdep(
        &self,
        e_pos: usize,
        e_prime_pos: usize,
        e_prime_thread: usize,
    ) -> Vec<usize> {
        let (e_accesses, e_thread) = if e_pos < self.branches.len() {
            (
                &self.branches[e_pos].active_accesses,
                self.branches[e_pos].active_thread,
            )
        } else {
            return vec![e_prime_thread];
        };

        // Compute the transitive happens-after closure of e (finding 2,
        // JACM'17 Def 3.2 / Alg.2 line 4). An event is excluded from notdep
        // (i.e. it happens-after e) if it:
        //   1. is by the same thread as e (program order from e), OR
        //   2. conflicts with e (direct dependency), OR
        //   3. is by a thread that already has an excluded event before it
        //      (program order from that excluded event), OR
        //   4. conflicts with any already-excluded event (conflict chain).
        // A single forward pass suffices because happens-after only flows
        // forward in trace order: once an event is excluded we accumulate its
        // thread (for program-order propagation) and its accesses (for
        // conflict-chain propagation).
        let mut excluded_threads: std::collections::HashSet<usize> = std::collections::HashSet::new();
        excluded_threads.insert(e_thread);
        let mut excluded_accesses: HashMap<u64, (AccessKind, AccessOrigin)> = e_accesses.clone();

        let mut sequence = Vec::new();
        for pos in (e_pos + 1)..e_prime_pos {
            if pos < self.branches.len() {
                let step_thread = self.branches[pos].active_thread;
                let step_accesses = &self.branches[pos].active_accesses;
                let happens_after = excluded_threads.contains(&step_thread)
                    || !accesses_are_independent(&excluded_accesses, step_accesses);
                if happens_after {
                    // This event happens-after e: exclude it and extend the
                    // closure with its thread (program order) and accesses
                    // (conflict chain).
                    excluded_threads.insert(step_thread);
                    for (obj, (kind, origin)) in step_accesses {
                        excluded_accesses
                            .entry(*obj)
                            .and_modify(|(k, o)| {
                                *k = k.merge(*kind);
                                *o = o.merge(*origin);
                            })
                            .or_insert((*kind, *origin));
                    }
                } else {
                    sequence.push(step_thread);
                }
            }
        }
        // Append e' thread — the thread whose execution reverses the race
        sequence.push(e_prime_thread);

        sequence
    }

    /// Process all deferred races collected during the just-completed execution.
    ///
    /// Paper ref: Algorithm 2 lines 1-6 (JACM'17 p.24). At a maximal execution
    /// (all threads finished or blocked), we process each detected race and
    /// insert notdep sequences into wakeup trees at the appropriate positions.
    ///
    /// **Hybrid approach**: The engine also calls `insert_wakeup()` inline
    /// during execution for each race (Algorithm 1 style). The deferred notdep
    /// processing here only adds ADDITIONAL coverage: notdep sequences whose
    /// first thread differs from the racing thread AND the racing thread is
    /// already covered by the inline insertion. When notdep starts with a
    /// different thread, it provides multi-step guidance through independent
    /// intermediates.
    ///
    /// Races are deduplicated by (prev_path_id, thread_id).
    pub fn process_deferred_races(&mut self, races: &[PendingRace]) {
        // Deduplicate: for each (prev_path_id, thread_id), keep the race with
        // the latest current_path_id (longest notdep sequence).
        let mut best: HashMap<(usize, usize), &PendingRace> = HashMap::new();
        for race in races {
            let key = (race.prev_path_id, race.thread_id);
            best.entry(key)
                .and_modify(|existing| {
                    if race.current_path_id > existing.current_path_id {
                        *existing = race;
                    }
                })
                .or_insert(race);
        }
        for race in best.values() {
            self.process_one_deferred_race(race);
        }
    }

    /// Process a single deferred race by computing its notdep sequence and
    /// inserting it into the wakeup tree at the race's first event position.
    ///
    /// Paper ref: Algorithm 2 lines 3-6 (JACM'17 p.24):
    ///   let e = race(E, e')                    — the first racing event
    ///   let v = notdep(e, E).e'                — independent prefix + racing thread
    ///   if sleep(E') ∩ WI[E'](v) = ∅ then      — not sleep-set blocked
    ///     wut(E') := insert[E'](v, wut(E'))    — add to wakeup tree
    ///
    /// **Hybrid semantics**: The engine may perform inline `insert_wakeup()`
    /// for the racing thread (Algorithm 1 style) during execution. For
    /// objects without resource groups, single-thread notdep sequences
    /// (len == 1) are already covered by inline insertion. For objects
    /// with resource groups (Defect #15), inline insertion is skipped,
    /// so deferred processing must handle all notdep lengths.
    ///
    /// When `notdep = [thread_id]` (no intermediates), we fall back to
    /// `insert_wakeup()` which deduplicates against existing entries.
    /// When `notdep = [T_indep, ..., thread_id]`, we insert the full
    /// sequence for multi-step guidance.
    fn process_one_deferred_race(&mut self, race: &PendingRace) {
        let path_id = race.prev_path_id;
        if path_id >= self.branches.len() {
            return;
        }

        let notdep = self.compute_notdep(path_id, race.current_path_id, race.thread_id);

        if notdep.is_empty() {
            return;
        }

        // For races where inline insertion was skipped (resource-grouped
        // objects, Defect #15), single-thread notdep must be handled here.
        // For races with inline insertion, len==1 is already covered.
        if notdep.len() <= 1 {
            if race.inline_skipped && notdep.len() == 1 {
                self.insert_wakeup(path_id, notdep[0], race.race_object);
            }
            return;
        }

        let first_thread = notdep[0];

        // Only insert if the first independent thread is feasible at this position.
        if !self.can_insert_thread(path_id, first_thread) {
            return;
        }

        // Preemption bound check for the first thread
        if let Some(bound) = self.preemption_bound {
            let branch = &self.branches[path_id];
            if branch.active_thread != first_thread && branch.preemptions >= bound {
                return;
            }
        }

        // Insert the full notdep sequence into the wakeup tree.
        // Paper: Algorithm 2 line 6 (JACM'17 p.24):
        //   wut(E') := insert[E'](v, wut(E'))
        // Multi-step sequences guide exploration via subtree extraction
        // (Phase 4b: Algorithm 2 line 17, WuT' = subtree(wut(E), p)).
        self.branches[path_id].wakeup.insert(&notdep);
    }

    /// Check if `thread_id` can be inserted at `path_id` (runnable, not sleeping).
    fn can_insert_thread(&self, path_id: usize, thread_id: usize) -> bool {
        let branch = &self.branches[path_id];
        // Runnability check
        match branch.threads.get(thread_id).copied() {
            Some(ThreadStatus::Pending) | Some(ThreadStatus::Yield) => {}
            _ => return false,
        }
        // Sleep set check (local + propagated)
        // Paper: Algorithm 2 line 5 (JACM'17 p.24):
        //   sleep(E') ∩ WI[E'](v) = ∅
        if branch.sleep.get(thread_id).copied().unwrap_or(false) {
            return false;
        }
        if branch.propagated_sleep_accesses.contains_key(&thread_id) {
            return false;
        }
        true
    }

    fn add_conservative_wakeup(&mut self, path_id: usize, thread_id: usize, bound: u32) {
        for i in (0..path_id).rev() {
            let branch = &self.branches[i];
            if let Some(status) = branch.threads.get(thread_id) {
                let would_preempt = branch.active_thread != thread_id && status.is_runnable();
                if matches!(status, ThreadStatus::Pending | ThreadStatus::Yield)
                    && (!would_preempt || branch.preemptions < bound)
                    && !branch.wakeup.contains_thread(thread_id)
                    && !branch.sleep.get(thread_id).copied().unwrap_or(false)
                    && !branch.propagated_sleep_accesses.contains_key(&thread_id)
                {
                    self.branches[i].wakeup.insert(&[thread_id]);
                    return;
                }
            }
        }
    }
}


/// Compute a stride value coprime to n, derived from the seed.
///
/// For n=1, returns 1. For n>1, tries `seed % (n-1) + 1` and adjusts
/// until gcd(stride, n) == 1.
fn coprime_stride(seed: u64, n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut s = (seed % (n as u64 - 1) + 1) as usize; // 1..n-1
    while gcd(s, n) != 1 {
        s += 1;
        if s >= n {
            s = 1;
        }
    }
    s
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Compute a bit-reversal permutation index.
///
/// Given a counter `step` and `n` items, returns an index in `0..n` such that
/// consecutive steps visit maximally spread positions (low-discrepancy sequence).
///
/// Uses a van der Corput-style construction: reverse the bits of `(step + seed)`
/// in the smallest power-of-two domain >= n, then map to `0..n`.
///
/// For any prefix of `n` consecutive calls, the returned indices cover all `n`
/// positions exactly once (when n is a power of 2) or approximately uniformly
/// (otherwise). Over `n` calls the full permutation is generated.
fn bit_reversal_index(step: u64, seed: u64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    // Find smallest power of 2 >= n
    let bits = (n as u64).next_power_of_two().trailing_zeros();
    let mask = (1u64 << bits) - 1;
    // Try successive values until we land in [0, n)
    // For power-of-two n this always succeeds on first try.
    // For non-power-of-two, we skip out-of-range values.
    let mut attempt = step;
    loop {
        let val = (attempt.wrapping_add(seed)) & mask;
        let reversed = val.reverse_bits() >> (64 - bits);
        if (reversed as usize) < n {
            return reversed as usize;
        }
        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reversal_covers_all() {
        // For n=4 (power of 2), 4 steps should cover all indices
        let indices: Vec<usize> = (0..4).map(|s| bit_reversal_index(s, 0, 4)).collect();
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3], "bit reversal should cover all 4 indices, got {:?}", indices);
    }

    #[test]
    fn test_bit_reversal_spreads() {
        // First two indices for n=4 should be maximally spread (0 and 2, or similar)
        let first = bit_reversal_index(0, 0, 4);
        let second = bit_reversal_index(1, 0, 4);
        assert_ne!(first, second);
        // They should be far apart (differ by n/2)
        let diff = (first as i64 - second as i64).unsigned_abs() as usize;
        assert_eq!(diff, 2, "first two should be n/2 apart for maximal spread");
    }

    #[test]
    fn test_bit_reversal_seed_rotates() {
        // Different seeds should produce different orderings
        let order_s0: Vec<usize> = (0..4).map(|s| bit_reversal_index(s, 0, 4)).collect();
        let order_s1: Vec<usize> = (0..4).map(|s| bit_reversal_index(s, 1, 4)).collect();
        assert_ne!(order_s0, order_s1);
        // But both should cover all indices
        let mut sorted_s1 = order_s1;
        sorted_s1.sort();
        assert_eq!(sorted_s1, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bit_reversal_non_power_of_two() {
        // For n=3, should still cover all indices in 3 steps
        let indices: Vec<usize> = (0..3).map(|s| bit_reversal_index(s, 0, 3)).collect();
        let mut sorted = indices;
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_bit_reversal_n1() {
        assert_eq!(bit_reversal_index(0, 0, 1), 0);
        assert_eq!(bit_reversal_index(5, 42, 1), 0);
    }

    /// step() replacing a branch's active thread must recompute that branch's
    /// preemption count (finding 3).
    ///
    /// Build: pos0 = T0 (no preemption), pos1 = T0 again (no preemption,
    /// preemptions=0). Insert a wakeup for T1 at pos1, then step(): step picks
    /// T1 at pos1, switching away from the still-runnable T0 — that IS a
    /// preemption, so the branch's preemption count must become 1. Before the
    /// fix it stayed at 0 (the count computed in Branch::new for the original
    /// thread T0).
    #[test]
    fn test_step_recomputes_preemptions_on_active_thread_replacement() {
        let mut path = Path::new(Some(8), SearchStrategy::Dfs);

        // pos 0: runnable [0,1], current 0 -> T0
        assert_eq!(path.schedule(&[0, 1], 0, 2), Some(0));
        // pos 1: runnable [0,1], current 0 -> T0 (no preemption)
        assert_eq!(path.schedule(&[0, 1], 0, 2), Some(0));
        assert_eq!(path.branches[1].preemptions, 0);

        // Insert a wakeup for T1 at pos 1 and step into it.
        path.insert_wakeup(1, 1, None);
        assert!(path.step());

        // pos 1's active thread is now T1, switching away from runnable T0:
        // that is a preemption, so the count must be 1, not the stale 0.
        assert_eq!(path.branches[1].active_thread, 1);
        assert_eq!(
            path.branches[1].preemptions, 1,
            "switching to T1 away from still-runnable T0 is a preemption"
        );
    }

    /// step() replacing the root branch's active thread keeps preemptions at 0
    /// (there is no previous thread to preempt) — guards against off-by-one in
    /// the recomputation (finding 3).
    #[test]
    fn test_step_recomputes_preemptions_at_root() {
        let mut path = Path::new(Some(8), SearchStrategy::Dfs);

        // pos 0: runnable [0,1], current 0 -> T0
        assert_eq!(path.schedule(&[0, 1], 0, 2), Some(0));
        assert_eq!(path.branches[0].preemptions, 0);

        path.insert_wakeup(0, 1, None);
        assert!(path.step());

        // Root branch now starts with T1; with no preceding thread there is
        // nothing to preempt, so the count stays 0.
        assert_eq!(path.branches[0].active_thread, 1);
        assert_eq!(path.branches[0].preemptions, 0, "root branch has no thread to preempt");
    }

    #[test]
    fn test_new_path() {
        let path = Path::new(None, SearchStrategy::Dfs);
        assert_eq!(path.depth(), 0);
        assert_eq!(path.current_position(), 0);
    }

    #[test]
    fn test_schedule_prefers_current_thread() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        assert_eq!(path.schedule(&[0, 1], 0, 2), Some(0));
        assert_eq!(path.depth(), 1);
    }

    /// A local scheduling preference must not replace wakeup-tree guidance.
    ///
    /// The pending subtree is the suffix of a multi-step notdep sequence and
    /// is required to reverse a previously observed race.  Dropping it to run
    /// a newly enabled clock actor can lose that race-reversing execution.
    #[test]
    fn test_schedule_preference_does_not_override_pending_wakeup_sequence() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        let mut pending = WakeupTree::empty();
        pending.insert(&[2]);
        path.pending_wakeup_subtree = Some(pending);

        assert_eq!(
            path.schedule_preferred(&[0, 1, 2], 0, 3, Some(1)),
            Some(2),
            "the pending race-reversing sequence must remain authoritative"
        );
    }

    #[test]
    fn test_insert_wakeup_and_step() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1], 0, 2);
        path.insert_wakeup(0, 1, None);
        assert!(path.step());
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
    }

    #[test]
    fn test_step_exhausted() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1], 0, 2);
        assert!(!path.step());
    }

    #[test]
    fn test_full_exploration_two_threads() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        let mut executions = Vec::new();

        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(chosen);
        path.insert_wakeup(0, 1, None);

        assert!(path.step());
        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(chosen);

        assert!(!path.step());
        assert_eq!(executions, vec![0, 1]);
    }

    #[test]
    fn test_preemption_bounding_zero() {
        let mut path = Path::new(Some(0), SearchStrategy::Dfs);
        path.schedule(&[0, 1], 0, 2);
        path.insert_wakeup(0, 1, None);
        // With bound=0, wakeup insertion for thread 1 is a preemption and should be suppressed
        assert!(!path.step());
    }

    // --- Source Sets / Wakeup Tree specific tests ---

    #[test]
    fn test_wakeup_tree_min_index_ordering() {
        // step() picks the minimum thread ID from the wakeup tree
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1, 2], 0, 3);
        path.insert_wakeup(0, 2, None); // add 2 first
        path.insert_wakeup(0, 1, None); // add 1 second

        // step() should pick the minimum index (1), not insertion order (2)
        assert!(path.step());
        let chosen = path.schedule(&[0, 1, 2], 0, 3);
        assert_eq!(chosen, Some(1));
    }

    #[test]
    fn test_sleep_set_visited_thread() {
        use crate::access::AccessKind;
        // After exploring thread 0 at position 0, it should be in the sleep set
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1, 2], 0, 3);
        // Record some access for thread 0
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.insert_wakeup(0, 1, None);

        // step() marks 0 as Visited and adds to sleep set
        assert!(path.step());

        // Now if a race suggests inserting 0 into the wakeup tree at position 0, it should be skipped
        path.insert_wakeup(0, 0, None); // 0 is Visited, so this should be a no-op
        // The wakeup tree should not have thread 0 (only thread 1 was from step())
        // This just verifies the Visited check prevents re-adding
    }

    #[test]
    fn test_visited_thread_excluded_from_wakeup() {
        use crate::access::AccessKind;
        // After exploring T0 at position 0, T0 is Visited and in the sleep set.
        // Wakeup insertions for T0 at position 0 should be rejected.
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);

        // Add T1 to wakeup tree
        path.insert_wakeup(0, 1, None);

        // step() → T0 becomes Visited, sleep[0] = true. T1 becomes Active.
        assert!(path.step());
        assert!(path.branches[0].sleep[0]);
        assert_eq!(path.branches[0].threads[0], ThreadStatus::Visited);
        assert_eq!(path.branches[0].threads[1], ThreadStatus::Active);

        // Try to insert T0 at position 0 — rejected because T0 is Visited
        path.insert_wakeup(0, 0, None);
        // T2 should still be addable
        path.insert_wakeup(0, 2, None);
        assert!(path.branches[0].wakeup.contains_thread(2));
    }

    #[test]
    fn test_duplicate_wakeup_rejected() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1], 0, 2);
        path.insert_wakeup(0, 1, None);
        path.insert_wakeup(0, 1, None); // duplicate
        // Wakeup tree should have only one entry
        assert_eq!(path.branches[0].wakeup.root_threads(), vec![1]);
    }

    #[test]
    fn test_wakeup_disabled_thread_rejected() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0], 0, 2); // Only thread 0 is runnable; thread 1 is Disabled
        path.insert_wakeup(0, 1, None);
        assert!(path.branches[0].wakeup.is_empty());
    }

    #[test]
    fn test_race_object_passed_through() {
        // All racing threads should be added regardless of object
        // (source set filtering is disabled for soundness without sleep sets)
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1, 2, 3], 0, 4);

        path.insert_wakeup(0, 1, Some(100));
        path.insert_wakeup(0, 2, Some(100));
        path.insert_wakeup(0, 3, Some(200));

        assert_eq!(path.branches[0].wakeup.root_threads(), vec![1, 2, 3]);
    }

    /// Verify that `contains_thread` in `insert_wakeup()` correctly deduplicates:
    /// same thread is rejected, different threads (even on same object) are added.
    #[test]
    fn test_source_set_check_via_contains_thread() {
        let mut path = Path::new(None, SearchStrategy::Dfs);
        path.schedule(&[0, 1, 2, 3], 0, 4);

        // First wakeup insertion for object 100: T1 added
        path.insert_wakeup(0, 1, Some(100));
        assert!(path.branches[0].wakeup.contains_thread(1));

        // Different thread, same object: T2 also added (different initials)
        path.insert_wakeup(0, 2, Some(100));
        assert!(path.branches[0].wakeup.contains_thread(2));

        // Same thread again: duplicate rejected by contains_thread check
        path.insert_wakeup(0, 1, Some(100));
        assert_eq!(path.branches[0].wakeup.root_threads(), vec![1, 2]);
    }

    // --- Sleep set propagation tests ---

    /// Test that sleep set propagation works during REPLAY of existing branches.
    ///
    /// Scenario: After exploring T0 and T1 at pos 1, when replaying through
    /// pos 0→pos 1, T1's local sleep at pos 1 should be checked against
    /// T0's active_accesses at pos 0 for independence.
    ///
    /// Paper ref: Algorithm 2 line 16 (JACM'17 p.24):
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    #[test]
    fn test_propagated_sleep_during_replay() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // --- First execution: T0 at pos 0, T1 at pos 1 ---
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1, 2], 1, 3);
        path.record_access(1, 200, AccessKind::Read, AccessOrigin::PythonMemory);

        // Add T1 to wakeup tree at pos 0 and T2 at pos 1
        path.insert_wakeup(0, 1, None);
        path.insert_wakeup(1, 2, None);

        // step(): pops to pos 1 (T1 Visited, wakeup has T2). T2 active at pos 1.
        assert!(path.step());

        // --- Second execution: replay pos 0 (T0), replay pos 1 (T2) ---
        let chosen = path.schedule(&[0, 1, 2], 0, 3);
        assert_eq!(chosen, Some(0)); // replay T0
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory); // T0 writes obj 100

        let chosen = path.schedule(&[1, 2], 1, 3);
        assert_eq!(chosen, Some(2)); // replay T2 (from wakeup)

        // Propagation from pos 0 to pos 1 during replay:
        // No threads are sleeping at pos 0 (T0 is Active, never Visited there).
        // T1 is locally sleeping at pos 1 (Visited), but that's handled by
        // the local sleep check in insert_wakeup(), not by propagation.
        // Propagation only carries sleeping threads FROM the source position.
        assert!(
            path.branches[1].propagated_sleep_accesses.is_empty(),
            "Nothing propagated from pos 0 (no sleeping threads there)"
        );
        // T1 IS locally sleeping at pos 1
        assert!(path.branches[1].sleep[1]);
    }

    /// Test that propagation works for new branches via trace caching (Phase 2b).
    ///
    /// The trace cache records per-thread access unions from the previous
    /// execution, enabling sleep set propagation to new branches (beyond the
    /// replay prefix). When a sleeping thread's cached accesses are
    /// independent of the active thread's accesses, it stays asleep.
    ///
    /// Paper ref: JACM'17 Section 10 (p.31-35) — Concuerror uses trace
    /// caching to determine sleeping threads' next actions. The independence
    /// check Sleep' = {q ∈ sleep(E) | E ⊢ p♦q} (Algorithm 2 line 16, p.24)
    /// uses the cached accesses for q's actions.
    #[test]
    fn test_propagation_to_new_branches_via_trace_cache() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // --- First execution: T0 writes obj 100, T1 reads obj 200 ---
        path.schedule(&[0, 1], 0, 2);
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2);
        path.record_access(1, 200, AccessKind::Read, AccessOrigin::PythonMemory);

        // Backtrack T1 at pos 0
        path.insert_wakeup(0, 1, None);

        // step(): saves trace, T0 → Visited at pos 0
        assert!(path.step());

        // --- Second execution: T1 at pos 0 (replay), then new pos 1 ---
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 200, AccessKind::Read, AccessOrigin::PythonMemory); // T1 reads different object

        // New branch at pos 1: T0 chosen
        path.schedule(&[0], 0, 2);

        // With trace caching, propagation to new branches is enabled.
        // T0's cached accesses = {100: Write}, T1's active at pos 0 = {200: Read}.
        // Disjoint objects → INDEPENDENT → T0 stays asleep at pos 1.
        assert!(
            path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 should be propagated to new branch (independent: obj 100 vs obj 200)"
        );
    }

    /// Test that trace caching correctly wakes threads on conflict at new branches.
    ///
    /// When a sleeping thread's cached accesses conflict with the active
    /// thread's accesses, the sleeping thread must be woken up to allow
    /// exploring the conflicting interleaving.
    ///
    /// Paper ref: Algorithm 2 line 16 (JACM'17 p.24) — threads are removed
    /// from Sleep' when their action is dependent (¬(p♦q)).
    #[test]
    fn test_trace_cache_wakes_on_conflict_at_new_branch() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // --- First execution: T0 writes obj 100, T1 writes obj 100 ---
        path.schedule(&[0, 1], 0, 2);
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2);
        path.record_access(1, 100, AccessKind::Write, AccessOrigin::PythonMemory);

        // Backtrack T1 at pos 0
        path.insert_wakeup(0, 1, None);

        // step(): saves trace, T0 → Visited
        assert!(path.step());

        // --- Second execution: T1 at pos 0 (replay), then new pos 1 ---
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory); // T1 writes same obj 100

        // New branch at pos 1: T0 chosen
        path.schedule(&[0], 0, 2);

        // T0's cached accesses = {100: Write}, T1's active at pos 0 = {100: Write}.
        // Write vs Write on obj 100 → CONFLICT → T0 wakes up.
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 should NOT be propagated (Write vs Write conflict on obj 100)"
        );
    }

    // --- Deferred race / notdep tests (Phase 3) ---

    /// Test compute_notdep with independent intermediate events.
    ///
    /// Setup: T0 writes obj 1 at pos 0, T1 writes obj 2 at pos 1,
    /// T2 writes obj 1 at pos 2. Race between pos 0 and pos 2.
    /// T1's access (obj 2) is independent of T0's (obj 1).
    /// Expected notdep: [1, 2] — T1 is independent, then T2 races.
    #[test]
    fn test_compute_notdep_basic() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // Build 3 branches manually
        path.schedule(&[0, 1, 2], 0, 3); // pos 0: T0
        path.record_access(0, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1, 2], 1, 3); // pos 1: T1
        path.record_access(1, 2, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[2], 2, 3); // pos 2: T2
        path.record_access(2, 1, AccessKind::Write, AccessOrigin::PythonMemory);

        let notdep = path.compute_notdep(0, 2, 2);
        assert_eq!(notdep, vec![1, 2], "T1 is independent of T0, then T2 races");
    }

    /// Test compute_notdep excludes dependent intermediate events.
    ///
    /// Setup: T0 writes obj 1 at pos 0, T1 writes obj 1 at pos 1 (DEPENDENT),
    /// T2 writes obj 1 at pos 2. Race between pos 0 and pos 2.
    /// T1's access (obj 1, Write) conflicts with T0's (obj 1, Write).
    /// Expected notdep: [2] — T1 is dependent, only T2 remains.
    #[test]
    fn test_compute_notdep_dependent_intermediate() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1, 2], 1, 3);
        path.record_access(1, 1, AccessKind::Write, AccessOrigin::PythonMemory); // same object — dependent
        path.schedule(&[2], 2, 3);
        path.record_access(2, 1, AccessKind::Write, AccessOrigin::PythonMemory);

        let notdep = path.compute_notdep(0, 2, 2);
        assert_eq!(notdep, vec![2], "T1 is dependent (same obj Write), excluded");
    }

    /// Test compute_notdep with adjacent racing events (no intermediates).
    #[test]
    fn test_compute_notdep_adjacent() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        path.schedule(&[0, 1], 0, 2);
        path.record_access(0, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2);
        path.record_access(1, 1, AccessKind::Write, AccessOrigin::PythonMemory);

        let notdep = path.compute_notdep(0, 1, 1);
        assert_eq!(notdep, vec![1], "Adjacent race: just the racing thread");
    }

    /// Test compute_notdep excludes same-thread events.
    ///
    /// JACM'17 Def 3.3 (p.13): events by the same process are ALWAYS
    /// dependent regardless of object access patterns.
    #[test]
    fn test_compute_notdep_excludes_same_thread() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // T0 at pos 0, T0 again at pos 1 (same thread!), T1 at pos 2
        path.schedule(&[0, 1], 0, 2);
        path.record_access(0, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 0, 2); // T0 runs again
        path.record_access(1, 2, AccessKind::Read, AccessOrigin::PythonMemory); // different obj, but same thread
        path.schedule(&[1], 1, 2);
        path.record_access(2, 1, AccessKind::Write, AccessOrigin::PythonMemory);

        let notdep = path.compute_notdep(0, 2, 1);
        assert_eq!(
            notdep,
            vec![1],
            "Same-thread event at pos 1 excluded even though objects differ"
        );
    }

    /// Test compute_notdep excludes events that happen-after e transitively
    /// through program order (finding 2).
    ///
    /// JACM'17 Def 3.2 / Alg.2 line 4: notdep(e, E) excludes EVERY event that
    /// happens-after e, transitively. The closure must follow program order:
    /// if T1's first event (pos1) conflicts with e and is excluded, then T1's
    /// later event (pos2) also happens-after e (program order through pos1)
    /// and must be excluded too — even though pos2's accesses do not directly
    /// conflict with e.
    ///
    /// Setup:
    ///   pos0: e  = T0 W(X)        — the first racing event
    ///   pos1:      T1 W(X)        — conflicts with e, excluded
    ///   pos2:      T1 W(Y)        — happens-after e via T1 program order, must
    ///                               be excluded (does NOT conflict with e)
    ///   pos3: e' = T2 W(X)        — the reversing event
    ///
    /// Correct notdep: [2]  (both T1 events follow e; only e' remains).
    /// Buggy notdep:   [1, 2]  (pos2 wrongly included as independent).
    #[test]
    fn test_compute_notdep_transitive_happens_after() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        path.schedule(&[0, 1, 2], 0, 3); // pos 0: T0 W(X)
        path.record_access(0, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1, 2], 1, 3); // pos 1: T1 W(X) — conflicts with e
        path.record_access(1, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1, 2], 1, 3); // pos 2: T1 W(Y) — program-order after pos1
        path.record_access(2, 2, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[2], 2, 3); // pos 3: T2 W(X)
        path.record_access(3, 1, AccessKind::Write, AccessOrigin::PythonMemory);

        let notdep = path.compute_notdep(0, 3, 2);
        assert_eq!(
            notdep,
            vec![2],
            "pos2 (T1 W(Y)) happens-after e transitively via T1 program order and must be excluded"
        );
    }

    /// Test compute_notdep excludes events that happen-after e transitively
    /// through a conflict chain (finding 2).
    ///
    /// Setup:
    ///   pos0: e  = T0 W(X)        — racing event
    ///   pos1:      T1 W(X)        — conflicts with e, excluded
    ///   pos2:      T2 W(Y)        — independent of e directly, but conflicts
    ///                               with pos1 (also W(Y))... see below
    ///
    /// To exercise the conflict-chain closure: pos1 = T1 writes Y (conflicts
    /// with e? no). Use: pos1 = T1 W(X) excluded; pos2 = T2 W(Z) conflicts with
    /// pos1's *other* access. We keep it simple: an event conflicting with an
    /// already-excluded event is itself excluded.
    #[test]
    fn test_compute_notdep_conflict_chain_closure() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // pos0: e = T0 W(X=1)
        path.schedule(&[0, 1, 2, 3], 0, 4);
        path.record_access(0, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        // pos1: T1 W(X=1) conflicts with e -> excluded
        path.schedule(&[1, 2, 3], 1, 4);
        path.record_access(1, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        // pos2: T1 W(Y=2) program-order after pos1 -> excluded, also touches Y
        path.schedule(&[1, 2, 3], 1, 4);
        path.record_access(2, 2, AccessKind::Write, AccessOrigin::PythonMemory);
        // pos3: T2 W(Y=2) conflicts with the excluded pos2 -> must be excluded
        path.schedule(&[2, 3], 2, 4);
        path.record_access(3, 2, AccessKind::Write, AccessOrigin::PythonMemory);
        // pos4: e' = T3 W(X=1)
        path.schedule(&[3], 3, 4);
        path.record_access(4, 1, AccessKind::Write, AccessOrigin::PythonMemory);

        let notdep = path.compute_notdep(0, 4, 3);
        assert_eq!(
            notdep,
            vec![3],
            "pos3 (T2 W(Y)) conflicts with already-excluded pos2 and must be excluded"
        );
    }

    /// Test process_one_deferred_race inserts notdep into wakeup tree.
    #[test]
    fn test_process_deferred_race_inserts_notdep() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // T0 writes obj 1 at pos 0, T1 writes obj 2 at pos 1, T2 writes obj 1 at pos 2
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 1, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1, 2], 1, 3);
        path.record_access(1, 2, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[2], 2, 3);
        path.record_access(2, 1, AccessKind::Write, AccessOrigin::PythonMemory);

        let race = PendingRace {
            prev_path_id: 0,
            current_path_id: 2,
            thread_id: 2,
            race_object: Some(1),
            inline_skipped: false,
            is_attribute_race: false,
        };

        path.process_deferred_races(&[race]);

        // Should have inserted [1, 2] into wakeup tree at pos 0
        let wakeup = &path.branches[0].wakeup;
        assert!(wakeup.contains_thread(1), "notdep [1,2] starts with T1");
        let leaves = wakeup.leaf_sequences();
        assert_eq!(leaves, vec![vec![1, 2]], "full notdep sequence [1, 2]");
    }

    /// Test that record_access merges Read + WeakRead to Read (not Write).
    ///
    /// Bug: record_access() unconditionally upgrades to Write when the same
    /// object is accessed with two different AccessKinds. The correct behavior
    /// is to use AccessKind::merge() which handles Read+WeakRead as Read.
    ///
    /// This matters for sleep set independence: a merged Read is independent
    /// of WeakRead (no conflict), but Write conflicts with everything.
    /// Incorrect merge causes spurious sleep set wakeups.
    #[test]
    fn test_record_access_merges_read_weakread_to_read() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        path.schedule(&[0, 1], 0, 2);

        // Thread 0 does Read then WeakRead on the same object
        path.record_access(0, 200, AccessKind::Read, AccessOrigin::PythonMemory);
        path.record_access(0, 200, AccessKind::WeakRead, AccessOrigin::PythonMemory);

        // The merged kind should be Read (Read subsumes WeakRead),
        // NOT Write (which would cause false conflicts with WeakRead).
        let (merged_kind, merged_origin) = path.branches[0].active_accesses.get(&200).unwrap();
        assert_eq!(
            *merged_kind,
            AccessKind::Read,
            "Read + WeakRead should merge to Read via AccessKind::merge(), \
             but got {:?}. record_access() is incorrectly upgrading to Write.",
            merged_kind
        );
        assert_eq!(*merged_origin, AccessOrigin::PythonMemory);
    }

    /// Test that record_access merges WeakRead + Read to Read (commutative).
    #[test]
    fn test_record_access_merges_weakread_read_to_read() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        path.schedule(&[0, 1], 0, 2);

        // Reverse order: WeakRead then Read
        path.record_access(0, 200, AccessKind::WeakRead, AccessOrigin::PythonMemory);
        path.record_access(0, 200, AccessKind::Read, AccessOrigin::PythonMemory);

        let (merged_kind, _) = path.branches[0].active_accesses.get(&200).unwrap();
        assert_eq!(
            *merged_kind,
            AccessKind::Read,
            "WeakRead + Read should merge to Read, but got {:?}.",
            merged_kind
        );
    }

    /// Test that record_access merges Read + Write to Write (correct).
    #[test]
    fn test_record_access_merges_read_write_to_write() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        path.schedule(&[0, 1], 0, 2);

        path.record_access(0, 200, AccessKind::Read, AccessOrigin::PythonMemory);
        path.record_access(0, 200, AccessKind::Write, AccessOrigin::PythonMemory);

        let (merged_kind, _) = path.branches[0].active_accesses.get(&200).unwrap();
        assert_eq!(
            *merged_kind,
            AccessKind::Write,
            "Read + Write should merge to Write, but got {:?}.",
            merged_kind
        );
    }

    /// Test that the buggy merge in record_access causes a false sleep set
    /// wakeup when a sleeping thread's WeakRead is checked against
    /// incorrectly-merged Write (should be Read).
    #[test]
    fn test_record_access_merge_bug_causes_false_wakeup() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // --- First execution: T0 at pos 0, T1 at pos 1 ---
        path.schedule(&[0, 1], 0, 2);
        // T0 does Read + WeakRead on obj 200
        path.record_access(0, 200, AccessKind::Read, AccessOrigin::PythonMemory);
        path.record_access(0, 200, AccessKind::WeakRead, AccessOrigin::PythonMemory);

        path.schedule(&[1], 1, 2);
        path.record_access(1, 200, AccessKind::WeakRead, AccessOrigin::PythonMemory); // T1 does WeakRead

        // Add T1 to wakeup tree at pos 0 to force backtracking
        path.insert_wakeup(0, 1, None);

        // step(): T0 → Visited at pos 0, T1 Active at pos 0
        assert!(path.step());

        // --- Second execution: T1 at pos 0 (replay), then new pos 1 ---
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 200, AccessKind::WeakRead, AccessOrigin::PythonMemory); // T1 WeakReads

        // New branch at pos 1: T0 chosen
        path.schedule(&[0], 0, 2);

        // T0 is sleeping. T0's cached accesses = {200: Read+WeakRead merged}.
        // T1's active at pos 0 = {200: WeakRead}.
        //
        // With correct merge (Read): Read vs WeakRead → NOT a conflict → T0 stays asleep.
        // With buggy merge (Write): Write vs WeakRead → conflict → T0 wakes up.
        assert!(
            path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 should stay asleep at pos 1 (Read vs WeakRead is independent). \
             If T0 woke up, record_access merged Read+WeakRead to Write instead of Read."
        );
    }

    /// Test that propagation correctly wakes threads on conflict during replay.
    #[test]
    fn test_propagated_sleep_wakes_on_conflict_during_replay() {
        use crate::access::AccessKind;
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // --- First execution: T0 at pos 0, T1 at pos 1 ---
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory); // T0 writes obj 100
        path.schedule(&[1, 2], 1, 3);
        path.record_access(1, 100, AccessKind::Read, AccessOrigin::PythonMemory); // T1 reads obj 100

        // Add to wakeup trees
        path.insert_wakeup(0, 2, None);
        path.insert_wakeup(1, 2, None);

        // step(): pops to pos 1 (T1 Visited, wakeup has T2). T2 active at pos 1.
        assert!(path.step());

        // --- Second execution: replay pos 0 (T0), replay pos 1 (T2) ---
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory); // T0 writes obj 100

        path.schedule(&[1, 2], 1, 3);

        // Propagation from pos 0 to pos 1:
        // T1 sleeping at pos 1 with explored_accesses[1] = {100: Read}.
        // T0's active at pos 0 = {100: Write}.
        // Write vs Read on same obj 100 → CONFLICT → T1 wakes up.
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&1),
            "T1 should NOT be propagated (Write vs Read conflict on obj 100)"
        );
    }

    // --- Position-sensitive future access cache tests (Fix 3) ---

    /// Test that exact steps support a lazily-derived full suffix union.
    #[test]
    fn test_step_builds_exact_steps_for_lazy_suffix_union() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // Build 3 scheduling points for thread 0 (T0 runs 3 times in a row):
        //   pos 0 (T0 step 0): writes obj 10
        //   pos 1 (T0 step 1): writes obj 20
        //   pos 2 (T0 step 2): writes obj 30
        // schedule() prefers current_thread, so T0 is chosen each time.
        // record_access uses path_id (position index).
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 10, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 0, 2); // pos 1: T0 (prefers current=0)
        path.record_access(1, 20, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 0, 2); // pos 2: T0
        path.record_access(2, 30, AccessKind::Write, AccessOrigin::PythonMemory);

        // Trigger step() to build the cache
        path.step();

        let accesses = path.future_accesses_for(0, 0).unwrap();
        assert!(accesses.contains_key(&10));
        assert!(accesses.contains_key(&20));
        assert!(accesses.contains_key(&30));
        assert_eq!(accesses.len(), 3);
    }

    /// Test that future_accesses_for() stops trusting an exhausted trace.
    #[test]
    fn test_future_accesses_for_counts_thread_steps() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // Thread 0 at pos 0 (writes obj 10), thread 1 at pos 1, thread 0 at pos 2 (writes obj 20)
        // To get T1 at pos 1, we pass current_thread=1 and runnable=[0,1].
        // Actually, schedule picks current_thread if runnable. So:
        //   pos 0: runnable=[0,1], current=0 → T0
        //   pos 1: runnable=[0,1], current=1 → T1
        //   pos 2: runnable=[0], current=0 → T0
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 10, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 1, 2); // pos 1: T1 (current_thread=1)
        path.record_access(1, 99, AccessKind::Read, AccessOrigin::PythonMemory); // T1 reads something unrelated
        path.schedule(&[0], 0, 2); // pos 2: T0
        path.record_access(2, 20, AccessKind::Write, AccessOrigin::PythonMemory);

        // Add a wakeup so step() keeps branches and starts a new execution
        path.insert_wakeup(1, 0, None);

        // Build cache (step pops to pos 1, T0 becomes active via wakeup)
        assert!(path.step());

        // After step(), branches are truncated to the backtrack point.
        // The cache was built from the full execution before truncation.
        // Now replay through the prefix to verify future_accesses_for.
        // Branches[0] still exists (T0 at pos 0). Use it to verify cache.

        let union = path.future_accesses_for(0, 0).unwrap();
        assert!(union.contains_key(&10));
        assert!(union.contains_key(&20));

        // Also verify future_accesses_for at pos=0 (T0 has 0 steps before pos 0)
        let f0 = path.future_accesses_for(0, 0).unwrap();
        assert!(f0.contains_key(&10));
        assert!(f0.contains_key(&20));

        // After one step the actor still has cached work, so the conservative
        // carrier remains available. Exact independence uses prev_thread_steps.
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(0)); // replay T0 at pos 0

        let f1 = path.future_accesses_for(0, 1).unwrap();
        assert!(!f1.contains_key(&10));
        assert!(f1.contains_key(&20));
        assert!(path.future_accesses_for(0, 2).unwrap().is_empty());
        assert!(path.future_accesses_for(0, 3).is_none());
    }

    /// Test that a sleeping thread stays asleep when its REMAINING work is independent,
    /// even if its PAST work would have conflicted.
    ///
    /// This is the key benefit of Fix 3: position-sensitive cache prevents
    /// false wakeups from past accesses that are no longer relevant.
    #[test]
    fn test_sleeping_thread_stays_asleep_when_remaining_work_independent() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // First execution: T0 at pos 0 (writes obj 100), T0 at pos 1 (writes obj 200),
        //                  T1 at pos 2 (reads obj 300)
        // T0's step 0 accesses obj 100, step 1 accesses obj 200.
        // T1 accesses obj 300.
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 0, 2); // pos 1: T0
        path.record_access(1, 200, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2); // pos 2: T1
        path.record_access(2, 300, AccessKind::Read, AccessOrigin::PythonMemory);

        // Backtrack T1 at pos 0
        path.insert_wakeup(0, 1, None);
        path.step(); // Builds T0's per-step cache and conservative access union.

        // Second execution: T1 at pos 0, then T0 continues
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        // T1 writes obj 100 at pos 0 — this conflicts with T0's step 0 (obj 100)!
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);

        // New branch at pos 1: T0 chosen
        path.schedule(&[0], 0, 2);

        // Key check: T0 is sleeping at pos 0, propagated to pos 1.
        // T0's FUTURE from step 0 = {100, 200} (it hasn't run any steps yet).
        // T1's access at pos 0 = {100: Write}.
        // 100 Write vs 100 Write → CONFLICT → T0 should wake up.
        // This is correct because T0's future still includes obj 100.
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 should wake up because its future includes obj 100 which conflicts with T1's write"
        );
    }

    /// Test that a sleeping thread is woken when its REMAINING work conflicts.
    #[test]
    fn test_sleeping_thread_wakes_when_remaining_work_conflicts() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // First execution: T0 writes obj 100 at step 0, T1 writes obj 100 at step 0
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2); // pos 1: T1
        path.record_access(1, 100, AccessKind::Write, AccessOrigin::PythonMemory);

        path.insert_wakeup(0, 1, None);
        path.step(); // Cache: T0 has one exact step and a conservative union.

        // Second execution: T1 first
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory); // T1 writes obj 100

        path.schedule(&[0], 0, 2);

        // T0's future from step 0 = {100: Write}, T1's access = {100: Write}
        // Write vs Write → CONFLICT → T0 wakes up
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 must wake up when remaining work conflicts"
        );
    }

    /// Test position-sensitive cache with interleaved threads.
    ///
    /// Verifies that the cache correctly tracks per-thread step counts
    /// when multiple threads interleave their scheduling points.
    #[test]
    fn test_position_sensitive_interleaved_threads() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // First execution:
        //   pos 0: T0 writes obj 100 (T0 step 0)
        //   pos 1: T0 writes obj 200 (T0 step 1)
        //   pos 2: T1 reads obj 300  (T1 step 0)
        //   pos 3: T2 writes obj 100 (T2 step 0)
        path.schedule(&[0, 1, 2], 0, 3); // pos 0: T0
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1, 2], 0, 3); // pos 1: T0
        path.record_access(1, 200, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1, 2], 1, 3); // pos 2: T1
        path.record_access(2, 300, AccessKind::Read, AccessOrigin::PythonMemory);
        path.schedule(&[2], 2, 3); // pos 3: T2
        path.record_access(3, 100, AccessKind::Write, AccessOrigin::PythonMemory);

        // Add T2 to wakeup at pos 0
        path.insert_wakeup(0, 2, None);

        path.step();
        // Cache: T0 has two exact steps and one conservative union.

        // Second execution: T2 at pos 0
        let chosen = path.schedule(&[0, 1, 2], 0, 3);
        assert_eq!(chosen, Some(2));
        // T2 writes obj 200 at pos 0
        path.record_access(0, 200, AccessKind::Write, AccessOrigin::PythonMemory);

        // New pos 1
        path.schedule(&[0, 1], 0, 3);

        // T0 sleeping at pos 0, propagated to pos 1.
        // T0's future from step 0 = {100, 200}.
        // T2's access at pos 0 = {200: Write}.
        // Write vs Write on obj 200 → CONFLICT → T0 wakes up.
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 wakes because future[0] includes obj 200 which conflicts with T2"
        );
    }

    /// A still-runnable thread that outgrows its cached trace has an unknown future.
    #[test]
    fn test_future_accesses_for_extra_steps_returns_unknown() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // First execution: T0 runs once, T1 runs once
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 10, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2); // pos 1: T1
        path.record_access(1, 20, AccessKind::Read, AccessOrigin::PythonMemory);

        path.insert_wakeup(0, 1, None);
        path.step(); // Cache: T0 futures = [{10: Write}, {}]

        // Second execution: T1 first, then T0 runs multiple steps
        path.schedule(&[0, 1], 1, 2); // pos 0: T1 (current=1)
        path.schedule(&[0], 0, 2);    // pos 1: T0 (step 0 of T0)
        path.schedule(&[0], 0, 2);    // pos 2: T0 (step 1 of T0)

        // At pos=3, T0 has taken 2 steps in new execution, but cache only has
        // 1 step for T0 (futures has len 2: [0]={10:Write}, [1]=empty).
        // k=2 >= the one cached step, so its future is unknown.  Treating it
        // as empty could keep the thread asleep and prune a new conflict.
        let result = path.future_accesses_for(0, 2);
        assert_eq!(result, None, "Extra steps must wake conservatively");
        assert_eq!(
            path.future_steps_independent_of(0, 2, &HashMap::new()),
            None,
            "An exhausted per-step cache must not certify independence"
        );
    }

    /// Test that a lazily-derived suffix union merges access kinds correctly.
    ///
    /// When the same object is accessed with Read at step 0 and Write at step 1,
    /// The union should have the object as Write (merged from Read + Write).
    #[test]
    fn test_step_future_merges_access_kinds() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // T0 step 0: reads obj 10
        // T0 step 1: writes obj 10
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 10, AccessKind::Read, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 0, 2); // pos 1: T0
        path.record_access(1, 10, AccessKind::Write, AccessOrigin::PythonMemory);

        path.step();

        let accesses = path.future_accesses_for(0, 0).unwrap();

        assert_eq!(accesses.get(&10).unwrap().0, AccessKind::Write);
    }

    /// Test that WeakWrite + WeakRead merges to WeakWrite in the future cache,
    /// not Write. This prevents spurious sleep-set wakeups: another thread
    /// doing WeakWrite on the same object should stay asleep (WeakWrite +
    /// WeakWrite is independent per access_kinds_conflict).
    #[test]
    fn test_future_cache_weak_write_weak_read_no_upgrade() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // T0 does WeakWrite(obj 100) at step 0, WeakRead(obj 100) at step 1.
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 100, AccessKind::WeakWrite, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 0, 2); // pos 1: T0
        path.record_access(1, 100, AccessKind::WeakRead, AccessOrigin::PythonMemory);
        path.step();

        let accesses = path.future_accesses_for(0, 0).unwrap();
        // Key: merged kind should be WeakWrite (not Write), since WeakWrite
        // subsumes WeakRead's conflict set.
        assert_eq!(
            accesses.get(&100).unwrap().0,
            AccessKind::WeakWrite,
            "WeakWrite + WeakRead should merge to WeakWrite, not Write"
        );
    }

    /// Test that per-step independence checking avoids false wakeup from
    /// WeakWrite + Read merge escalation to Write.
    ///
    /// T0: WeakWrite(obj 100) at step 0, Read(obj 100) at step 1
    /// T1: WeakRead(obj 100)
    ///
    /// Merged suffix union: merge(WeakWrite, Read) = Write (catch-all in AccessKind::merge).
    ///   Write vs WeakRead → CONFLICT → T0 wakes up (FALSE wakeup).
    ///
    /// Per-step check:
    ///   step 0: WeakWrite vs WeakRead → independent (not in conflict rules)
    ///   step 1: Read vs WeakRead → independent (not in conflict rules)
    ///   → T0 should STAY ASLEEP (correct)
    #[test]
    fn test_per_step_independence_avoids_false_wakeup() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        // First execution: T0 runs 2 steps, then T1 runs 1 step
        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 100, AccessKind::WeakWrite, AccessOrigin::PythonMemory);
        path.schedule(&[0, 1], 0, 2); // pos 1: T0
        path.record_access(1, 100, AccessKind::Read, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2); // pos 2: T1
        path.record_access(2, 100, AccessKind::WeakRead, AccessOrigin::PythonMemory);

        // Insert T1 at pos 0 to force a second execution with T1 first
        path.insert_wakeup(0, 1, None);
        path.step(); // Builds cache

        // Second execution: T1 at pos 0
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 100, AccessKind::WeakRead, AccessOrigin::PythonMemory);

        // New branch at pos 1
        path.schedule(&[0], 0, 2);

        // T0 is sleeping at pos 0, propagated to pos 1.
        // T0's future steps: step 0 = {100: WeakWrite}, step 1 = {100: Read}
        // T1's access at pos 0 = {100: WeakRead}
        //
        // Per-step check:
        //   step 0: WeakWrite vs WeakRead → independent (not in conflict rules)
        //   step 1: Read vs WeakRead → independent (not in conflict rules)
        //   → T0 should STAY ASLEEP
        //
        // Merged check (current):
        //   merged future = {100: Write} (merge(WeakWrite, Read) = Write)
        //   Write vs WeakRead → CONFLICT
        //   → T0 wakes up (FALSE wakeup)
        assert!(
            path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 should stay asleep: per-step independence check should avoid false wakeup \
             from WeakWrite+Read merge escalation to Write"
        );
    }

    /// Regression guard: per-step independence must still detect REAL conflicts.
    /// T0: Write(obj 100) at step 0
    /// T1: Read(obj 100)
    /// Write vs Read → conflict → T0 must wake up.
    #[test]
    fn test_per_step_keeps_real_conflicts() {
        use crate::access::{AccessKind, AccessOrigin};
        let mut path = Path::new(None, SearchStrategy::Dfs);

        path.schedule(&[0, 1], 0, 2); // pos 0: T0
        path.record_access(0, 100, AccessKind::Write, AccessOrigin::PythonMemory);
        path.schedule(&[1], 1, 2); // pos 1: T1
        path.record_access(1, 100, AccessKind::Read, AccessOrigin::PythonMemory);

        path.insert_wakeup(0, 1, None);
        path.step();

        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 100, AccessKind::Read, AccessOrigin::PythonMemory);

        path.schedule(&[0], 0, 2);

        // T0's step 0: Write(100). T1's access: Read(100).
        // Write vs Read → CONFLICT → T0 must wake up.
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 must wake up: Write vs Read is a real conflict"
        );
    }
}
