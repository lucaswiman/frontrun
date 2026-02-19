# DPOR Specification for Frontrun

## 1. Motivation and Background

### The Coverage Gap

Frontrun's `explore_interleavings()` currently generates random schedules — a random
walk through the interleaving space. For a program with `T` threads and `N` shared
operations per thread, the number of distinct interleavings grows as
`(T*N)! / (N!)^T`. Even modest programs (3 threads, 10 operations each) have billions
of interleavings. Random sampling provides no feedback about coverage and may
redundantly explore equivalent orderings while missing the one interleaving that
triggers a bug.

**Dynamic Partial Order Reduction (DPOR)** is the principled solution. It systematically
explores only *distinct* interleavings — those that differ in the ordering of
**dependent** (conflicting) operations. Two interleavings that differ only in the
ordering of independent operations (e.g., two threads writing to different variables)
are equivalent under Mazurkiewicz trace theory, and only one representative needs to
be tested.

### Why a Rust Core

The DPOR engine is the performance-critical inner loop: it must track vector clocks,
compute backtrack sets, and manage an exploration tree across potentially millions of
execution replays. Writing this core in Rust offers several advantages:

1. **Direct translation from loom.** Tokio's [loom](https://github.com/tokio-rs/loom)
   library has a proven, battle-tested DPOR implementation in Rust. Writing
   frontrun's DPOR core in Rust lets us translate loom's data structures and
   algorithms with minimal impedance mismatch, reusing the design decisions that
   loom's maintainers have already validated (CDSChecker-style exploration, bounded
   DPOR with preemption limits, vector clock dependency tracking).

2. **Performance.** Vector clock operations (`join`, `<=`, indexing) and exploration
   tree traversal are tight loops over small arrays. Rust eliminates Python's
   per-operation overhead, making exhaustive exploration of programs with tens of
   thousands of interleavings practical.

3. **Memory efficiency.** The exploration tree can grow large. Rust's predictable
   memory layout and lack of per-object overhead (no dict, no GC headers) keeps the
   working set small.

4. **PyO3 integration.** The Rust core exposes a clean Python API via
   [PyO3](https://pyo3.rs/), keeping the user-facing interface in Python while the
   engine runs at native speed.

5. **Future portability.** A Rust DPOR core could be reused by Rust-native
   concurrency testing tools, or embedded into other language runtimes via FFI.

### Relationship to TLA+ Integration

The TLA+ integration described in `ideas/tla_plus_integration.md` offers an
alternative path to exhaustive coverage: outsource state-space exploration to TLC and
replay counterexamples in Python. **DPOR and TLA+ are complementary, not competing
approaches:**

| Dimension | DPOR | TLA+/TLC |
|-----------|------|----------|
| What it explores | The *implementation's* concrete state space | The *model's* abstract state space |
| Granularity | Bytecode/opcode level (fine-grained) | Action/label level (coarse) |
| Fidelity | Tests real Python code directly | Tests an abstraction; real code via replay |
| Setup cost | Zero (no spec needed) | Agent generates and maintains a spec |
| Bug class | Finds bugs in the implementation | Finds bugs in the design |
| Coverage guarantee | All distinct interleavings at the concrete level | All distinct behaviors of the abstract model |

DPOR is the right tool when you want to verify that a **specific piece of Python
code** is correct under all interleavings, without writing or maintaining a formal
model. TLA+ is the right tool when you want to verify that the **algorithm design**
is correct before or alongside implementation. The two can be combined: TLA+ finds
interesting abstract schedules, DPOR exhaustively verifies the concrete implementation
around those schedules.

---

## 2. Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Python User Code                      │
│  explore_interleavings(..., strategy="dpor")             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Python Orchestration Layer                   │
│  frontrun/dpor.py                                       │
│  - Wraps Rust engine in Python API                       │
│  - Drives execution: run program, collect events,        │
│    feed to engine, get next schedule                     │
│  - Integrates with existing TraceExecutor / bytecode     │
│    infrastructure                                        │
└──────────────────────┬──────────────────────────────────┘
                       │ PyO3 FFI boundary
┌──────────────────────▼──────────────────────────────────┐
│              Rust DPOR Engine                             │
│  frontrun-dpor/src/                                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Exploration  │  │ Execution    │  │ Vector Clocks  │  │
│  │ Tree (Path)  │  │ State        │  │ (VersionVec)   │  │
│  │              │  │              │  │                │  │
│  │ - Branch     │  │ - Thread     │  │ - join()       │  │
│  │   entries    │  │   states     │  │ - partial_le() │  │
│  │ - Backtrack  │  │ - Object     │  │ - increment()  │  │
│  │ - step()     │  │   tracking   │  │                │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────┘  │
│         │                 │                    │          │
│  ┌──────▼─────────────────▼────────────────────▼───────┐  │
│  │              DPOR Algorithm Core                     │  │
│  │  - Dependency detection (Access tracking)            │  │
│  │  - Happens-before computation                        │  │
│  │  - Backtrack set insertion                           │  │
│  │  - Schedule generation                               │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Directory Layout

```
frontrun/
├── frontrun/
│   ├── dpor.py                  # Python API and orchestration
│   └── ...                      # Existing modules unchanged
├── frontrun-dpor/              # Rust crate (PyO3 extension)
│   ├── Cargo.toml
│   ├── pyproject.toml           # maturin build config
│   └── src/
│       ├── lib.rs               # PyO3 module definition
│       ├── engine.rs            # Top-level DporEngine struct
│       ├── path.rs              # Exploration tree (branches, backtracking)
│       ├── execution.rs         # Per-run execution state
│       ├── thread.rs            # Thread state machine
│       ├── vv.rs                # VersionVec (vector clocks)
│       ├── access.rs            # DPOR access tracking
│       ├── object.rs            # Shared object registry
│       └── schedule.rs          # Schedule generation from path
└── tests/
    ├── test_dpor.py             # Python integration tests
    └── ...
```

---

## 3. Core Data Structures (Rust)

These data structures are modeled directly on loom's internal representations,
adapted for frontrun's Python-centric use case. Where loom models C11 atomic
operations with modification orders and reads-from relations, frontrun models
Python shared-memory accesses (object attribute reads/writes, dict operations, list
mutations) with a simpler dependency relation.

### 3.1 VersionVec (Vector Clock)

The fundamental building block for happens-before tracking.

```rust
/// A vector clock indexed by thread ID.
///
/// Tracks causal ordering between events. If `a.vv <= b.vv`, then event `a`
/// happens-before event `b`. If neither dominates, the events are concurrent
/// (and potentially racing).
///
/// Corresponds to loom's `rt::vv::VersionVec`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionVec {
    /// Clock values indexed by thread ID. Thread IDs are dense integers 0..N.
    clocks: Vec<u32>,
}

impl VersionVec {
    /// Create a zero vector for `num_threads` threads.
    pub fn new(num_threads: usize) -> Self {
        Self { clocks: vec![0; num_threads] }
    }

    /// Increment the clock for `thread_id` (tick on local event).
    pub fn increment(&mut self, thread_id: usize) {
        self.clocks[thread_id] += 1;
    }

    /// Point-wise maximum: self = max(self, other).
    /// Used for acquire synchronization and thread join.
    pub fn join(&mut self, other: &VersionVec) {
        for (a, b) in self.clocks.iter_mut().zip(other.clocks.iter()) {
            *a = (*a).max(*b);
        }
    }

    /// Returns true if `self` happens-before-or-equal `other`.
    /// i.e., self[i] <= other[i] for all i.
    pub fn partial_le(&self, other: &VersionVec) -> bool {
        self.clocks.iter().zip(other.clocks.iter()).all(|(a, b)| a <= b)
    }

    /// Returns true if `self` and `other` are concurrent
    /// (neither happens-before the other).
    pub fn concurrent_with(&self, other: &VersionVec) -> bool {
        !self.partial_le(other) && !other.partial_le(self)
    }
}
```

### 3.2 Thread State

```rust
/// State of a single modeled thread within one execution.
///
/// Tracks the thread's causal knowledge (via vector clocks) and its status
/// in the exploration tree.
///
/// Corresponds to loom's `rt::thread::Thread` (the internal representation).
pub struct Thread {
    /// Thread identifier (dense integer, 0-indexed).
    pub id: usize,

    /// Happens-before vector clock. Updated on synchronization events
    /// (lock acquire/release, thread join, etc.). Tracks which events
    /// this thread has "observed."
    pub causality: VersionVec,

    /// DPOR-specific vector clock. Updated on scheduling decisions.
    /// Used to determine whether two accesses are causally ordered
    /// for the purpose of backtrack set computation.
    ///
    /// Distinction from `causality`: `causality` tracks the program's
    /// happens-before relation (semantic synchronization). `dpor_vv` tracks
    /// the explorer's scheduling decisions. An access at dpor_vv position X
    /// by thread T means "T was last scheduled at branch position X."
    pub dpor_vv: VersionVec,

    /// Current status in the exploration.
    pub status: ThreadStatus,
}

/// Scheduling status for a thread at a branch point.
///
/// Corresponds to loom's thread scheduling states within `Schedule` branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadStatus {
    /// Thread has not been created yet or has finished.
    Disabled,

    /// Thread is runnable and has not been explored at this branch.
    /// DPOR has not identified a reason to explore it here.
    Pending,

    /// Thread is runnable and has been marked for exploration by DPOR
    /// backtracking. When the explorer backtracks to this branch, this
    /// thread will be tried as the active thread.
    Backtrack,

    /// Thread yielded (e.g., called `yield_now()`). Available for
    /// exploration but lower priority than Backtrack.
    Yield,

    /// Thread is currently executing at this branch point.
    Active,

    /// Thread has already been explored from this branch point.
    Visited,

    /// Thread should be skipped (blocked on a lock, waiting on a
    /// condition, etc.).
    Blocked,
}
```

### 3.3 Access (DPOR Dependency Tracking)

```rust
/// Records the last access to a shared object for DPOR dependency detection.
///
/// When thread T accesses object O, we record T's current position in the
/// exploration tree (`path_id`) and T's current DPOR vector clock (`dpor_vv`).
/// When a later thread T' accesses the same object, we compare the recorded
/// `dpor_vv` against T's current DPOR vector clock to determine if the two
/// accesses are causally ordered or concurrent.
///
/// Corresponds to loom's `rt::access::Access`.
#[derive(Clone, Debug)]
pub struct Access {
    /// Position in the exploration tree (index into Path::branches) where
    /// this access occurred. Used to locate the branch point for backtracking.
    pub path_id: usize,

    /// DPOR vector clock of the accessing thread at the time of access.
    /// Used to check whether a later access is causally ordered with this one.
    pub dpor_vv: VersionVec,

    /// The thread that performed this access.
    pub thread_id: usize,
}

impl Access {
    /// Returns true if this access happens-before the given vector clock.
    /// If true, the accesses are ordered and no backtracking is needed.
    /// If false, the accesses are concurrent and DPOR must add a backtrack point.
    pub fn happens_before(&self, later_vv: &VersionVec) -> bool {
        self.dpor_vv.partial_le(later_vv)
    }
}
```

### 3.4 Shared Object Tracking

```rust
/// Identifies a shared memory location in the Python program.
///
/// Unlike loom (which tracks atomic cells, mutexes, etc. as distinct object types),
/// frontrun must track Python-level shared state: object attributes, dict keys,
/// list indices. The `ObjectId` captures the identity of the shared location.
///
/// Object identity is determined by the Python orchestration layer and passed
/// to the Rust engine as an opaque identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ObjectId {
    /// An attribute access: (id(obj), attr_name).
    /// e.g., `account.balance` -> Attr(id(account), "balance")
    Attr { object_id: u64, attr: String },

    /// A dictionary key access: (id(dict), key_repr).
    /// e.g., `d["x"]` -> DictKey(id(d), "x")
    DictKey { dict_id: u64, key: String },

    /// A list index access: (id(list), index).
    /// e.g., `lst[0]` -> ListIndex(id(lst), 0)
    ListIndex { list_id: u64, index: i64 },

    /// A global variable access: (module_name, var_name).
    Global { module: String, name: String },

    /// A named synchronization primitive (lock, semaphore, etc.).
    /// Used for happens-before tracking rather than conflict detection.
    SyncPrimitive { name: String },

    /// Opaque integer ID assigned by the Python layer. Useful when the
    /// Python orchestrator has its own object numbering scheme.
    Opaque(u64),
}

/// The kind of access to a shared object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    /// A read-only access (e.g., `x = obj.attr`).
    Read,

    /// A write access (e.g., `obj.attr = x`).
    Write,
}

/// Tracks the last accesses to a shared object for DPOR.
///
/// Determines which pairs of accesses are "dependent" (conflicting):
/// - Read/Read: independent (two reads to the same location don't conflict)
/// - Read/Write: dependent
/// - Write/Read: dependent
/// - Write/Write: dependent
///
/// This mirrors loom's `last_access` / `last_non_load_access` tracking
/// in `rt::atomic::State`.
pub struct ObjectState {
    /// Last access of any kind (read or write).
    pub last_access: Option<Access>,

    /// Last write access. Read-only accesses don't update this.
    /// Used for the dependency check: a Read depends only on the last Write,
    /// not on other Reads.
    pub last_write_access: Option<Access>,
}

impl ObjectState {
    pub fn new() -> Self {
        Self { last_access: None, last_write_access: None }
    }

    /// Returns the last dependent access for the given access kind.
    ///
    /// - A Read depends on the last Write (reads are independent of each other).
    /// - A Write depends on any prior access (read or write).
    ///
    /// This is the key DPOR dependency relation. It determines whether
    /// backtracking is needed.
    pub fn last_dependent_access(&self, kind: AccessKind) -> Option<&Access> {
        match kind {
            AccessKind::Read => self.last_write_access.as_ref(),
            AccessKind::Write => self.last_access.as_ref(),
        }
    }

    /// Record a new access, updating tracking state.
    pub fn record_access(&mut self, access: Access, kind: AccessKind) {
        if kind == AccessKind::Write {
            self.last_write_access = Some(access.clone());
        }
        self.last_access = Some(access);
    }
}
```

### 3.5 Exploration Tree (Path)

The exploration tree is the central data structure driving exhaustive search. Each
node represents a non-deterministic choice point (which thread to schedule next).
DPOR prunes branches by only adding backtrack points where thread reorderings at
**dependent** operations would produce observably different results.

```rust
/// A single branch point in the exploration tree.
///
/// At each branch, the scheduler chose one thread to run. The `threads` array
/// records which threads were available and their exploration status.
///
/// Corresponds to loom's `Schedule` variant of `path::Branch`.
#[derive(Clone, Debug)]
pub struct Branch {
    /// Per-thread scheduling status at this branch point.
    /// Index is thread ID; value is the thread's status.
    pub threads: Vec<ThreadStatus>,

    /// Which thread was (or will be) selected to run at this branch.
    pub active_thread: usize,

    /// Number of preemptions so far on the path leading to this branch.
    /// A preemption occurs when the scheduler switches away from a thread
    /// that could have continued running. Used for preemption bounding.
    pub preemptions: u32,
}

/// The exploration tree: a sequence of branches representing one execution path,
/// with backtrack information for DFS exploration of alternatives.
///
/// During an execution, branches are appended as scheduling decisions are made.
/// After the execution completes, `step()` backtracks through the branches to
/// find the next unexplored alternative.
///
/// Corresponds to loom's `rt::path::Path`.
pub struct Path {
    /// The sequence of branch points for the current (or replaying) execution.
    branches: Vec<Branch>,

    /// Current position during replay. During a new execution, branches up to
    /// `replay_pos` are replayed (using the recorded active_thread), and new
    /// branches are appended after that.
    replay_pos: usize,
}

impl Path {
    pub fn new() -> Self {
        Self { branches: Vec::new(), replay_pos: 0 }
    }

    /// Called when the scheduler needs to pick a thread at a new branch point.
    ///
    /// If we're replaying (pos < replay_pos), return the previously recorded choice.
    /// Otherwise, pick an enabled thread and record a new branch.
    pub fn schedule(&mut self, runnable: &[usize], current_thread: usize) -> usize {
        let pos = self.branches.len();
        if pos < self.replay_pos {
            // Replaying: follow the recorded path
            return self.branches[pos].active_thread;
        }

        // New branch: pick the first runnable thread (or the current one if runnable)
        let chosen = if runnable.contains(&current_thread) {
            current_thread
        } else {
            runnable[0]
        };

        let mut threads = vec![ThreadStatus::Disabled; /* num_threads */];
        for &tid in runnable {
            threads[tid] = if tid == chosen {
                ThreadStatus::Active
            } else {
                ThreadStatus::Pending
            };
        }

        self.branches.push(Branch {
            threads,
            active_thread: chosen,
            preemptions: 0, // computed by caller
        });

        chosen
    }

    /// Mark `thread_id` for backtracking at branch `path_id`.
    ///
    /// This is the core DPOR operation: when we detect that the current thread's
    /// access conflicts with a previous access at `path_id`, we mark the current
    /// thread as needing exploration at that earlier branch point.
    ///
    /// Corresponds to loom's `Path::backtrack()`.
    pub fn backtrack(&mut self, path_id: usize, thread_id: usize, preemption_bound: Option<u32>) {
        if path_id >= self.branches.len() {
            return;
        }

        let branch = &mut self.branches[path_id];

        // Only mark for backtracking if the thread is in a state that allows it
        match branch.threads[thread_id] {
            ThreadStatus::Pending | ThreadStatus::Yield => {
                // Check preemption bound
                if let Some(bound) = preemption_bound {
                    if branch.preemptions >= bound {
                        // Would exceed preemption bound. Conservative: add a
                        // fallback backtrack point at an earlier position where
                        // the preemption count is lower.
                        //
                        // This is loom's approach: when bounded DPOR would skip
                        // a backtrack point due to the bound, it adds an extra
                        // point earlier to preserve soundness.
                        self.add_conservative_backtrack(path_id, thread_id, bound);
                        return;
                    }
                }
                branch.threads[thread_id] = ThreadStatus::Backtrack;
            }
            // Already active, visited, or backtrack-marked: no change needed
            _ => {}
        }
    }

    /// Advance to the next unexplored execution path.
    ///
    /// Walks backward through branches, looking for a branch with a thread
    /// marked `Backtrack` that hasn't been tried yet. When found, truncates
    /// the path to that point, sets the backtrack thread as active, and
    /// returns `true`. Returns `false` when all paths are exhausted.
    ///
    /// Corresponds to loom's `Path::step()` / `Execution::step()`.
    pub fn step(&mut self) -> bool {
        while let Some(branch) = self.branches.last_mut() {
            // Mark the current active thread as visited
            let active = branch.active_thread;
            if branch.threads[active] == ThreadStatus::Active {
                branch.threads[active] = ThreadStatus::Visited;
            }

            // Look for a thread marked for backtracking
            if let Some(next) = branch.threads.iter().position(|s| *s == ThreadStatus::Backtrack) {
                branch.threads[next] = ThreadStatus::Active;
                branch.active_thread = next;
                self.replay_pos = self.branches.len();
                return true;
            }

            // No more alternatives at this branch: pop and continue backtracking
            self.branches.pop();
        }

        false // All paths exhausted
    }

    fn add_conservative_backtrack(&mut self, _path_id: usize, _thread_id: usize, _bound: u32) {
        // Walk backward from path_id to find a branch where preemptions < bound
        // and the thread is available. This ensures bounded DPOR doesn't miss
        // interleavings that could be reached within the bound via a different
        // path through the tree.
        //
        // See loom's bounded DPOR implementation for details.
        todo!("Conservative backtrack for preemption bounding")
    }
}
```

---

## 4. The DPOR Algorithm

### 4.1 Overview

The algorithm performs depth-first exploration of the interleaving space, using
vector clocks to detect dependencies and prune equivalent interleavings.

```
PROCEDURE ExploreExhaustively(program):
    path = new Path()
    loop:
        // Phase 1: Execute the program, collecting access events
        execution = new Execution(num_threads)
        events = RunProgram(program, path, execution)

        // Phase 2: Process events for DPOR
        for event in events:
            ProcessEvent(execution, path, event)

        // Phase 3: Check properties (invariants, assertions)
        CheckProperties(execution)

        // Phase 4: Backtrack to next unexplored path
        if not path.step():
            break  // All paths exhausted

    return AllPropertiesHeld
```

### 4.2 Event Processing (The Core DPOR Logic)

When a thread accesses a shared object, the DPOR algorithm:

1. Looks up the **last dependent access** to that object.
2. Checks whether the last access **happens-before** the current one.
3. If **not** (the accesses are concurrent), inserts a **backtrack point** at the
   earlier branch, marking the current thread for exploration there.
4. Updates the object's access tracking.

```rust
/// The main DPOR engine. Holds the exploration tree and per-execution state.
///
/// The Python orchestration layer creates one DporEngine and calls it
/// repeatedly until exploration is complete.
pub struct DporEngine {
    /// The exploration tree, persisted across executions.
    path: Path,

    /// Number of threads in the program under test.
    num_threads: usize,

    /// Optional preemption bound. None = unbounded (full DPOR).
    /// Some(n) = explore only interleavings with at most n preemptions.
    preemption_bound: Option<u32>,

    /// Maximum number of branches per execution (safety limit).
    max_branches: usize,

    /// Maximum number of executions (safety limit).
    max_executions: Option<u64>,

    /// Counter: total executions completed so far.
    executions_completed: u64,
}

/// Per-execution state. Reset at the start of each execution.
pub struct Execution {
    /// Per-thread state (vector clocks, status).
    threads: Vec<Thread>,

    /// Per-object access tracking.
    objects: HashMap<ObjectId, ObjectState>,

    /// The currently active thread.
    active_thread: usize,

    /// Whether this execution has been aborted (e.g., deadlock detected).
    aborted: bool,
}

impl DporEngine {
    /// Process a shared memory access event from the Python layer.
    ///
    /// This is the core DPOR operation. Called by the Python orchestration
    /// layer each time a thread accesses shared state.
    ///
    /// Arguments:
    /// - `execution`: the current execution state
    /// - `thread_id`: the thread performing the access
    /// - `object_id`: which shared object is being accessed
    /// - `kind`: Read or Write
    ///
    /// Corresponds to the DPOR logic in loom's `Execution::schedule()` combined
    /// with access tracking in `rt::atomic` and `rt::object`.
    pub fn process_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: &ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position();
        let current_dpor_vv = execution.threads[thread_id].dpor_vv.clone();

        // Look up the last dependent access to this object
        let object_state = execution.objects
            .entry(object_id.clone())
            .or_insert_with(ObjectState::new);

        if let Some(prev_access) = object_state.last_dependent_access(kind) {
            // Check if the previous access happens-before the current one
            if !prev_access.happens_before(&current_dpor_vv) {
                // Concurrent dependent accesses: insert backtrack point
                //
                // Mark the current thread for exploration at the branch where
                // the previous access occurred. This ensures we will explore
                // the alternative ordering where the current thread runs before
                // the previous accessor at that point.
                self.path.backtrack(
                    prev_access.path_id,
                    thread_id,
                    self.preemption_bound,
                );
            }
        }

        // Record this access for future dependency checks
        let access = Access {
            path_id: current_path_id,
            dpor_vv: current_dpor_vv,
            thread_id,
        };
        object_state.record_access(access, kind);
    }

    /// Process a synchronization event (lock acquire, release, thread join, etc.).
    ///
    /// Synchronization events update the happens-before relation (causality
    /// vector clocks) but don't directly trigger DPOR backtracking. They
    /// indirectly affect DPOR by establishing ordering between accesses,
    /// which may cause future access pairs to be classified as "ordered"
    /// rather than "concurrent," reducing the backtrack set.
    pub fn process_sync(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        event: SyncEvent,
    ) {
        match event {
            SyncEvent::LockAcquire { lock_id, release_vv } => {
                // Acquire semantics: join the lock's release vector clock
                // into the acquiring thread's causality.
                execution.threads[thread_id].causality.join(&release_vv);
            }
            SyncEvent::LockRelease { lock_id } => {
                // Release semantics: the lock now carries this thread's
                // causality, to be joined by the next acquirer.
                // (Stored in a separate sync_state map, not shown here.)
            }
            SyncEvent::ThreadJoin { joined_thread } => {
                // The joining thread observes all events of the joined thread.
                let joined_vv = execution.threads[joined_thread].causality.clone();
                execution.threads[thread_id].causality.join(&joined_vv);
                execution.threads[thread_id].dpor_vv.join(
                    &execution.threads[joined_thread].dpor_vv
                );
            }
            SyncEvent::ThreadSpawn { child_thread } => {
                // The child inherits the parent's causal knowledge.
                let parent_causality = execution.threads[thread_id].causality.clone();
                let parent_dpor_vv = execution.threads[thread_id].dpor_vv.clone();
                execution.threads[child_thread].causality.join(&parent_causality);
                execution.threads[child_thread].dpor_vv.join(&parent_dpor_vv);
            }
        }
    }

    /// Advance to the next execution. Returns false when all paths exhausted.
    pub fn next_execution(&mut self) -> bool {
        self.executions_completed += 1;
        if let Some(max) = self.max_executions {
            if self.executions_completed >= max {
                return false;
            }
        }
        self.path.step()
    }
}

/// Synchronization events that affect the happens-before relation.
pub enum SyncEvent {
    LockAcquire { lock_id: ObjectId, release_vv: VersionVec },
    LockRelease { lock_id: ObjectId },
    ThreadJoin { joined_thread: usize },
    ThreadSpawn { child_thread: usize },
    CondWait { cond_id: ObjectId },
    CondNotify { cond_id: ObjectId },
}
```

### 4.3 Scheduling Decisions

At each scheduling point (where multiple threads are runnable), the engine consults
the exploration tree:

```rust
impl DporEngine {
    /// Pick which thread to run at the current scheduling point.
    ///
    /// During replay (re-executing a prefix), follows the recorded path.
    /// At new branch points, picks an enabled thread and records the choice.
    ///
    /// Returns the thread ID to run next, or None if deadlock.
    pub fn schedule(
        &mut self,
        execution: &mut Execution,
        runnable: &[usize],
    ) -> Option<usize> {
        if runnable.is_empty() {
            // Deadlock: no runnable threads
            execution.aborted = true;
            return None;
        }

        let chosen = self.path.schedule(runnable, execution.active_thread);

        // Update DPOR vector clocks for the scheduling decision
        let path_pos = self.path.current_position();
        execution.threads[chosen].dpor_vv.increment(chosen);

        // Track preemptions
        if execution.active_thread != chosen
            && runnable.contains(&execution.active_thread)
        {
            // This is a preemption: we switched away from a thread that
            // could have continued
            // (preemption count tracked in the Branch)
        }

        execution.active_thread = chosen;
        Some(chosen)
    }
}
```

### 4.4 The Exploration Loop (Python Side)

The Python orchestration layer drives the Rust engine:

```python
# frontrun/dpor.py

from frontrun_dpor import DporEngine, AccessKind, ObjectId
from typing import Callable, Any, Optional
from dataclasses import dataclass

@dataclass
class DporResult:
    """Result of exhaustive DPOR exploration."""
    executions_explored: int
    property_holds: bool
    counterexample: Optional[Any]  # State from the failing execution
    counterexample_schedule: Optional[list]  # Schedule that triggered failure

def explore_dpor(
    setup: Callable[[], Any],
    threads: list[Callable[[Any], None]],
    invariant: Callable[[Any], bool],
    max_executions: Optional[int] = None,
    preemption_bound: Optional[int] = None,
    max_branches: int = 100_000,
) -> DporResult:
    """
    Exhaustively explore interleavings using DPOR.

    This is the DPOR equivalent of `explore_interleavings()`. Instead of
    random sampling, it systematically explores all distinct interleavings
    (or a bounded subset if `preemption_bound` is set).

    Args:
        setup: Creates the shared state. Called once per execution.
        threads: List of callables, each taking the shared state.
        invariant: Predicate over the shared state. Must be true after
            all threads complete for the execution to pass.
        max_executions: Safety limit on total executions.
        preemption_bound: Limit on number of preemptions per execution.
            None = unbounded (full DPOR). 2-3 is typically sufficient.
        max_branches: Maximum branch points per execution.

    Returns:
        DporResult with exploration statistics and any counterexample found.
    """
    engine = DporEngine(
        num_threads=len(threads),
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        max_executions=max_executions,
    )

    while True:
        # Setup fresh state for this execution
        state = setup()

        # Run all threads under DPOR control
        schedule = run_under_dpor(engine, state, threads)

        # Check invariant
        if not invariant(state):
            return DporResult(
                executions_explored=engine.executions_completed,
                property_holds=False,
                counterexample=state,
                counterexample_schedule=schedule,
            )

        # Advance to next execution
        if not engine.next_execution():
            break

    return DporResult(
        executions_explored=engine.executions_completed,
        property_holds=True,
        counterexample=None,
        counterexample_schedule=None,
    )
```

---

## 5. Conflict Detection for Python

The hardest part of adapting DPOR from loom to frontrun is conflict detection.
Loom instruments explicit `Atomic*` types — the user opts into tracking by using
`loom::sync::AtomicUsize` instead of `std::sync::atomic::AtomicUsize`. Frontrun
must detect accesses to arbitrary Python shared state.

### 5.1 Approaches to Conflict Detection

#### Approach A: Opcode-Level Instrumentation (Bytecode Mode)

The existing bytecode instrumentation already intercepts every opcode via
`sys.settrace` with `f_trace_opcodes = True`. We can extend the trace callback to
identify shared-memory operations:

| Opcode | Shared Access? | ObjectId |
|--------|---------------|----------|
| `LOAD_ATTR` | Read of `obj.attr` | `Attr(id(obj), attr)` |
| `STORE_ATTR` | Write to `obj.attr` | `Attr(id(obj), attr)` |
| `BINARY_SUBSCR` | Read of `obj[key]` | `DictKey(id(obj), repr(key))` or `ListIndex(id(obj), key)` |
| `STORE_SUBSCR` | Write to `obj[key]` | `DictKey(id(obj), repr(key))` or `ListIndex(id(obj), key)` |
| `LOAD_GLOBAL` | Read of global var | `Global(module, name)` |
| `STORE_GLOBAL` | Write to global var | `Global(module, name)` |
| `DELETE_ATTR` | Write to `obj.attr` | `Attr(id(obj), attr)` |
| `DELETE_SUBSCR` | Write to `obj[key]` | `DictKey(id(obj), repr(key))` |

**Implementation:** The trace callback examines the opcode, peeks at the relevant
stack values (using `frame.f_locals`, `frame.f_code.co_names`, and/or
`ctypes`-based stack inspection), and reports the access to the DPOR engine.

**Advantages:** No code changes required. Works with unmodified third-party code.
Complete visibility into all shared-memory operations.

**Disadvantages:** High overhead (trace callback on every opcode + stack inspection).
Some accesses are to thread-local state and are falsely flagged as shared (leading to
unnecessary backtracking, not unsoundness). Requires careful filtering to exclude
accesses to the scheduler's own state, stdlib internals, etc.

#### Approach B: Marker-Based Conflict Annotation (Trace Marker Mode)

Extend the trace marker syntax to declare accessed objects:

```python
def transfer(self, amount):
    # frontrun: read_balance [self.balance:read]
    current = self.balance
    new_balance = current + amount
    # frontrun: write_balance [self.balance:write]
    self.balance = new_balance
```

The marker `[self.balance:read]` tells DPOR that this marker accesses `self.balance`
as a read. The Python layer evaluates `self.balance` in the frame's context to
produce an `ObjectId`.

**Advantages:** Low overhead (only evaluated at markers, not every opcode). Precise
(no false sharing). Clear intent.

**Disadvantages:** Requires manual annotation. May miss accesses between markers.

#### Approach C: Loom-Style Cooperative Primitives

The `frontrun.sync` module proposed in FUTURE_WORK.md would provide primitives that
self-report to the DPOR engine:

```python
from frontrun.sync import SharedVar

balance = SharedVar(100)

def transfer(amount):
    current = balance.read()    # Reports Read access to DPOR engine
    balance.write(current + amount)  # Reports Write access to DPOR engine
```

**Advantages:** Precise conflict tracking. Natural API. No runtime overhead outside
controlled execution.

**Disadvantages:** Requires code changes (import swaps). Can't test unmodified code.

### 5.2 Recommended Approach

Support all three, with the orchestration layer abstracting over the conflict source:

```python
class ConflictSource(Protocol):
    """Provides access events to the DPOR engine."""

    def get_events(self) -> Iterator[AccessEvent]:
        """Yield access events from the current execution step."""
        ...

class OpcodeConflictSource(ConflictSource):
    """Conflict detection via opcode instrumentation."""
    ...

class MarkerConflictSource(ConflictSource):
    """Conflict detection via annotated trace markers."""
    ...

class CooperativeConflictSource(ConflictSource):
    """Conflict detection via frontrun.sync primitives."""
    ...
```

The DPOR engine itself is agnostic to the conflict source — it receives
`(thread_id, object_id, access_kind)` tuples and applies the algorithm.

### 5.3 Filtering Heuristics

Not every Python memory access is a meaningful shared-state access. The conflict
detector should filter out:

1. **Thread-local accesses.** Accesses to local variables (never shared), function
   parameters, loop variables. The opcode tracer can detect these via
   `LOAD_FAST`/`STORE_FAST` (always local).

2. **Immutable objects.** Accesses to `int`, `str`, `bytes`, `tuple`, `frozenset`
   values. Reads of immutable objects are always independent.

3. **Scheduler internals.** Accesses to the frontrun scheduler's own state, the
   cooperative lock wrappers, etc. These must be invisible to DPOR.

4. **Stdlib internals.** Accesses within `threading`, `queue`, `collections`, etc.
   These are already synchronized or irrelevant.

The filtering is configurable: users can specify which modules/objects to include
or exclude from conflict tracking.

---

## 6. Synchronization-Aware Happens-Before

DPOR's effectiveness depends on accurately tracking the happens-before relation.
Two accesses to the same object that are ordered by happens-before don't need
backtracking — only **concurrent** conflicting accesses do.

### 6.1 Python Synchronization Primitives

Each primitive establishes happens-before edges:

| Primitive | Happens-Before Edge |
|-----------|-------------------|
| `Lock.release()` -> `Lock.acquire()` | Release happens-before subsequent acquire |
| `RLock.release()` -> `RLock.acquire()` | Same as Lock |
| `Event.set()` -> `Event.wait()` | Set happens-before wait returns |
| `Condition.notify()` -> `Condition.wait()` | Notify happens-before wait returns |
| `Semaphore.release()` -> `Semaphore.acquire()` | Release happens-before acquire |
| `Queue.put()` -> `Queue.get()` | Put happens-before get returns |
| `Thread.start()` -> first op in child | Start happens-before child's first event |
| last op in child -> `Thread.join()` | Child's last event happens-before join returns |
| `Barrier.wait()` -> `Barrier.wait()` | All arrivals happen-before all departures |

### 6.2 Implementation in the Cooperative Wrappers

The existing cooperative wrappers (`_CooperativeLock`, `_CooperativeRLock`, etc.)
already intercept acquire/release calls. They need to be extended to report
synchronization events to the DPOR engine:

```python
class _CooperativeLock:
    def acquire(self, blocking=True, timeout=-1):
        # ... existing cooperative logic ...

        # Report to DPOR engine
        if dpor_engine is not None:
            dpor_engine.process_sync(
                thread_id=current_thread_id(),
                event=SyncEvent.LockAcquire(
                    lock_id=self._object_id,
                    release_vv=self._last_release_vv,
                ),
            )

    def release(self):
        # Report to DPOR engine
        if dpor_engine is not None:
            self._last_release_vv = dpor_engine.process_sync(
                thread_id=current_thread_id(),
                event=SyncEvent.LockRelease(lock_id=self._object_id),
            )

        # ... existing cooperative logic ...
```

### 6.3 Async Happens-Before

For async code, the happens-before relation is simpler because the event loop is
single-threaded. Race conditions only occur at `await` points. The happens-before
edges are:

| Primitive | Happens-Before Edge |
|-----------|-------------------|
| `asyncio.Lock.release()` -> `asyncio.Lock.acquire()` | Same as threading |
| `asyncio.Event.set()` -> `asyncio.Event.wait()` | Same as threading |
| `await` (yield to event loop) -> next task resumes | Scheduling order is the happens-before |
| `Task.cancel()` -> `CancelledError` raised | Cancel happens-before exception |

For async DPOR, the custom `FrontrunEventLoop` proposed in FUTURE_WORK.md would
intercept all task scheduling decisions and report them to the DPOR engine.

---

## 7. Preemption Bounding

Full DPOR can still explore a large number of interleavings. **Preemption bounding**
limits exploration to executions with at most `k` preemptions (forced context
switches away from a thread that could have continued running). Research (Musuvathi
and Qadeer, PLDI 2007) shows that most concurrency bugs manifest within 2-3
preemptions.

### 7.1 How It Works

Each `Branch` in the exploration tree tracks the cumulative preemption count. A
context switch from thread A to thread B is a preemption if:
- Thread A is still runnable (not blocked, not finished)
- Thread B is different from A

When the DPOR algorithm would insert a backtrack point that would cause the
preemption count to exceed the bound, it applies a **conservative fallback**: instead
of adding the backtrack at the exact conflict point, it searches backward for an
earlier branch where the same thread can be explored within the bound.

### 7.2 Soundness Under Bounding

Bounded DPOR is **unsound** in the strict sense: it may miss interleavings that
require more preemptions than the bound allows. However:

- With a bound of 0 (no preemptions), it explores all **non-preemptive** schedules.
- With a bound of 2, it catches the vast majority of real concurrency bugs.
- The user can increase the bound for higher confidence, trading off exploration time.

The API exposes the bound as an explicit parameter with a clear warning:

```python
result = explore_dpor(
    ...,
    preemption_bound=2,  # Explore up to 2 preemptions (default)
    # preemption_bound=None,  # Unbounded: explore ALL interleavings
)
```

---

## 8. Optimal DPOR (Future Enhancement)

The algorithm described above implements **classic DPOR** (Flanagan & Godefroid,
2005), which may explore some Mazurkiewicz traces more than once. **Optimal DPOR**
(Abdulla et al., POPL 2014) guarantees that each distinct trace is explored
**exactly once**, using *source sets* and *wakeup trees* instead of simple
backtrack sets.

### 8.1 Key Differences

| Aspect | Classic DPOR | Optimal DPOR |
|--------|-------------|--------------|
| Backtrack mechanism | Backtrack sets (may overlap) | Source sets + wakeup trees |
| Redundancy | May explore equivalent traces multiple times | Each trace explored exactly once |
| Complexity | Simpler to implement | More complex bookkeeping |
| Memory | Lower (just backtrack flags) | Higher (wakeup trees) |
| Total explorations | More (redundant) | Provably minimal |

### 8.2 Why Defer It

Classic DPOR with preemption bounding is sufficient for the initial implementation
and provides substantial improvement over random exploration. Optimal DPOR can be
added later as a second exploration strategy without changing the architecture —
the difference is entirely in how the `Path` computes backtrack points.

Loom itself uses classic bounded DPOR, not optimal DPOR (see
[loom#79](https://github.com/tokio-rs/loom/issues/79)). This validates the approach:
classic DPOR is practically effective for real-world concurrency testing.

### 8.3 Source-Sets DPOR Sketch

For future reference, here is a sketch of how source-set DPOR would modify the
algorithm:

```
PROCEDURE OptimalDPOR(program):
    // Instead of backtrack sets, maintain source sets at each branch.
    // A source set at branch b is a set of threads such that exploring
    // ANY ONE of them is sufficient to cover all distinct traces through b.
    //
    // Classic DPOR adds ALL conflicting threads to the backtrack set.
    // Optimal DPOR adds only ONE thread to the source set, but uses
    // wakeup trees to ensure the right one is chosen.
    //
    // See Abdulla et al. 2014 for the full algorithm.
```

---

## 9. PyO3 Interface

### 9.1 Exposed Python Classes

```rust
use pyo3::prelude::*;

/// The DPOR exploration engine, exposed to Python.
#[pyclass]
struct DporEngine {
    inner: engine::DporEngine,
}

#[pymethods]
impl DporEngine {
    #[new]
    fn new(
        num_threads: usize,
        preemption_bound: Option<u32>,
        max_branches: usize,
        max_executions: Option<u64>,
    ) -> Self { ... }

    /// Start a new execution. Resets per-execution state.
    fn begin_execution(&mut self) { ... }

    /// Pick the next thread to run. Returns thread ID or None (deadlock).
    fn schedule(&mut self, runnable: Vec<usize>) -> Option<usize> { ... }

    /// Report a shared memory access.
    fn report_access(
        &mut self,
        thread_id: usize,
        object_id: u64,  // Opaque ID from Python
        kind: &str,       // "read" or "write"
    ) { ... }

    /// Report a synchronization event.
    fn report_sync(
        &mut self,
        thread_id: usize,
        event_type: &str,  // "lock_acquire", "lock_release", "thread_join", etc.
        sync_id: u64,      // ID of the sync primitive
    ) { ... }

    /// Finish the current execution and advance to the next path.
    /// Returns true if there's another path to explore.
    fn next_execution(&mut self) -> bool { ... }

    /// Get the total number of executions completed.
    #[getter]
    fn executions_completed(&self) -> u64 { ... }

    /// Get the current exploration tree depth (number of branches).
    #[getter]
    fn tree_depth(&self) -> usize { ... }

    /// Check if exploration is complete.
    #[getter]
    fn is_complete(&self) -> bool { ... }
}
```

### 9.2 Build Configuration

```toml
# frontrun-dpor/Cargo.toml
[package]
name = "frontrun-dpor"
version = "0.1.0"
edition = "2021"

[lib]
name = "frontrun_dpor"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
```

```toml
# frontrun-dpor/pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["pyo3/extension-module"]
```

---

## 10. Integration with Existing Frontrun Infrastructure

### 10.1 Bytecode Mode Integration

The existing `explore_interleavings()` function generates random schedules. DPOR
replaces the schedule generation with systematic exploration:

```python
# Current API (random):
result = explore_interleavings(
    setup=lambda: Counter(0),
    threads=[lambda c: c.increment(), lambda c: c.increment()],
    invariant=lambda c: c.value == 2,
    max_attempts=200,
)

# New API (DPOR):
result = explore_dpor(
    setup=lambda: Counter(0),
    threads=[lambda c: c.increment(), lambda c: c.increment()],
    invariant=lambda c: c.value == 2,
    preemption_bound=2,
)
```

The internal execution machinery (cooperative wrappers, opcode tracing) remains
the same. The difference is that the schedule is generated by the DPOR engine
rather than randomly.

### 10.2 Trace Marker Mode Integration

For trace markers with conflict annotations (Approach B from section 5), the
`TraceExecutor` can optionally use DPOR to generate schedules:

```python
executor = TraceExecutor(strategy="dpor", preemption_bound=2)
executor.run("t1", lambda: account.transfer(50))
executor.run("t2", lambda: account.transfer(50))
results = executor.explore_all()  # Returns list of DporResult
```

### 10.3 API Compatibility

DPOR is exposed as a new exploration strategy alongside the existing random
strategy, not as a replacement. The existing `explore_interleavings()` API
remains unchanged. Users opt into DPOR explicitly.

---

## 11. Correctness Properties

### 11.1 Soundness

The DPOR implementation must satisfy:

1. **No false negatives (unbounded mode).** If a bug exists in any interleaving,
   DPOR will find it. This requires:
   - The dependency relation is an overapproximation of the true dependency
     (i.e., independent operations are never falsely classified as dependent,
     but dependent operations may be falsely classified as independent — wait,
     this is backwards). Actually:
   - The independence relation must be **sound**: if two operations are classified
     as independent, they must truly be independent (commutative). False
     independence leads to missed interleavings (unsoundness). False dependence
     leads to unnecessary exploration (redundancy, but not unsoundness).
   - All scheduling points are observed (no untracked thread switches).

2. **Termination.** The exploration must terminate. This is guaranteed because:
   - The number of threads is finite.
   - Each thread executes a finite number of operations.
   - The exploration tree has finite branching factor (number of threads) and
     finite depth (total operations).
   - Each branch point is visited at most once per thread.

### 11.2 Liveness

Deadlock detection: if at any scheduling point no thread is runnable, the engine
reports a deadlock. This is a valid bug report — the interleaving led to a deadlock.

### 11.3 Python-Specific Soundness Concerns

1. **GIL interactions.** CPython's Global Interpreter Lock serializes Python
   bytecode execution, but C extensions can release the GIL. Frontrun's opcode
   tracing only sees Python-level operations. Accesses that happen inside C
   extensions (while the GIL is released) are invisible to DPOR. This is a
   fundamental limitation — the same limitation applies to loom, which only sees
   operations on `loom::sync` types.

2. **Object identity instability.** Python's `id()` returns the memory address,
   which can be reused after an object is garbage collected. For short-lived
   objects, this could cause false aliasing in `ObjectId`. Mitigation: use a
   generation counter or weak reference tracking.

3. **Hash-based object IDs.** Dictionary keys used in `ObjectId::DictKey` are
   identified by `repr(key)`. For unhashable or mutable keys, this may not
   produce stable identifiers. Mitigation: restrict to hashable keys and use
   `(id(dict), hash(key))` as the identifier.

---

## 12. Testing Strategy

### 12.1 Unit Tests (Rust)

Test the core data structures and algorithm in isolation:

- **VersionVec:** `join`, `partial_le`, `concurrent_with` with various orderings.
- **Path:** `schedule`, `backtrack`, `step` — verify correct DFS traversal and
  backtrack point insertion.
- **Dependency detection:** Verify that Read/Read is independent, Read/Write and
  Write/Write are dependent.
- **Known examples:** The classic "two-thread counter increment" should explore
  exactly the distinct interleavings (not more, not fewer).

### 12.2 Integration Tests (Python)

- **Counter increment:** Two threads incrementing a shared counter. DPOR should
  find the lost-update bug and report a counterexample.
- **Lock-protected counter:** Same, but with proper locking. DPOR should explore
  all interleavings and find no bugs.
- **Bank account transfer:** The race condition from the README. DPOR should find it.
- **Producer-consumer queue:** Verify that DPOR correctly handles Queue put/get
  happens-before edges.
- **Comparison with random:** Run the same test with both `explore_interleavings()`
  and `explore_dpor()`. The DPOR result should be definitive (either finds the bug
  or proves its absence), while random may or may not find it.

### 12.3 Regression Tests Against Loom

For the Rust core, port loom's own unit tests to verify that our DPOR implementation
produces the same exploration behavior on equivalent programs.

---

## 13. Implementation Plan

### Phase 1: Rust Core (MVP)

Implement the minimal Rust DPOR engine:

- [ ] `VersionVec` with `new`, `increment`, `join`, `partial_le`
- [ ] `Access` and `ObjectState` with dependency tracking
- [ ] `Branch`, `Path` with `schedule`, `backtrack`, `step`
- [ ] `DporEngine` with `process_access`, `schedule`, `next_execution`
- [ ] Unit tests for all core data structures
- [ ] PyO3 bindings exposing `DporEngine` to Python

### Phase 2: Python Integration (Bytecode Mode)

Connect the Rust engine to the existing bytecode instrumentation:

- [ ] Extend opcode trace callback to detect shared-memory opcodes
- [ ] Map opcodes to `(object_id, access_kind)` tuples
- [ ] Implement `explore_dpor()` using `DporEngine` for schedule generation
- [ ] Integration tests: counter, bank account, producer-consumer

### Phase 3: Synchronization-Aware Happens-Before

Add happens-before tracking to reduce false dependencies:

- [ ] Extend cooperative wrappers to report sync events to DPOR engine
- [ ] Implement `process_sync` for Lock, RLock, Event, Condition, Semaphore, Queue
- [ ] Implement thread spawn/join happens-before
- [ ] Test: lock-protected counter should explore fewer interleavings than unprotected

### Phase 4: Preemption Bounding

- [ ] Track preemption count in `Branch`
- [ ] Implement bounded backtracking in `Path::backtrack`
- [ ] Implement conservative fallback for over-bound backtrack points
- [ ] Expose `preemption_bound` parameter in Python API

### Phase 5: Trace Marker Integration

- [ ] Design conflict annotation syntax for trace markers
- [ ] Implement `MarkerConflictSource`
- [ ] Integrate with `TraceExecutor`

### Phase 6: Optimization and Polish

- [ ] Profile and optimize hot paths (vector clock operations, tree traversal)
- [ ] Add progress reporting (executions/sec, tree depth, estimated remaining)
- [ ] Add checkpoint/resume support (serialize exploration tree to disk)
- [ ] Documentation and examples

### Future: Optimal DPOR

- [ ] Implement source sets and wakeup trees
- [ ] Benchmark against classic DPOR on real-world test cases
- [ ] Expose as `strategy="optimal_dpor"` option

---

## 14. References

### Foundational Papers

- Flanagan, C. and Godefroid, P. **"Dynamic Partial-Order Reduction for Model
  Checking Software."** POPL 2005.
  *The original DPOR algorithm. Introduces persistent sets and sleep sets for
  pruning equivalent interleavings.*

- Abdulla, P., Aronis, S., Jonsson, B., and Sagonas, K. **"Optimal Dynamic
  Partial Order Reduction."** POPL 2014.
  *Introduces source sets and wakeup trees for provably minimal exploration.
  Each Mazurkiewicz trace is explored exactly once.*

- Musuvathi, M. and Qadeer, S. **"Iterative Context Bounding for Systematic
  Testing of Multithreaded Programs."** PLDI 2007.
  *Preemption bounding: most bugs manifest within 2-3 preemptions. Provides a
  practical way to bound exhaustive exploration.*

- Norris, B. and Demsky, B. **"CDSChecker: Checking Concurrent Data Structures
  Written with C/C++ Atomics."** OOPSLA 2013.
  *The paper loom is based on. Models C11 memory model with reads-from exploration
  and modification orders.*

### Tools

- **Loom** (tokio-rs/loom): Rust concurrency testing tool implementing bounded
  DPOR. The primary reference implementation for this spec.
  https://github.com/tokio-rs/loom

- **Shuttle** (awslabs/shuttle): Rust randomized concurrency testing with PCT
  (Probabilistic Concurrency Testing). Complementary to DPOR.
  https://github.com/awslabs/shuttle

- **CHESS** (Microsoft Research): Systematic concurrency testing for .NET with
  preemption bounding. Foundational work that influenced loom and Coyote.

- **Coyote** (Microsoft): Binary rewriting + controlled scheduling for C#.
  https://microsoft.github.io/coyote/

### Python-Specific

- **PyO3**: Rust bindings for Python.
  https://pyo3.rs/

- **Maturin**: Build tool for PyO3 crates.
  https://www.maturin.rs/

- **sys.settrace / f_trace_opcodes**: CPython's tracing infrastructure used by
  frontrun's bytecode instrumentation.
  https://docs.python.org/3/library/sys.html#sys.settrace
