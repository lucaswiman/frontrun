//! DPOR (Dynamic Partial Order Reduction) prototype engine for frontrun.
//!
//! This implements the classic DPOR algorithm (Flanagan & Godefroid, POPL 2005)
//! with optional preemption bounding (Musuvathi & Qadeer, PLDI 2007).
//!
//! The design follows loom's approach (tokio-rs/loom) adapted for Python's
//! concurrency model: instead of C11 atomics, we track Python-level shared
//! memory accesses (attribute reads/writes, dict operations, etc.).

pub mod vv;
pub mod access;
pub mod object;
pub mod thread;
pub mod path;
pub mod engine;

pub use vv::VersionVec;
pub use access::{Access, AccessKind};
pub use object::{ObjectId, ObjectState};
pub use thread::{Thread, ThreadStatus};
pub use path::{Branch, Path};
pub use engine::{DporEngine, Execution, SyncEvent, ExplorationResult, Step, run_model_simple};
