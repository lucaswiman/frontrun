# DPOR Improvements Roadmap

Remaining work for the optimal DPOR implementation. Last reviewed: 2026-06-12.

Everything from the original optimal-DPOR plan through Fix 6 and Defect #15 Approach 2
(resource grouping) is implemented — see git history of this file for the full list.
What remains below is low-priority and should only be picked up with a motivating
workload in hand.

---

## Phase 4c: Wakeup Tree Insert Equivalence (Optimization)

**Paper ref**: Definition 6.2 (JACM'17 p.22), Lemma 6.2 (p.22)

Before inserting a new sequence into the wakeup tree, check if any existing leaf
explores an equivalent interleaving using independence information.

**Complexity**: Medium. **Priority**: Low. Benefit uncertain without empirical data.

---

## Fix 7: Reduce Scheduling Points for Redundant Shared Opcodes

Selectively suppress shared opcodes redundant with stronger semantic events:
- Lock metadata lookups (`LOAD_ATTR`, `LOAD_SPECIAL` on lock objects) already covered by `report_sync()`
- Container iteration reads already pinned to first access
- Immutable-object lookups already covered by `_IMMUTABLE_TYPES`

**Complexity**: Low–Medium. **Priority**: Low. No impact on trace counts, only per-execution overhead.

---

## Defect #15: Complex SQL Race Detection (Remaining Approaches)

Approach 2 (resource grouping) is implemented. Remaining approaches for further improvement:

### Approach 1: Conflict Prioritization

Prioritize backtrack points involving `:seq` resources (phantom-read conflicts) over same-row
conflicts. May improve convergence for multi-table patterns.

**Complexity**: Medium. **Priority**: Low.

### Approach 3: Targeted Exploration (User-Facing)

Users annotate which resources are "interesting" for race detection. DPOR only creates
backtrack points for those resources.

**Complexity**: Low. **Priority**: Low.

### Approach 4: Hybrid Random + Systematic (Fallback)

Use DPOR for small trees; fall back to random scheduling when tree exceeds threshold.

**Complexity**: Low–Medium. **Priority**: Low.
