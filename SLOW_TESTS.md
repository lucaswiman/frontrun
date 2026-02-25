# Slow Tests on 3.14t (Free-Threaded Python)

Benchmarked on CPython 3.14.0rc2 (free-threaded) using `frontrun pytest` with the
LD_PRELOAD I/O interception library active. Full suite: 236 tests, 2 skipped.

## Summary

Full suite completes in ~58-60s on most runs. However, two tests intermittently
hang for minutes (observed both in isolated runs and in GitHub Actions CI).

### Consistently Slow Tests

| Test | Typical time | Notes |
|------|-------------|-------|
| `test_integration_http.py::TestHttpCounterRace::test_naive_threading_race_rate` | **12-14s** | Runs 500 HTTP trials with random timing offsets |
| `test_concurrency_bug_classes.py::test_atomicity_violation_hypothesis` | **5-7s** | Hypothesis with 200 examples, each running `run_with_schedule` |
| `test_threading_primitives.py::test_queue_put_race_condition` | **4-5s** | `explore_interleavings` with 100 attempts, 400 max_ops, 3 threads |
| `test_concurrency_bug_classes.py::test_order_violation_hypothesis` | **4-6s** | Hypothesis with 200 examples |
| `test_threading_primitives.py::test_rlock_race_condition` | **3-5s** | `explore_interleavings` with 100 attempts, 300 max_ops |
| `test_integration_json_file.py::test_bytecode_detects_lost_update` | **2-3s** | |
| `test_concurrency_bug_classes.py::test_deadlock_exact_schedule` | **2-3s** | Waits for 1s deadlock timeout, then verifies threads are stuck |
| `test_concurrency_bug_classes.py::test_async_suspension_point_race_hypothesis` | **1-2s** | Hypothesis with 200 examples |

### Intermittently Hanging Tests

These tests usually complete in 2-7s but occasionally hang indefinitely:

1. **`test_integration_http.py::TestHttpTransferRace::test_dpor_detects_transfer_anomaly`**
   - Normal: ~2-3s
   - Observed: timed out at 60s and 90s in separate runs
   - Uses `explore_dpor` with `detect_io=True` over HTTP sockets
   - The hang likely occurs in the DPOR scheduler when HTTP I/O events
     interact with the LD_PRELOAD pipe or the `DporScheduler._report_and_wait`
     condition variable

2. **`test_concurrency_bug_classes.py::test_order_violation_hypothesis`**
   - Normal: ~5-6s
   - Observed: timed out at 120s in one run
   - Uses Hypothesis to generate 200 random schedules via `run_with_schedule`
   - The hang likely occurs when a particular Hypothesis-generated schedule
     causes a thread to block in `OpcodeScheduler.wait_for_turn` and the
     5s `deadlock_timeout` fails to fire (missed condition notification)

## Root Cause Analysis

### Intermittent Hangs

The hanging mechanism involves the `_report_and_wait` / `wait_for_turn` loop in
both `DporScheduler` and `OpcodeScheduler`:

```python
while True:
    if self._finished or self._error:
        return False
    if self._current_thread == thread_id:
        # ... schedule next, notify all
        return True
    if not self._condition.wait(timeout=self.deadlock_timeout):
        # timeout fired — check if current thread is done
        # ... or raise TimeoutError
```

On free-threaded Python (3.14t), condition variable notifications can be lost
under high contention because:

1. `notify_all()` only wakes threads currently in `wait()` — a thread that
   hasn't entered `wait()` yet misses the notification entirely
2. The `deadlock_timeout` (default 5.0s) should catch this, but with HTTP I/O
   the C-level send/recv can block a thread outside the Python scheduler's
   control, causing the timeout check to stall
3. The `DporBytecodeRunner.run()` join loop uses a shared deadline:
   ```python
   deadline = time.monotonic() + timeout
   for t in self.threads:
       remaining = max(0, deadline - time.monotonic())
       t.join(timeout=remaining)
   ```
   If the first thread consumes most of the timeout, later threads get
   `join(timeout=0)` and are effectively abandoned (daemon threads).

### Consistently Slow Tests

- **`test_naive_threading_race_rate`**: Inherently slow — 500 HTTP round-trips
  with random delays (0-15ms each). This is a demonstration test, not a unit test.
- **Hypothesis tests**: 200 examples each, where each example does a full
  `run_with_schedule` cycle (thread creation, monitoring setup/teardown).
  The overhead per example is ~25-30ms.
- **`test_queue_put_race_condition`**: 3 threads x 100 attempts x 400 max_ops
  is a large search space.
- **`test_rlock_race_condition`**: 2 threads x 100 attempts x 300 max_ops with
  reentrant locking adds cooperative lock overhead.

## Recommendations

1. **Add per-test timeouts** via `@pytest.mark.timeout(30)` to prevent CI hangs
2. **Reduce `test_naive_threading_race_rate` trials** from 500 to ~100 (still
   enough to observe the race, saves ~10s)
3. **Reduce Hypothesis `max_examples`** from 200 to 50 for the slow tests
   (still sufficient to explore the interleaving space)
4. **Reduce `explore_interleavings` `max_attempts`** in threading primitive
   tests (e.g., 100 → 30 for `test_rlock_race_condition`)
5. **Add a total timeout to `explore_dpor`** to bound the entire exploration,
   not just individual runs
