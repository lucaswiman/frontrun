# Test ownership

Add regressions to the suite for the behavior they protect, not to a new
`test_bugfixes`, `test_defect<N>`, or agent-session file. Keep issue numbers in
the test's docstring when they explain the failure mode.

Current consolidation boundaries:

| Behavior | Suite |
| --- | --- |
| Result assertions and representations | `test_interleaving_result.py` |
| Patch rollback and native-consumer compatibility | `test_patching.py` |
| Shared runner lifetime and deadlines | `test_threaded_runner_helpers.py` |
| Shared DPOR ownership and exploration contracts | `test_dpor_core.py` |
| Sync conditions, locks, scheduler reentrancy | `test_cooperative_condition.py`, `test_cooperative_locks.py`, `test_cooperative_reentrancy.py` |
| Async primitives, failures, task context | `test_async_cooperative.py`, `test_async_scheduler_failures.py`, `test_async_task_context.py` |
| Positional/I/O replay and access-anchor drift | `test_dpor_replay.py`, `test_dpor_access_replay.py` |
| Redis replay and PostgreSQL connection identity | `test_redis_replay.py`, `test_postgres_connection_identity.py` |
| SQL syntax, parameters, predicates, transactions | `test_sql_syntax.py`, `test_sql_params.py`, `test_sql_predicates.py`, `test_sql_transactions.py` |

Other existing feature suites remain their features' owners; this is not a
complete index. A new file is appropriate for a distinct component or a
different dependency/fixture boundary, not simply a new bug report.

## Consolidating coverage

- Parameterize the same setup/action/assertion over meaningful inputs. Preserve
  execution modes, operation kinds, deadlines, and expected failures.
- Merge tests only when the surviving assertions subsume the removed contract.
  Similar names or matching test counts are not evidence of equivalent coverage.
- Keep end-to-end regressions and differential oracles when they cover paths
  that a helper-level unit test cannot. Avoid source-text assertions and
  probabilistic tests that usually skip when a deterministic contract is available.
- Share fixtures at the narrowest useful scope. Moving a test must not silently
  make it inherit unrelated autouse fixtures, service requirements, or skips.
- Measure net lines including helpers and documentation. Moving files alone is
  organization, not a code reduction.

Run tests through the Makefile wrappers described in `CLAUDE.md`. If agents share
one checkout, serialize those wrappers: concurrent native builds can replace
the extension while another test process is starting.
