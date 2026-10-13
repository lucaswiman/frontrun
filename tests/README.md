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

## Native Python compatibility

For each newly supported interpreter, first build with `make build-dpor-<version>`
without PyO3's forward-compatibility override. Import the resulting extension
using that venv's Python and check that importing it leaves the GIL disabled on
free-threaded builds. Run `make test-<version>` and `make test-e2e-<version>` to
exercise both in-process scheduling and worker bootstrap/replay.

For Python 3.15 support, start with `make build-dpor-3.15` and
`make build-dpor-3.15t`. PyO3 0.25 fails this procedure because its maximum
supported interpreter is Python 3.14.

The artifact smoke test must also finish under the Makefile's preload wrapper:
`make test-e2e-3.15 PYTEST_ARGS="-v tests/test_sdist_build.py"`. During the
3.15 compatibility run, the wheel build stalled with a Cargo child blocked
before executing rustc. Compiler subprocesses inherited the preload library;
its process-local mutexes are unsafe across a multithreaded Cargo fork. Keep
pytest instrumented while running these external build tools without preload.

## PostgreSQL driver baseline

The PostgreSQL trace-count and SQLAlchemy helper fixtures probe psycopg2, the
driver installed by `make build-integration-<version>`. SQLAlchemy 2.1 changed
an unqualified `postgresql://` engine URL to select psycopg3 instead.

With the native extensions already built and `DATABASE_URL` pointing at the
test database, reproduce the driver mismatch with:

```sh
make -o build-dpor-3.15 -o build-io test-integration-3.15 \
  PYTEST_ARGS="-v tests/test_exact_mazurkiewicz_trace_count.py::TestPostgreSQLTraceCount::test_two_threads_n_shared_rows_select_for_update[1] tests/test_integration_sqlalchemy_dpor.py::test_sqlalchemy_dpor_toctou_race"
```

The unqualified fixtures failed identically on Python 3.14 and 3.15 with the
same dependencies: the one-row trace count was 8 instead of at most 6, and
TOCTOU INSERT tracking raised `NondeterministicSQLError` because psycopg3 has no
`lastrowid`. After explicitly selecting psycopg2 in these fixtures, rerun all
three shared-row parameters and the TOCTOU test. Keep their trace bounds and
nondeterminism guard enabled; the ORM fixture and real dual-driver cursor
regression retain psycopg3 coverage.
