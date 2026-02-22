# Contributing to Frontrun

Bug fixes accompanied by tests are welcome.

Before submitting a pull request:

1. Verify that `make check` passes (ruff + pyright).
2. Verify that `make test` passes (runs the test suite across Python 3.10,
   3.14, and 3.14t).
3. Keep the diff clean: it should contain only changes relevant to the fix.
   Avoid unrelated reformatting, import reordering, comment rewording, or
   other stylistic changes.

Use of coding agents (Claude Code, Copilot, Cursor, etc.) is encouraged, but
please note which agent(s) were used in the PR description.

This project is **not currently seeking contributors for new feature work**.
If you have an idea for a new feature, please open an issue to discuss it
rather than submitting a PR.

## Windows Compatibility

Windows compatibility patches are **especially welcome**.  Currently only
trace markers, bytecode exploration, and DPOR work on Windows (the
pure-Python and Rust PyO3 components).  The `frontrun` CLI and C-level
I/O interception library rely on `LD_PRELOAD` / `DYLD_INSERT_LIBRARIES`
and have no Windows equivalent.  Possible avenues include:

- IAT (Import Address Table) hooking or Detours-style function patching
  for the I/O interception library.
- A Windows-compatible `frontrun` CLI wrapper.
- CI and test-suite support for Windows.
