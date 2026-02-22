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
