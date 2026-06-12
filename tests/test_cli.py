"""Tests for the command-line interface."""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

from frontrun.cli import main


def test_usage_goes_to_stderr() -> None:
    fake_stderr = io.StringIO()
    fake_stdout = io.StringIO()

    with patch.object(sys, "stderr", fake_stderr), patch.object(sys, "stdout", fake_stdout):
        ret = main([])

    assert ret == 1
    assert fake_stdout.getvalue() == ""
    stderr_output = fake_stderr.getvalue()
    assert "Usage: frontrun" in stderr_output
    assert "Examples:" in stderr_output
    assert "Environment variables:" in stderr_output
