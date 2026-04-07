"""Shared pytest configuration for SCS tests."""
import sys

import pytest

# Detect free-threaded build (3.13t+)
NOGIL_BUILD = hasattr(sys, "_is_gil_enabled")
gil_enabled_at_start = True
if NOGIL_BUILD:
    gil_enabled_at_start = sys._is_gil_enabled()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Fail the test run if the GIL was re-enabled during tests.

    Modeled after NumPy's conftest.py — if a C extension module is missing
    the Py_MOD_GIL_NOT_USED declaration, CPython will silently re-enable
    the GIL at import time, defeating free-threading. This hook catches that.
    """
    if NOGIL_BUILD and not gil_enabled_at_start and sys._is_gil_enabled():
        tr = terminalreporter
        tr.ensure_newline()
        tr.section("GIL re-enabled", sep="=", red=True, bold=True)
        tr.line("The GIL was re-enabled at runtime during the tests.")
        tr.line("This can happen with no test failures if the RuntimeWarning")
        tr.line("raised by Python when this happens is filtered by a test.")
        tr.line("")
        tr.line("Please ensure all C extension modules declare")
        tr.line("Py_MOD_GIL_NOT_USED via PyUnstable_Module_SetGIL().")
        pytest.exit("GIL re-enabled during tests", returncode=1)
