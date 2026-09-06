"""The pre-3.3.0 boolean solver-selection flags still work, with a warning.

3.3.0 replaced `use_indirect`/`gpu`/`mkl`/`cudss` with the `LinearSolver` enum.
Removing them outright meant an upgrading caller got a bare `TypeError` from the
C extension, naming the rejected kwarg but not its replacement. These tests pin
the deprecation path that replaced that: the flags are translated, and each
translation announces itself.

The mapping is asserted against `_pop_legacy_solver_flags` directly rather than
by solving, so the GPU/MKL/cuDSS cases are checked on builds that do not ship
those extensions.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import scs
from scs import LinearSolver, _pop_legacy_solver_flags

# maximize x  subject to  0 <= x <= 1
_A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
_B = np.array([1.0, 0.0])
_C = np.array([-1.0])
_CONE = {"l": 2}


def _data():
    return {"A": _A.copy(), "b": _B.copy(), "c": _C.copy()}


# ---------------------------------------------------------------------------
# The mapping, including the two quirks inherited from 3.2.11
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"use_indirect": True}, LinearSolver.CPU_INDIRECT),
        ({"use_indirect": False}, LinearSolver.QDLDL),
        ({"mkl": True}, LinearSolver.MKL),
        ({"mkl": True, "use_indirect": False}, LinearSolver.MKL),
        ({"mkl": False}, LinearSolver.QDLDL),
        # 3.2.11: `gpu=True` alone fell through to the cuDSS branch, because
        # `use_indirect` defaulted to False. Preserved deliberately.
        ({"gpu": True}, LinearSolver.CUDSS),
        ({"gpu": True, "use_indirect": False}, LinearSolver.CUDSS),
        ({"gpu": True, "use_indirect": True}, LinearSolver.GPU_INDIRECT),
        ({"gpu": False}, LinearSolver.QDLDL),
        # 3.2.11 rejected this outright; it now means what it says.
        ({"cudss": True}, LinearSolver.CUDSS),
    ],
)
def test_legacy_flags_map_to_enum(flags, expected):
    stgs = dict(flags)
    with pytest.warns(DeprecationWarning):
        assert _pop_legacy_solver_flags(stgs) is expected
    # The flags are consumed, so they never reach the C extension.
    assert stgs == {}


def test_no_legacy_flag_returns_none_and_does_not_warn():
    stgs = {"verbose": False, "eps_abs": 1e-9}
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _pop_legacy_solver_flags(stgs) is None
    assert stgs == {"verbose": False, "eps_abs": 1e-9}


def test_mkl_indirect_is_refused():
    with pytest.raises(NotImplementedError, match="no MKL indirect solver"):
        _pop_legacy_solver_flags({"mkl": True, "use_indirect": True})


# ---------------------------------------------------------------------------
# The warning itself
# ---------------------------------------------------------------------------


def test_warning_names_the_replacement():
    with pytest.warns(DeprecationWarning) as record:
        _pop_legacy_solver_flags({"use_indirect": True})
    message = str(record[0].message)
    assert "`use_indirect`" in message
    assert "linear_solver=scs.LinearSolver.CPU_INDIRECT" in message
    assert "3.3.0" in message


def test_one_warning_lists_every_flag_passed():
    with pytest.warns(DeprecationWarning) as record:
        _pop_legacy_solver_flags({"gpu": True, "use_indirect": True})
    assert len(record) == 1
    message = str(record[0].message)
    assert "`gpu`" in message and "`use_indirect`" in message


# ---------------------------------------------------------------------------
# End to end, on backends every build has
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_indirect", [True, False])
def test_solving_with_legacy_flag_still_works(use_indirect):
    with pytest.warns(DeprecationWarning):
        sol = scs.SCS(
            _data(), _CONE, use_indirect=use_indirect, verbose=False
        ).solve()
    assert sol["info"]["status"] == "solved"
    np.testing.assert_almost_equal(sol["x"][0], 1.0, decimal=3)


def test_legacy_flag_reaches_the_intended_backend():
    with pytest.warns(DeprecationWarning):
        solver = scs.SCS(_data(), _CONE, use_indirect=True, verbose=False)
    assert "indirect" in solver.solve()["info"]["lin_sys_solver"].lower()


def test_explicit_linear_solver_wins_over_legacy_flag():
    """The legacy flag supplies a default; it does not override a choice."""
    with pytest.warns(DeprecationWarning):
        solver = scs.SCS(
            _data(),
            _CONE,
            use_indirect=True,
            linear_solver=LinearSolver.QDLDL,
            verbose=False,
        )
    info = solver.solve()["info"]
    assert info["status"] == "solved"
    assert "indirect" not in info["lin_sys_solver"].lower()


def test_legacy_solve_helper_accepts_the_flags():
    with pytest.warns(DeprecationWarning):
        sol = scs.solve(_data(), _CONE, use_indirect=False, verbose=False)
    assert sol["info"]["status"] == "solved"
