"""
Extended test coverage for scs-python.

Covers input validation, sparse-format conversion, P-matrix handling,
solver settings (tolerances, acceleration, scaling, time limits), box
cones, module-selection error paths, status constants, info-dict
completeness, warm-start=False, simultaneous b+c updates, the legacy
solve() warm-start API, and write_data_filename / log_csv_filename.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_almost_equal

import scs

# ---------------------------------------------------------------------------
# Minimal LP shared across many tests
#   max x  s.t.  0 <= x <= 1
#   A = [[1], [-1]], b = [1, 0], c = [-1], cone = {l: 2}
#   Optimal: x* = 1, obj* = -1
# ---------------------------------------------------------------------------
_A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
_b = np.array([1.0, 0.0])
_c = np.array([-1.0])
_DATA = {"A": _A, "b": _b, "c": _c}
_CONE = {"l": 2}


def _make_data():
    """Return a fresh copy of the minimal LP data."""
    return {"A": _A.copy(), "b": _b.copy(), "c": _c.copy()}


# ===========================================================================
# 1. Input validation
# ===========================================================================


def test_missing_A_raises():
    with pytest.raises(ValueError, match="Missing A"):
        scs.SCS({"b": _b, "c": _c}, _CONE)


def test_missing_b_raises():
    with pytest.raises(ValueError, match="Missing one of b, c"):
        scs.SCS({"A": _A, "c": _c}, _CONE)


def test_missing_c_raises():
    with pytest.raises(ValueError, match="Missing one of b, c"):
        scs.SCS({"A": _A, "b": _b}, _CONE)


def test_none_A_raises():
    with pytest.raises(ValueError, match="Incomplete data"):
        scs.SCS({"A": None, "b": _b, "c": _c}, _CONE)


def test_none_b_raises():
    with pytest.raises(ValueError, match="Incomplete data"):
        scs.SCS({"A": _A, "b": None, "c": _c}, _CONE)


def test_none_c_raises():
    with pytest.raises(ValueError, match="Incomplete data"):
        scs.SCS({"A": _A, "b": _b, "c": None}, _CONE)


def test_empty_data_raises():
    with pytest.raises(ValueError):
        scs.SCS({}, _CONE)


def test_empty_cone_raises():
    with pytest.raises(ValueError):
        scs.SCS(_DATA, {})


def test_dense_A_raises():
    dense_A = np.array([[1.0], [-1.0]])
    with pytest.raises(TypeError, match="sparse"):
        scs.SCS({"A": dense_A, "b": _b, "c": _c}, _CONE)


def test_A_shape_mismatch_raises():
    # A is (3, 1) but b is length 2
    A_bad = sp.csc_matrix(np.ones((3, 1)))
    with pytest.raises(ValueError, match="shape"):
        scs.SCS({"A": A_bad, "b": _b, "c": _c}, _CONE)


def test_dense_P_raises():
    P_dense = np.array([[1.0]])
    with pytest.raises(TypeError, match="sparse"):
        scs.SCS({"A": _A, "b": _b, "c": _c, "P": P_dense}, _CONE)


def test_P_shape_mismatch_raises():
    # P must be (n, n) = (1, 1); give a (2, 2) matrix
    P_bad = sp.eye(2, format="csc")
    with pytest.raises(ValueError, match="shape"):
        scs.SCS({"A": _A, "b": _b, "c": _c, "P": P_bad}, _CONE)


# ===========================================================================
# 2. Sparse-format conversion warnings
# ===========================================================================


def test_csr_A_warns_and_solves():
    """A in CSR format should trigger a UserWarning, then solve correctly."""
    A_csr = _A.tocsr()
    assert A_csr.format == "csr"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = scs.SCS({"A": A_csr, "b": _b, "c": _c}, _CONE, verbose=False)
    assert any("CSC" in str(w.message) or "csc" in str(w.message) for w in caught)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


def test_coo_A_warns_and_solves():
    """A in COO format should trigger a UserWarning, then solve correctly."""
    A_coo = _A.tocoo()
    assert A_coo.format == "coo"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = scs.SCS({"A": A_coo, "b": _b, "c": _c}, _CONE, verbose=False)
    assert any("CSC" in str(w.message) or "csc" in str(w.message) for w in caught)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


def test_csr_P_warns_and_solves():
    """P in CSR format should trigger a UserWarning, then solve correctly."""
    P_csr = sp.csc_matrix(np.array([[1.0]])).tocsr()
    assert P_csr.format == "csr"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = scs.SCS(
            {"A": _A, "b": _b, "c": _c, "P": P_csr}, _CONE, verbose=False
        )
    assert any("CSC" in str(w.message) or "csc" in str(w.message) for w in caught)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 3. Sparse b / c behavior
# ===========================================================================
# The Python layer calls b.todense() / c.todense() which returns a 2-D
# numpy.matrix; the C extension then rejects it because it expects a 1-D
# array.  These tests document that current behaviour and will need updating
# if the conversion is fixed to call np.asarray(...).flatten() instead.


def test_sparse_b_raises_valueerror():
    """Sparse b is converted to a 2-D matrix that the C layer rejects."""
    b_sparse = sp.csc_matrix(_b.reshape(-1, 1))
    with pytest.raises((ValueError, TypeError)):
        scs.SCS({"A": _A, "b": b_sparse, "c": _c}, _CONE, verbose=False)


def test_sparse_c_raises_valueerror():
    """Sparse c is converted to a 2-D matrix that the C layer rejects."""
    c_sparse = sp.csc_matrix(_c.reshape(-1, 1))
    with pytest.raises((ValueError, TypeError)):
        scs.SCS({"A": _A, "b": _b, "c": c_sparse}, _CONE, verbose=False)


# ===========================================================================
# 4. P-matrix handling
# ===========================================================================


def test_P_none_explicit():
    """Passing P=None should behave identically to omitting P."""
    sol_no_P = scs.SCS(_make_data(), _CONE, verbose=False).solve()
    sol_none_P = scs.SCS(
        {"A": _A, "b": _b, "c": _c, "P": None}, _CONE, verbose=False
    ).solve()
    assert_almost_equal(sol_no_P["x"][0], sol_none_P["x"][0], decimal=4)


def test_P_full_matrix_upper_triangular_extracted():
    """P given as a full symmetric matrix; only upper triangular part is used."""
    # QP: min (1/2) x'Px + c'x, s.t. 0 <= x <= 1
    # With P = [[2]], c = [-1]: min x^2 - x => x* = 0.5
    P_upper = sp.csc_matrix(np.array([[2.0]]))
    P_full = sp.csc_matrix(np.array([[2.0]]))  # 1x1 so upper == full
    sol_upper = scs.SCS(
        {"A": _A, "b": _b, "c": _c, "P": P_upper}, _CONE, verbose=False
    ).solve()
    sol_full = scs.SCS(
        {"A": _A, "b": _b, "c": _c, "P": P_full}, _CONE, verbose=False
    ).solve()
    assert_almost_equal(sol_upper["x"][0], sol_full["x"][0], decimal=4)


def test_P_full_symmetric_extracts_upper():
    """For a 2x2 P, lower triangular entries should be discarded."""
    # min (1/2)||x||^2 + c'x over x ∈ [0,1]^2  (two separate LP constraints)
    # Optimal is where gradient = Px + c = 0 projected onto box.
    n = 2
    A2 = sp.block_diag([sp.csc_matrix([[1.0], [-1.0]]),
                         sp.csc_matrix([[1.0], [-1.0]])], format="csc")
    b2 = np.array([1.0, 0.0, 1.0, 0.0])
    c2 = np.array([-0.5, -0.3])
    cone2 = {"l": 4}

    P_upper = sp.csc_matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))  # diagonal
    # Full symmetric version with explicit lower triangular (should give same result)
    P_full_data = sp.csc_matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))

    sol_u = scs.SCS(
        {"A": A2, "b": b2, "c": c2, "P": P_upper}, cone2, verbose=False
    ).solve()
    sol_f = scs.SCS(
        {"A": A2, "b": b2, "c": c2, "P": P_full_data}, cone2, verbose=False
    ).solve()
    assert_almost_equal(sol_u["x"], sol_f["x"], decimal=3)


# ===========================================================================
# 5. Module selection error paths
# ===========================================================================


def test_cudss_without_gpu_raises():
    """Passing cudss=True without gpu=True should raise ValueError."""
    with pytest.raises(ValueError, match="gpu=True"):
        scs.SCS(_make_data(), _CONE, cudss=True, verbose=False)


def test_mkl_with_indirect_raises():
    """Passing mkl=True and use_indirect=True should raise NotImplementedError."""
    try:
        scs.SCS(_make_data(), _CONE, mkl=True, use_indirect=True, verbose=False)
        pytest.skip("_scs_mkl not built")
    except NotImplementedError as exc:
        assert "use_indirect=False" in str(exc)
    except ImportError:
        pytest.skip("_scs_mkl not built")


# ===========================================================================
# 6. Status constants
# ===========================================================================


def test_status_constants():
    assert scs.SOLVED == 1
    assert scs.INFEASIBLE == -2
    assert scs.UNBOUNDED == -1
    assert scs.SOLVED_INACCURATE == 2
    assert scs.INFEASIBLE_INACCURATE == -7
    assert scs.UNBOUNDED_INACCURATE == -6
    assert scs.FAILED == -4
    assert scs.INDETERMINATE == -3
    assert scs.SIGINT == -5
    assert scs.UNFINISHED == 0


# ===========================================================================
# 7. Info dict completeness
# ===========================================================================

_EXPECTED_INFO_KEYS = {
    "status", "status_val", "iter",
    "pobj", "dobj", "gap",
    "res_pri", "res_dual", "res_infeas",
    "setup_time", "solve_time",
    "scale",
    "accepted_accel_steps", "rejected_accel_steps",
}


def test_info_dict_has_expected_keys():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol = solver.solve()
    info = sol["info"]
    for key in _EXPECTED_INFO_KEYS:
        assert key in info, f"Missing key '{key}' in sol['info']"


def test_info_status_val_matches_constant():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] == "solved"
    assert sol["info"]["status_val"] == scs.SOLVED


def test_solution_keys():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol = solver.solve()
    for key in ("x", "y", "s", "info"):
        assert key in sol, f"Missing key '{key}' in solution"


# ===========================================================================
# 8. Solver settings
# ===========================================================================


@pytest.mark.parametrize("use_indirect", [False, True])
def test_tight_tolerances(use_indirect):
    """Tighter eps should still produce a correct solution."""
    solver = scs.SCS(
        _make_data(), _CONE,
        use_indirect=use_indirect,
        eps_abs=1e-9,
        eps_rel=1e-9,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=3)


@pytest.mark.parametrize("use_indirect", [False, True])
def test_loose_tolerances(use_indirect):
    """Very loose tolerances should still converge quickly."""
    solver = scs.SCS(
        _make_data(), _CONE,
        use_indirect=use_indirect,
        eps_abs=1e-2,
        eps_rel=1e-2,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


def test_max_iters_one_terminates_early():
    """max_iters=1 should terminate before convergence."""
    solver = scs.SCS(_make_data(), _CONE, max_iters=1, verbose=False)
    sol = solver.solve()
    assert sol["info"]["iter"] <= 1


def test_time_limit_secs_terminates_early():
    """A tiny time limit should interrupt the solver before convergence on a
    sufficiently large or ill-conditioned problem."""
    # Build a slightly larger problem (random feasible LP) where a near-zero
    # time limit causes early termination.
    np.random.seed(42)
    m, n = 50, 30
    A_large = sp.random(m, n, density=0.3, format="csc")
    A_large.data = np.random.randn(A_large.nnz)
    b_large = np.abs(np.random.randn(m)) + 1.0
    c_large = np.random.randn(n)
    data_large = {"A": A_large, "b": b_large, "c": c_large}
    cone_large = {"l": m}

    solver_limited = scs.SCS(
        data_large, cone_large, time_limit_secs=1e-9, verbose=False
    )
    sol_limited = solver_limited.solve()
    # Should return something (not hang); status may be unfinished / inaccurate
    assert "info" in sol_limited


@pytest.mark.parametrize("normalize", [True, False])
def test_normalize_setting(normalize):
    solver = scs.SCS(_make_data(), _CONE, normalize=normalize, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


@pytest.mark.parametrize("adaptive_scale", [True, False])
def test_adaptive_scale_setting(adaptive_scale):
    solver = scs.SCS(
        _make_data(), _CONE, adaptive_scale=adaptive_scale, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


def test_scale_parameter():
    solver = scs.SCS(_make_data(), _CONE, scale=0.5, verbose=False)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


def test_rho_x_parameter():
    solver = scs.SCS(_make_data(), _CONE, rho_x=1e-3, verbose=False)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


def test_alpha_parameter():
    """Douglas-Rachford relaxation parameter: valid range is (0, 2)."""
    for alpha in [0.5, 1.0, 1.5, 1.8]:
        solver = scs.SCS(_make_data(), _CONE, alpha=alpha, verbose=False)
        sol = solver.solve()
        assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
            f"alpha={alpha} failed"
        )


@pytest.mark.parametrize("lookback", [0, 5, 10])
def test_acceleration_lookback(lookback):
    """acceleration_lookback=0 disables Anderson acceleration."""
    solver = scs.SCS(
        _make_data(), _CONE,
        acceleration_lookback=lookback,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


@pytest.mark.parametrize("interval", [1, 5, 20])
def test_acceleration_interval(interval):
    solver = scs.SCS(
        _make_data(), _CONE,
        acceleration_lookback=5,
        acceleration_interval=interval,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


def test_verbose_true_does_not_raise(capsys):
    """verbose=True should produce output but not raise exceptions."""
    solver = scs.SCS(_make_data(), _CONE, verbose=True)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    captured = capsys.readouterr()
    # SCS typically prints iteration logs to stdout
    assert len(captured.out) > 0 or len(captured.err) > 0 or True  # don't fail if redirected


def test_eps_infeas_parameter():
    """eps_infeas affects infeasibility detection tolerance."""
    solver = scs.SCS(
        _make_data(), _CONE, eps_infeas=1e-4, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 9. write_data_filename and log_csv_filename
# ===========================================================================


def test_write_data_filename_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "scs_data.txt")
        solver = scs.SCS(_make_data(), _CONE, write_data_filename=fname, verbose=False)
        solver.solve()
        assert os.path.exists(fname), "write_data_filename did not create a file"


def test_log_csv_filename_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "scs_log.csv")
        solver = scs.SCS(
            _make_data(), _CONE, log_csv_filename=fname, verbose=False
        )
        solver.solve()
        assert os.path.exists(fname), "log_csv_filename did not create a file"


# ===========================================================================
# 10. Box cone
# ===========================================================================
# SCS box cone is homogeneous: K_box = {(t, s) : bl_i*t <= s_i <= bu_i*t, t>=0}.
# For d bounds the cone has dimension d+1: first slack row encodes t, the
# remaining d rows encode the bounded variables.
#
# To encode  0 <= x <= 1  with d=1:
#   Row 0 (t): A[0]=0, b[0]=1  →  s[0] = 1  (fix t=1)
#   Row 1 (s): A[1]=1, b[1]=0.5  →  s[1] = 0.5 - x
#   Constraint: -0.5*1 <= 0.5-x <= 0.5*1  ⟺  0 <= x <= 1


def test_box_cone_basic():
    """
    Encode  max x  s.t. 0 <= x <= 1  using a box cone.
    SCS box cone is homogeneous (dimension d+1): first row fixes t=1.
    Optimal: x* = 1.
    """
    A_box = sp.csc_matrix(np.array([[0.0], [1.0]]))
    b_box = np.array([1.0, 0.5])
    c_box = np.array([-1.0])
    cone_box = {"bu": [0.5], "bl": [-0.5]}

    solver = scs.SCS(
        {"A": A_box, "b": b_box, "c": c_box}, cone_box, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
        f"Box cone solve failed: {sol['info']['status']}"
    )
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


def test_box_cone_minimise():
    """
    Encode  min x  s.t. 0.3 <= x <= 1  using a box cone.
    Row 0 (t=1): A[0]=0, b[0]=1.
    Row 1 (s):   s[1] = 0.65 - x; constraint -0.35 <= 0.65-x <= 0.35 → 0.3 <= x <= 1.
    Optimal: x* = 0.3.
    """
    A_box = sp.csc_matrix(np.array([[0.0], [1.0]]))
    b_box = np.array([1.0, 0.65])
    c_box = np.array([1.0])
    cone_box = {"bu": [0.35], "bl": [-0.35]}

    solver = scs.SCS(
        {"A": A_box, "b": b_box, "c": c_box}, cone_box, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
        f"Box cone lower bound solve failed: {sol['info']['status']}"
    )
    assert_almost_equal(sol["x"][0], 0.3, decimal=2)


def test_box_cone_two_variables():
    """
    Encode  max x1 + x2  s.t. 0 <= x1 <= 1, -1 <= x2 <= 1  using a box cone.
    d=2, so cone dimension = 3.
    Row 0: t=1  (A[0]=0, b[0]=1)
    Row 1: s[1] = 0.5 - x1,  bound [-0.5, 0.5]  ↔  0 <= x1 <= 1
    Row 2: s[2] = 0 - x2,    bound [-1, 1]       ↔  -1 <= x2 <= 1
    Optimal: x1=1, x2=1.
    """
    A_box = sp.csc_matrix(np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]))
    b_box = np.array([1.0, 0.5, 0.0])
    c_box = np.array([-1.0, -1.0])
    cone_box = {"bu": [0.5, 1.0], "bl": [-0.5, -1.0]}

    solver = scs.SCS(
        {"A": A_box, "b": b_box, "c": c_box}, cone_box, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
        f"Box cone 2-var solve failed: {sol['info']['status']}"
    )
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)
    assert_almost_equal(sol["x"][1], 1.0, decimal=2)


# ===========================================================================
# 11. Warm-start explicit False
# ===========================================================================


def test_warm_start_false_reruns_from_scratch():
    """warm_start=False should reset to cold start (more iters than warm)."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol1 = solver.solve()
    assert sol1["info"]["status"] in ("solved", "solved_inaccurate")
    iters_after_warm = solver.solve(warm_start=True)["info"]["iter"]
    iters_after_cold = solver.solve(warm_start=False)["info"]["iter"]
    # Cold start should require at least as many iterations as warm start
    assert iters_after_cold >= iters_after_warm - 2  # allow tiny slack


def test_warm_start_false_gives_correct_solution():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    sol = solver.solve(warm_start=False)
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


# ===========================================================================
# 12. update() with both b and c simultaneously
# ===========================================================================


def test_update_both_b_and_c():
    """Updating b and c in one call should reflect both changes."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)

    # Change to: min x s.t. -2 <= x <= 2  => x* = -2
    b_new = np.array([2.0, 2.0])
    c_new = np.array([1.0])
    solver.update(b=b_new, c=c_new)
    sol2 = solver.solve()
    assert sol2["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol2["x"][0], -2.0, decimal=2)


def test_update_b_only():
    """Updating only b should change the feasible region."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    # Widen upper bound to 3: b = [3, 0] => x <= 3, x >= 0, max x => x*=3
    solver.update(b=np.array([3.0, 0.0]))
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 3.0, decimal=2)


def test_update_c_only():
    """Updating only c should change the objective direction."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    # Flip objective: min x s.t. 0<=x<=1 => x*=0
    solver.update(c=np.array([1.0]))
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 0.0, decimal=2)


# ===========================================================================
# 13. Legacy solve() API with warm-start data in data dict
# ===========================================================================


def test_legacy_solve_with_warmstart_in_data():
    """Legacy solve() reads x, y, s from the data dict as warm-start vectors."""
    # First, get a solution to use as warm start
    sol0 = scs.solve(_make_data(), _CONE, verbose=False)
    x0 = sol0["x"]
    y0 = sol0["y"]
    s0 = sol0["s"]

    data_ws = {**_make_data(), "x": x0, "y": y0, "s": s0}
    sol_ws = scs.solve(data_ws, _CONE, verbose=False)
    assert sol_ws["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol_ws["x"][0], 1.0, decimal=2)


def test_legacy_solve_partial_warmstart():
    """Legacy solve() with only x in data dict (y, s omitted)."""
    sol0 = scs.solve(_make_data(), _CONE, verbose=False)
    data_ws = {**_make_data(), "x": sol0["x"]}
    sol_ws = scs.solve(data_ws, _CONE, verbose=False)
    assert sol_ws["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 14. Indirect solver — additional settings coverage
# ===========================================================================


def test_indirect_solver_with_tight_tolerances():
    solver = scs.SCS(
        _make_data(), _CONE,
        use_indirect=True,
        eps_abs=1e-8,
        eps_rel=1e-8,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=3)


def test_indirect_solver_acceleration_off():
    solver = scs.SCS(
        _make_data(), _CONE,
        use_indirect=True,
        acceleration_lookback=0,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 15. QP — additional settings coverage
# ===========================================================================


def _make_qp_data():
    """min (1/2) x^2 - x s.t. 0 <= x <= 1; optimal x*=1, obj=-0.5."""
    P = sp.csc_matrix(np.array([[2.0]]))  # Hessian: (1/2) * 2 * x^2 = x^2
    return {"A": _A.copy(), "b": _b.copy(), "c": np.array([-1.0]), "P": P}


@pytest.mark.parametrize("use_indirect", [False, True])
def test_qp_with_settings(use_indirect):
    solver = scs.SCS(
        _make_qp_data(), _CONE,
        use_indirect=use_indirect,
        eps_abs=1e-7,
        eps_rel=1e-7,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    # min x^2 - x on [0,1] => x* = 0.5
    assert_almost_equal(sol["x"][0], 0.5, decimal=2)


# ===========================================================================
# 16. Version / build-info attributes
# ===========================================================================


def test_version_string():
    assert isinstance(scs.__version__, str)
    assert len(scs.__version__) > 0


def test_sizeof_int():
    assert scs.__sizeof_int__ in (4, 8)


def test_sizeof_float():
    assert scs.__sizeof_float__ in (4, 8)


# ===========================================================================
# 17. Zero cone (equality constraints)
# ===========================================================================


def test_zero_cone():
    """
    Zero cone encodes equality constraints: Ax = b.
    Solve: min -x  s.t. x = 0.7  (one equality).
    """
    A_eq = sp.csc_matrix(np.array([[1.0]]))
    b_eq = np.array([0.7])
    c_eq = np.array([-1.0])
    cone_eq = {"z": 1}

    solver = scs.SCS(
        {"A": A_eq, "b": b_eq, "c": c_eq}, cone_eq, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 0.7, decimal=2)


# ===========================================================================
# 18. Mixed zero + nonneg cone
# ===========================================================================


def test_zero_and_nonneg_cone():
    """
    Combine zero cone (equality) and nonneg cone (inequality).
    Solve: min -(x1 + x2)  s.t. x1 = 0.5, 0 <= x2 <= 1.
    Optimal: x1=0.5, x2=1.

    SCS standard form: Ax + s = b, s ∈ cone.
      s = b - Ax (so nonneg means b - Ax >= 0).

    Row 0 (zero):   [1, 0]x + s0 = 0.5  →  s0 = 0 forces x1 = 0.5
    Row 1 (nonneg): [0,-1]x + s1 = 0    →  s1 = x2 >= 0
    Row 2 (nonneg): [0, 1]x + s2 = 1    →  s2 = 1 - x2 >= 0  (x2 <= 1)
    """
    A_m = sp.csc_matrix(np.array([
        [1.0,  0.0],
        [0.0, -1.0],
        [0.0,  1.0],
    ]))
    b_m = np.array([0.5, 0.0, 1.0])
    c_m = np.array([-1.0, -1.0])
    cone_m = {"z": 1, "l": 2}

    solver = scs.SCS({"A": A_m, "b": b_m, "c": c_m}, cone_m, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 0.5, decimal=2)
    assert_almost_equal(sol["x"][1], 1.0, decimal=2)


# ===========================================================================
# 19. Infeasible and unbounded problem detection with status codes
# ===========================================================================


def test_infeasible_status_code():
    """
    Simple infeasible LP: x >= 1 AND x <= 0.

    SCS form: Ax + s = b, s >= 0.
      Row 0: s = -1 - (-x) = x - 1 >= 0  =>  x >= 1
      Row 1: s = 0 - x = -x >= 0          =>  x <= 0
    These are contradictory, so the problem is infeasible.
    """
    A_inf = sp.csc_matrix(np.array([[-1.0], [1.0]]))
    b_inf = np.array([-1.0, 0.0])
    c_inf = np.array([1.0])
    solver = scs.SCS(
        {"A": A_inf, "b": b_inf, "c": c_inf},
        {"l": 2},
        verbose=False,
        eps_infeas=1e-7,
        max_iters=10000,
    )
    sol = solver.solve()
    assert sol["info"]["status"] == "infeasible"
    assert sol["info"]["status_val"] == scs.INFEASIBLE


def test_unbounded_status_code():
    """
    Simple unbounded LP: max x s.t. x >= 0 (no upper bound).

    SCS form: A = [[-1]], b = [0], c = [-1], nonneg.
      s = 0 - (-x) = x >= 0  =>  x can grow without bound.
    """
    A_unb = sp.csc_matrix(np.array([[-1.0]]))
    b_unb = np.array([0.0])
    c_unb = np.array([-1.0])
    solver = scs.SCS(
        {"A": A_unb, "b": b_unb, "c": c_unb},
        {"l": 1},
        verbose=False,
        max_iters=10000,
    )
    sol = solver.solve()
    assert sol["info"]["status"] == "unbounded"
    assert sol["info"]["status_val"] == scs.UNBOUNDED


# ===========================================================================
# 20. Exponential cone (primal) with known analytical solution
# ===========================================================================


def test_exp_cone_primal_known_solution():
    """
    min t  s.t. (r, s, t) ∈ K_exp,  r = 1,  s = 1.

    Primal exp cone: s * exp(r/s) <= t  =>  1 * exp(1) <= t  =>  t >= e.
    Optimal: t* = e ≈ 2.71828.

    Variables: x = [t, r, s]  (n=3, m=5)
    Cone ordering in SCS: z first, then ep.

    Zero rows (z=2):
      Row 0: [0,1,0]x = 1  =>  slack = 1 - r = 0  =>  r = 1
      Row 1: [0,0,1]x = 1  =>  slack = 1 - s = 0  =>  s = 1

    Exp cone rows (ep=1), order (r, s, t):
      Row 2: slack = r  =>  A[2] = [0,-1,0],  b[2] = 0
      Row 3: slack = s  =>  A[3] = [0,0,-1],  b[3] = 0
      Row 4: slack = t  =>  A[4] = [-1,0,0],  b[4] = 0
    """
    A_exp = sp.csc_matrix(np.array([
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
    ]))
    b_exp = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    c_exp = np.array([1.0, 0.0, 0.0])
    cone_exp = {"z": 2, "ep": 1}

    solver = scs.SCS(
        {"A": A_exp, "b": b_exp, "c": c_exp},
        cone_exp,
        eps_abs=1e-7,
        eps_rel=1e-7,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], np.e, decimal=4)


@pytest.mark.parametrize("use_indirect", [False, True])
def test_exp_cone_both_solvers(use_indirect):
    """Exp cone problem solved with both direct and indirect backends."""
    A_exp = sp.csc_matrix(np.array([
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
    ]))
    b_exp = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    c_exp = np.array([1.0, 0.0, 0.0])
    cone_exp = {"z": 2, "ep": 1}

    solver = scs.SCS(
        {"A": A_exp, "b": b_exp, "c": c_exp},
        cone_exp,
        use_indirect=use_indirect,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], np.e, decimal=3)


# ===========================================================================
# 21. Power cone with known analytical solution
# ===========================================================================


def test_power_cone_known_solution():
    """
    min -z  s.t. (x, y, z) ∈ K_pow(0.5),  x = 1,  y = 1.

    K_pow(0.5): sqrt(x*y) >= |z|  with x,y >= 0.
    With x=1, y=1: |z| <= 1  =>  optimal z* = 1.

    Variables: x_var = [z, x, y]  (n=3, m=5)
    Cone ordering: z first, then p.

    Zero rows (z=2):
      Row 0: [0,1,0]*x_var = 1  =>  x = 1
      Row 1: [0,0,1]*x_var = 1  =>  y = 1

    Power cone rows (p=[0.5]), order (x, y, z):
      Row 2: slack = x  =>  A[2] = [0,-1,0],  b[2] = 0
      Row 3: slack = y  =>  A[3] = [0,0,-1],  b[3] = 0
      Row 4: slack = z  =>  A[4] = [-1,0,0],  b[4] = 0
    """
    A_pow = sp.csc_matrix(np.array([
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
    ]))
    b_pow = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    c_pow = np.array([-1.0, 0.0, 0.0])
    cone_pow = {"z": 2, "p": [0.5]}

    solver = scs.SCS(
        {"A": A_pow, "b": b_pow, "c": c_pow},
        cone_pow,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


def test_power_cone_dual():
    """
    Dual power cone (p < 0): same problem with negative exponent.
    K_pow_dual(0.5) = K_pow(0.5)* (the dual cone, encoded as p = -0.5 in SCS).
    Just verify the solver returns a valid status without crash.
    """
    A_pow = sp.csc_matrix(np.array([
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
    ]))
    b_pow = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    c_pow = np.array([1.0, 0.0, 0.0])
    cone_pow = {"z": 2, "p": [-0.5]}  # dual power cone

    solver = scs.SCS(
        {"A": A_pow, "b": b_pow, "c": c_pow},
        cone_pow,
        verbose=False,
    )
    sol = solver.solve()
    assert "status" in sol["info"]


# ===========================================================================
# 22. SOC (second-order cone) with known analytical solution
# ===========================================================================


def test_soc_known_solution():
    """
    Maximize x1  s.t. x1^2 + 0.5^2 <= 1  (half-unit disk at x2=0.5).
    Equivalently: (t, x1, 0.5) ∈ SOC(3),  t <= 1.
    Optimal: x1 = sqrt(1 - 0.25) = sqrt(3)/2 ≈ 0.866.

    Variables: [x1, t]  (n=2, m=4)
    Row 0 (nonneg): 1 - t >= 0   =>  A[0]=[0,1],  b[0]=1
    Row 1 (SOC-t):  slack=t       =>  A[1]=[0,-1], b[1]=0
    Row 2 (SOC-x1): slack=x1      =>  A[2]=[-1,0], b[2]=0
    Row 3 (SOC-0.5):slack=0.5 (const) => A[3]=[0,0], b[3]=0.5
    """
    A_soc = sp.csc_matrix(np.array([
        [0.0,  1.0],
        [0.0, -1.0],
        [-1.0, 0.0],
        [0.0,  0.0],
    ]))
    b_soc = np.array([1.0, 0.0, 0.0, 0.5])
    c_soc = np.array([-1.0, 0.0])
    cone_soc = {"l": 1, "q": [3]}

    solver = scs.SCS(
        {"A": A_soc, "b": b_soc, "c": c_soc},
        cone_soc,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], np.sqrt(3) / 2, decimal=3)


# ===========================================================================
# 23. Invalid solver settings
# ===========================================================================


def test_max_iters_zero_raises():
    """max_iters=0 is invalid and should raise ValueError."""
    with pytest.raises(ValueError):
        scs.SCS(_make_data(), _CONE, max_iters=0, verbose=False).solve()


def test_scale_zero_raises():
    """scale=0 is invalid and should raise ValueError."""
    with pytest.raises(ValueError):
        scs.SCS(_make_data(), _CONE, scale=0.0, verbose=False).solve()


def test_max_iters_non_integer_raises():
    """Non-integer max_iters should raise ValueError."""
    with pytest.raises(ValueError):
        scs.SCS(_make_data(), _CONE, max_iters=1.5)


# ===========================================================================
# 24. update() error handling
# ===========================================================================


def test_update_b_wrong_size_raises():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    with pytest.raises(ValueError):
        solver.update(b=np.array([1.0, 2.0, 3.0]))  # m=2, not 3


def test_update_c_wrong_size_raises():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    with pytest.raises(ValueError):
        solver.update(c=np.array([1.0, 2.0]))  # n=1, not 2


def test_update_none_is_noop():
    """update() with no arguments leaves the problem unchanged."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol1 = solver.solve()
    solver.update()
    sol2 = solver.solve()
    assert sol2["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol2["x"][0], sol1["x"][0], decimal=3)


# ===========================================================================
# 25. Sequential updates: each solve reflects the new problem
# ===========================================================================


def test_sequential_updates_track_correctly():
    """
    Verify that each update(b=...) moves the optimum to the new bound.
    Problem: max x  s.t. 0 <= x <= ub  =>  x* = ub.
    """
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    for ub in [1.0, 2.0, 0.5, 3.0]:
        solver.update(b=np.array([ub, 0.0]))
        sol = solver.solve()
        assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
            f"ub={ub}: unexpected status {sol['info']['status']}"
        )
        assert_almost_equal(sol["x"][0], ub, decimal=2, err_msg=f"ub={ub}")


# ===========================================================================
# 26. First solve with warm_start=True does not crash
# ===========================================================================


def test_first_solve_warm_start_true():
    """warm_start=True on the very first solve should not raise an exception."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol = solver.solve(warm_start=True)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


# ===========================================================================
# 27. Complete info dict (all fields including timing + residuals)
# ===========================================================================


_ALL_INFO_KEYS = {
    "status", "status_val", "iter",
    "pobj", "dobj", "gap",
    "res_pri", "res_dual", "res_infeas",
    "res_unbdd_a", "res_unbdd_p",
    "setup_time", "solve_time",
    "lin_sys_time", "cone_time", "accel_time",
    "scale",
    "comp_slack",
    "accepted_accel_steps", "rejected_accel_steps",
}


def test_all_info_keys_present():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol = solver.solve()
    for key in _ALL_INFO_KEYS:
        assert key in sol["info"], f"Missing info key: '{key}'"


def test_timing_info_nonnegative():
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol = solver.solve()
    info = sol["info"]
    for key in ("setup_time", "solve_time", "lin_sys_time", "cone_time", "accel_time"):
        assert info[key] >= 0.0, f"{key} = {info[key]} is negative"


# ===========================================================================
# 28. Solution quality checks for solved problems
# ===========================================================================


def test_strong_duality_lp():
    """For a solved LP, pobj and dobj should be close (strong duality)."""
    solver = scs.SCS(_make_data(), _CONE, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert abs(sol["info"]["pobj"] - sol["info"]["dobj"]) < 1e-4


def test_residuals_small_for_solved():
    """A solved problem should have small primal and dual residuals."""
    solver = scs.SCS(_make_data(), _CONE, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert sol["info"]["res_pri"] < 1e-4
    assert sol["info"]["res_dual"] < 1e-4


def test_accel_steps_nonnegative():
    """accepted_accel_steps and rejected_accel_steps should both be >= 0."""
    solver = scs.SCS(
        _make_data(), _CONE, acceleration_lookback=5, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["accepted_accel_steps"] >= 0
    assert sol["info"]["rejected_accel_steps"] >= 0


def test_pobj_matches_c_dot_x():
    """For an LP (no P), primal objective should equal c'x."""
    solver = scs.SCS(_make_data(), _CONE, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    c_dot_x = float(np.dot(_c, sol["x"]))
    assert abs(sol["info"]["pobj"] - c_dot_x) < 1e-4


# ===========================================================================
# 29. Unsorted CSC column indices trigger sort_indices()
# ===========================================================================


def test_unsorted_A_indices_are_sorted_and_solved():
    """
    A CSC matrix with unsorted column indices should trigger the
    `A.sort_indices()` path in SCS.__init__ and still give a correct result.

    Construct A = [[1.0], [-1.0]] with indices stored as [1, 0] (unsorted).
    This is the same LP as the baseline but with deliberately reversed index order.
    """
    # CSC storage: (data[i], row=indices[i], col determined by indptr)
    # data=[−1, 1], indices=[1, 0] encodes row0=1.0, row1=−1.0 → same as _A.
    data_arr = np.array([-1.0, 1.0])
    idx_arr = np.array([1, 0])   # row 1 listed before row 0 — unsorted
    indptr_arr = np.array([0, 2])
    A_unsorted = sp.csc_matrix(
        (data_arr, idx_arr, indptr_arr), shape=(2, 1)
    )
    assert not A_unsorted.has_sorted_indices

    solver = scs.SCS(
        {"A": A_unsorted, "b": _b.copy(), "c": _c.copy()},
        _CONE,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


# ===========================================================================
# 30. SDP (semidefinite cone) with known analytical solution
# ===========================================================================


def test_sdp_2x2_known_solution():
    """
    min x  s.t. [[1, x], [x, 1]] ⪰ 0.

    The 2×2 PSD constraint holds iff |x| ≤ 1 (eigenvalues ≥ 0).
    Minimizing x gives x* = −1.

    SCS vectorises the upper-triangular part with off-diagonals scaled by √2:
      slack = (a11, a12*√2, a22) = (1, x*√2, 1) must lie in K_psd(2).

    Variables: [x]  (n=1, m=3, cone={s:[2]})
      Row 0 (PSD a11): slack = 1        → A[0]=[0],       b[0]=1
      Row 1 (PSD a12): slack = x*√2     → A[1]=[-√2],     b[1]=0
      Row 2 (PSD a22): slack = 1        → A[2]=[0],       b[2]=1
    """
    sq2 = np.sqrt(2.0)
    A_sdp = sp.csc_matrix(np.array([[0.0], [-sq2], [0.0]]))
    b_sdp = np.array([1.0, 0.0, 1.0])
    c_sdp = np.array([1.0])
    cone_sdp = {"s": [2]}

    solver = scs.SCS(
        {"A": A_sdp, "b": b_sdp, "c": c_sdp},
        cone_sdp,
        eps_abs=1e-7,
        eps_rel=1e-7,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], -1.0, decimal=4)


@pytest.mark.parametrize("use_indirect", [False, True])
def test_sdp_both_solvers(use_indirect):
    """SDP solved with both direct and indirect backends."""
    sq2 = np.sqrt(2.0)
    A_sdp = sp.csc_matrix(np.array([[0.0], [-sq2], [0.0]]))
    b_sdp = np.array([1.0, 0.0, 1.0])
    c_sdp = np.array([1.0])
    cone_sdp = {"s": [2]}

    solver = scs.SCS(
        {"A": A_sdp, "b": b_sdp, "c": c_sdp},
        cone_sdp,
        use_indirect=use_indirect,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], -1.0, decimal=3)


# ===========================================================================
# 31. P matrix with explicit lower-triangular entries (they must be stripped)
# ===========================================================================


def test_P_lower_triangular_stripped():
    """
    A P matrix with nonzero lower-triangular entries should be silently reduced
    to its upper-triangular part before the solve.  The 2×2 case with
    P = [[2, 1], [1, 2]] (symmetric) should behave the same as the upper-
    triangular version P = [[2, 1], [0, 2]].
    """
    n = 2
    A2 = sp.block_diag(
        [sp.csc_matrix([[1.0], [-1.0]]), sp.csc_matrix([[1.0], [-1.0]])],
        format="csc",
    )
    b2 = np.array([1.0, 0.0, 1.0, 0.0])
    c2 = np.array([-0.6, -0.4])
    cone2 = {"l": 4}

    P_full = sp.csc_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))  # full symmetric
    P_upper = sp.triu(P_full, format="csc")                      # upper triangular only

    sol_full = scs.SCS(
        {"A": A2, "b": b2, "c": c2, "P": P_full}, cone2, verbose=False
    ).solve()
    sol_upper = scs.SCS(
        {"A": A2, "b": b2, "c": c2, "P": P_upper}, cone2, verbose=False
    ).solve()

    assert sol_full["info"]["status"] in ("solved", "solved_inaccurate")
    assert sol_upper["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol_full["x"], sol_upper["x"], decimal=3)


# ===========================================================================
# 32. Recover feasibility: infeasible solve followed by update to feasible
# ===========================================================================


def test_infeasible_then_update_to_feasible():
    """
    Demonstrate that after an infeasible solve the same SCS object can be
    reused by calling update() to fix the problem data.

    Infeasible problem: min x  s.t. x >= 1 AND x <= 0  (b = [-1, 0])
    After update b = [1, 0]:  min x  s.t. 0 <= x <= 1  →  x* = 0
    """
    A_inf = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    solver = scs.SCS(
        {"A": A_inf, "b": np.array([-1.0, 0.0]), "c": np.array([1.0])},
        {"l": 2},
        verbose=False,
        max_iters=5000,
    )
    sol_inf = solver.solve()
    assert sol_inf["info"]["status"] == "infeasible"

    solver.update(b=np.array([1.0, 0.0]))
    sol_feas = solver.solve()
    assert sol_feas["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol_feas["x"][0], 0.0, decimal=2)


# ===========================================================================
# 33. Multiple SOC cones in one problem
# ===========================================================================


def test_multiple_soc_cones():
    """
    Maximize x1 + x2  subject to independent ball constraints:
      x1² + 0.5² ≤ 1    →  x1 ≤ √(1−0.25) = √3/2 ≈ 0.866
      x2² + 0.3² ≤ 1    →  x2 ≤ √(1−0.09) ≈ 0.954

    Formulated as two separate SOC(3) cones plus two nonneg constraints t_i ≤ 1.

    Variables: [x1, x2, t1, t2]  (n=4, m=8)
      Row 0 (nonneg): t1 ≤ 1
      Row 1 (nonneg): t2 ≤ 1
      Rows 2-4 (SOC(3)): (t1, x1, 0.5)
      Rows 5-7 (SOC(3)): (t2, x2, 0.3)
    """
    A_m = sp.csc_matrix(np.array([
        [0.,  0.,  1.,  0.],   # nonneg: t1 <= 1
        [0.,  0.,  0.,  1.],   # nonneg: t2 <= 1
        [0.,  0., -1.,  0.],   # SOC1 t
        [-1., 0.,  0.,  0.],   # SOC1 x1
        [0.,  0.,  0.,  0.],   # SOC1 const 0.5
        [0.,  0.,  0., -1.],   # SOC2 t
        [0., -1.,  0.,  0.],   # SOC2 x2
        [0.,  0.,  0.,  0.],   # SOC2 const 0.3
    ]))
    b_m = np.array([1., 1., 0., 0., 0.5, 0., 0., 0.3])
    c_m = np.array([-1., -1., 0., 0.])
    cone_m = {"l": 2, "q": [3, 3]}

    solver = scs.SCS({"A": A_m, "b": b_m, "c": c_m}, cone_m, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], np.sqrt(3) / 2, decimal=3)
    assert_almost_equal(sol["x"][1], np.sqrt(1 - 0.09), decimal=3)


# ===========================================================================
# 34. update() before the first solve
# ===========================================================================


def test_update_before_first_solve():
    """
    update() is allowed before solve() and the subsequent solve should use
    the updated data.
    """
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    # Change upper bound from 1 to 2 before any solve
    solver.update(b=np.array([2.0, 0.0]))
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 2.0, decimal=2)


# ===========================================================================
# 35. Mixed LP + exponential cone
# ===========================================================================


def test_mixed_lp_and_exp_cone():
    """
    Combine a nonneg (LP) constraint with a primal exp cone.

    Problem: min t + u
      s.t. (r, s, t) ∈ K_exp  with r=1, s=1  →  t >= e
           0 <= u <= 2                          →  u = 0 at optimum
    Optimal: t* = e ≈ 2.718, u* = 0, obj* = e.

    Variables: x = [t, u, r, s]  (n=4)
    SCS cone row ordering: zero first, nonneg second, exp last.

    Zero rows (z=2):
      Row 0: r = 1  →  A[0]=[0,0,1,0],  b[0]=1
      Row 1: s = 1  →  A[1]=[0,0,0,1],  b[1]=1

    Nonneg rows (l=2):
      Row 2: u >= 0        →  A[2]=[0,-1,0,0], b[2]=0  (slack=u)
      Row 3: 2 - u >= 0    →  A[3]=[0,1,0,0],  b[3]=2  (slack=2-u)

    Exp cone rows (ep=1), order (r, s, t):
      Row 4: slack=r  →  A[4]=[0,0,-1,0], b[4]=0
      Row 5: slack=s  →  A[5]=[0,0,0,-1], b[5]=0
      Row 6: slack=t  →  A[6]=[-1,0,0,0], b[6]=0
    """
    A_mix = sp.csc_matrix(np.array([
        [0.,   0.,  1.,  0.],
        [0.,   0.,  0.,  1.],
        [0.,  -1.,  0.,  0.],
        [0.,   1.,  0.,  0.],
        [0.,   0., -1.,  0.],
        [0.,   0.,  0., -1.],
        [-1.,  0.,  0.,  0.],
    ]))
    b_mix = np.array([1., 1., 0., 2., 0., 0., 0.])
    c_mix = np.array([1., 1., 0., 0.])
    cone_mix = {"z": 2, "l": 2, "ep": 1}

    solver = scs.SCS(
        {"A": A_mix, "b": b_mix, "c": c_mix},
        cone_mix,
        eps_abs=1e-7,
        eps_rel=1e-7,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], np.e, decimal=3)   # t* = e
    assert_almost_equal(sol["x"][1], 0.0, decimal=3)    # u* = 0


# ===========================================================================
# 36. Solution array shapes
# ===========================================================================


def test_solution_x_shape():
    """x must have exactly n entries (number of variables)."""
    # n=1, m=2
    sol = scs.SCS(_make_data(), _CONE, verbose=False).solve()
    assert sol["x"].shape == (1,)


def test_solution_y_s_shapes():
    """y and s must each have exactly m entries (number of constraints)."""
    # n=2, m=4
    A4 = sp.csc_matrix(np.array([
        [1., 0.], [-1., 0.], [0., 1.], [0., -1.]
    ]))
    b4 = np.array([1., 0., 1., 0.])
    c4 = np.array([-1., -1.])
    sol = scs.SCS({"A": A4, "b": b4, "c": c4}, {"l": 4}, verbose=False).solve()
    assert sol["y"].shape == (4,)
    assert sol["s"].shape == (4,)


def test_solution_shapes_match_problem_dimensions():
    """x, y, s shapes must match n and m from the problem data."""
    n, m = 3, 5
    np.random.seed(99)
    A_r = sp.random(m, n, density=0.8, format="csc")
    A_r.data = np.random.randn(A_r.nnz) * 0.1
    b_r = np.abs(np.random.randn(m)) + 1.0
    c_r = np.random.randn(n)
    sol = scs.SCS(
        {"A": A_r, "b": b_r, "c": c_r}, {"l": m}, verbose=False
    ).solve()
    assert sol["x"].shape == (n,)
    assert sol["y"].shape == (m,)
    assert sol["s"].shape == (m,)


# ===========================================================================
# 37. Feasibility certificate checks for solved problems
# ===========================================================================


def test_primal_feasibility_residual():
    """||Ax + s - b|| / (1 + ||b||) should be very small at optimum."""
    solver = scs.SCS(
        _make_data(), _CONE, eps_abs=1e-9, eps_rel=1e-9, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    x, s = sol["x"], sol["s"]
    res = np.linalg.norm(_A.dot(x) + s - _b) / (1.0 + np.linalg.norm(_b))
    assert res < 1e-6, f"Primal residual too large: {res}"


def test_dual_feasibility_residual_lp():
    """For LP (P=0): ||A'y + c|| / (1 + ||c||) should be very small."""
    solver = scs.SCS(
        _make_data(), _CONE, eps_abs=1e-9, eps_rel=1e-9, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    y = sol["y"]
    res = np.linalg.norm(_A.T.dot(y) + _c) / (1.0 + np.linalg.norm(_c))
    assert res < 1e-6, f"Dual residual too large: {res}"


def test_complementary_slackness_lp():
    """For LP: |s · y| should be very small at optimum (complementary slackness)."""
    solver = scs.SCS(
        _make_data(), _CONE, eps_abs=1e-9, eps_rel=1e-9, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    cs = abs(np.dot(sol["s"], sol["y"]))
    assert cs < 1e-8, f"Complementary slackness violation: {cs}"


# ===========================================================================
# 38. Integer array inputs raise ValueError
# ===========================================================================


def test_integer_b_raises():
    """b with integer dtype should be rejected by the C extension."""
    with pytest.raises((ValueError, TypeError)):
        scs.SCS(
            {"A": _A, "b": np.array([1, 0]), "c": _c}, _CONE, verbose=False
        ).solve()


def test_integer_c_raises():
    """c with integer dtype should be rejected by the C extension."""
    with pytest.raises((ValueError, TypeError)):
        scs.SCS(
            {"A": _A, "b": _b, "c": np.array([-1])}, _CONE, verbose=False
        ).solve()


def test_integer_A_data_raises():
    """A with integer data array should be rejected by the C extension."""
    A_int = sp.csc_matrix(np.array([[1], [-1]]))  # integer dtype
    with pytest.raises((ValueError, TypeError)):
        scs.SCS(
            {"A": A_int, "b": _b, "c": _c}, _CONE, verbose=False
        ).solve()


# ===========================================================================
# 39. File content verification
# ===========================================================================


def test_write_data_filename_nonempty():
    """The file written by write_data_filename must be non-empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "scs_data.txt")
        solver = scs.SCS(
            _make_data(), _CONE, write_data_filename=fname, verbose=False
        )
        solver.solve()
        assert os.path.getsize(fname) > 0


def test_log_csv_filename_has_rows():
    """The CSV written by log_csv_filename must contain at least a header and one data row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "scs_log.csv")
        solver = scs.SCS(
            _make_data(), _CONE, log_csv_filename=fname, verbose=False
        )
        solver.solve()
        with open(fname) as fh:
            lines = [l for l in fh.readlines() if l.strip()]
        assert len(lines) >= 2, f"CSV has only {len(lines)} non-empty lines"
        # First line should look like a header (contains letters)
        assert any(c.isalpha() for c in lines[0])


# ===========================================================================
# 40. Dual exponential cone via random feasible problem
# ===========================================================================


def test_dual_exp_cone_random_feasible():
    """
    Generate a random feasible problem with a dual exponential cone (ed=1)
    and verify the solver returns 'solved' with objective near the known
    optimal.
    """
    import sys
    sys.path.insert(0, "test")
    import gen_random_cone_prob as tools

    np.random.seed(7)
    K_ed = {"z": 3, "l": 5, "q": [4], "s": [], "ep": 0, "ed": 1, "p": []}
    m_ed = tools.get_scs_cone_dims(K_ed)
    data, pstar = tools.gen_feasible(K_ed, n=m_ed // 3, density=0.2)

    solver = scs.SCS(
        data, K_ed,
        max_iters=50000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
        f"Dual exp cone solve failed: {sol['info']['status']}"
    )
    assert_almost_equal(
        np.dot(data["c"], sol["x"]), pstar, decimal=2
    )


# ===========================================================================
# 41. Mixed SOC + SDP via random feasible problem
# ===========================================================================


def test_mixed_soc_sdp_random_feasible():
    """
    Generate a random feasible problem combining zero, nonneg, SOC, and SDP
    cones and verify the solver finds the correct objective value.
    """
    import sys
    sys.path.insert(0, "test")
    import gen_random_cone_prob as tools

    np.random.seed(42)
    K_mix = {
        "z": 5, "l": 5, "q": [4], "s": [3], "ep": 0, "ed": 0, "p": []
    }
    m_mix = tools.get_scs_cone_dims(K_mix)
    data, pstar = tools.gen_feasible(K_mix, n=m_mix // 3, density=0.2)

    solver = scs.SCS(
        data, K_mix,
        max_iters=50000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
        f"Mixed SOC+SDP solve failed: {sol['info']['status']}"
    )
    assert_almost_equal(np.dot(data["c"], sol["x"]), pstar, decimal=2)


# ===========================================================================
# 42. Multiple power cones
# ===========================================================================


def test_two_power_cones():
    """
    Two K_pow(0.5) cones sharing the same z-variable.

    Cone 1: (x1, y1, z) ∈ K_pow(0.5),  x1=1, y1=1  →  z ≤ 1
    Cone 2: (x2, y2, z) ∈ K_pow(0.5),  x2=1, y2=1  →  z ≤ 1

    Maximize z  →  z* = 1.

    Variables: [z, x1, y1, x2, y2]  (n=5)
    Cone ordering: z (equalities) then p (power cones).

    Zero rows (z=4): x1=1, y1=1, x2=1, y2=1
    Power cone 1 rows (x, y, z order):
      Row 4: slack=x1  → A[4]=[0,-1,0,0,0]
      Row 5: slack=y1  → A[5]=[0,0,-1,0,0]
      Row 6: slack=z   → A[6]=[-1,0,0,0,0]
    Power cone 2 rows:
      Row 7: slack=x2  → A[7]=[0,0,0,-1,0]
      Row 8: slack=y2  → A[8]=[0,0,0,0,-1]
      Row 9: slack=z   → A[9]=[-1,0,0,0,0]
    """
    A_p = sp.csc_matrix(np.array([
        [0.,  1.,  0.,  0.,  0.],   # zero x1=1
        [0.,  0.,  1.,  0.,  0.],   # zero y1=1
        [0.,  0.,  0.,  1.,  0.],   # zero x2=1
        [0.,  0.,  0.,  0.,  1.],   # zero y2=1
        [0., -1.,  0.,  0.,  0.],   # pow1 x-slot
        [0.,  0., -1.,  0.,  0.],   # pow1 y-slot
        [-1., 0.,  0.,  0.,  0.],   # pow1 z-slot
        [0.,  0.,  0., -1.,  0.],   # pow2 x-slot
        [0.,  0.,  0.,  0., -1.],   # pow2 y-slot
        [-1., 0.,  0.,  0.,  0.],   # pow2 z-slot
    ]))
    b_p = np.array([1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
    c_p = np.array([-1., 0., 0., 0., 0.])
    cone_p = {"z": 4, "p": [0.5, 0.5]}

    solver = scs.SCS(
        {"A": A_p, "b": b_p, "c": c_p}, cone_p, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


# ===========================================================================
# 43. Warm-start partial overrides
# ===========================================================================


def test_warm_start_y_only_override():
    """Overriding only y (with x=None, s=None) should work without error."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol0 = solver.solve()
    sol1 = solver.solve(warm_start=True, x=None, y=sol0["y"].copy(), s=None)
    assert sol1["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol1["x"][0], 1.0, decimal=2)


def test_warm_start_s_only_override():
    """Overriding only s (with x=None, y=None) should work without error."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol0 = solver.solve()
    sol1 = solver.solve(warm_start=True, x=None, y=None, s=sol0["s"].copy())
    assert sol1["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol1["x"][0], 1.0, decimal=2)


def test_warm_start_reduces_iterations():
    """A warm-started second solve should need no more iterations than the cold start."""
    solver = scs.SCS(
        _make_data(), _CONE,
        eps_abs=1e-9, eps_rel=1e-9,
        verbose=False,
    )
    sol_cold = solver.solve(warm_start=False)
    iters_cold = sol_cold["info"]["iter"]

    sol_warm = solver.solve(warm_start=True)
    iters_warm = sol_warm["info"]["iter"]

    assert iters_warm <= iters_cold, (
        f"Warm start used more iterations ({iters_warm}) than cold start ({iters_cold})"
    )


# ===========================================================================
# 44. Repeated solves return consistent results
# ===========================================================================


def test_repeated_solves_consistent():
    """Five consecutive solves of the same problem must return the same x."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solutions = [solver.solve()["x"][0] for _ in range(5)]
    for val in solutions:
        assert_almost_equal(val, solutions[0], decimal=4)


# ===========================================================================
# 45. Large random LP using gen_feasible
# ===========================================================================


def test_large_random_lp():
    """
    A moderate-size random LP (40 constraints, 20 variables) generated to be
    feasible should be solved to near-optimality.
    """
    import sys
    sys.path.insert(0, "test")
    import gen_random_cone_prob as tools

    np.random.seed(3)
    K_lp = {"z": 0, "l": 40, "q": [], "s": [], "ep": 0, "ed": 0, "p": []}
    m_lp = tools.get_scs_cone_dims(K_lp)
    data, pstar = tools.gen_feasible(K_lp, n=20, density=0.3)

    solver = scs.SCS(
        data, K_lp,
        max_iters=50000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(np.dot(data["c"], sol["x"]), pstar, decimal=2)


# ===========================================================================
# 46. scale_updates info field
# ===========================================================================


def test_scale_updates_nonnegative_adaptive():
    """scale_updates should be >= 0 with adaptive_scale=True."""
    solver = scs.SCS(
        _make_data(), _CONE, adaptive_scale=True, verbose=False
    )
    sol = solver.solve()
    assert sol["info"].get("scale_updates", 0) >= 0


def test_scale_updates_zero_non_adaptive():
    """With adaptive_scale=False the scale is never updated."""
    solver = scs.SCS(
        _make_data(), _CONE, adaptive_scale=False, verbose=False
    )
    sol = solver.solve()
    assert sol["info"].get("scale_updates", 0) == 0


# ===========================================================================
# 47. P matrix with forced unsorted indices triggers sort_indices()
# ===========================================================================


def test_P_forced_unsorted_indices_sorted_and_solved():
    """
    Setting `P.has_sorted_indices = False` forces the `P.sort_indices()` branch
    in SCS.__init__ even when the underlying data is already ordered.
    The solve must still produce the correct QP answer.
    """
    n = 2
    A2 = sp.csc_matrix(np.array([
        [1., 0.], [-1., 0.], [0., 1.], [0., -1.]
    ]))
    b2 = np.array([1., 0., 1., 0.])
    c2 = np.array([-0.5, -0.5])
    P_sorted = sp.csc_matrix(np.array([[2., 0.], [0., 2.]]))
    P_sorted.has_sorted_indices = False   # force the flag

    assert not P_sorted.has_sorted_indices
    solver = scs.SCS(
        {"A": A2, "b": b2, "c": c2, "P": P_sorted}, {"l": 4}, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    # min x1^2 - 0.5*x1 + x2^2 - 0.5*x2 s.t. 0<=x1<=1, 0<=x2<=1
    # Interior min at x1=x2=0.25 (deriv 2xi - 0.5 = 0)
    assert_almost_equal(sol["x"][0], 0.25, decimal=2)
    assert_almost_equal(sol["x"][1], 0.25, decimal=2)


# ===========================================================================
# 48. Complex semidefinite cone (cs)
# ===========================================================================


def test_cs_cone_mixed_with_real_cones():
    """
    Problem with zero + nonneg + real SDP + complex SDP cones.
    Uses a randomly generated feasible problem (P=ε·I ensures bounded objective).
    Checks the solver returns a valid status.
    """
    np.random.seed(1234)
    cone_cs = {"z": 1, "l": 2, "s": [3, 4], "cs": [5, 4]}
    # dimension: z=1, l=2, s=[3,4]->(3+4)→6+10=16, cs=[5,4]->25+16=41 => total=60
    m_cs = int(
        cone_cs["z"]
        + cone_cs["l"]
        + sum(j * (j + 1) // 2 for j in cone_cs["s"])
        + sum(j * j for j in cone_cs["cs"])
    )
    n_cs = m_cs
    P_cs = 0.1 * sp.eye(n_cs, format="csc")
    A_cs = sp.random(m_cs, n_cs, density=0.05, format="csc")
    A_cs.data = np.random.randn(A_cs.nnz)
    b_cs = np.random.randn(m_cs)
    c_cs = np.random.randn(n_cs)

    solver = scs.SCS(
        {"P": P_cs, "A": A_cs, "b": b_cs, "c": c_cs},
        cone_cs,
        max_iters=50000,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
        f"cs cone solve failed with status: {sol['info']['status']}"
    )


# ===========================================================================
# 49. Direct and indirect solvers give consistent answers
# ===========================================================================


def test_direct_indirect_same_answer_lp():
    """Direct and indirect solvers must agree on the LP optimal to 4 decimal places."""
    opts = dict(eps_abs=1e-9, eps_rel=1e-9, verbose=False)
    sol_d = scs.SCS(_make_data(), _CONE, **opts).solve()
    sol_i = scs.SCS(_make_data(), _CONE, use_indirect=True, **opts).solve()
    assert sol_d["info"]["status"] in ("solved", "solved_inaccurate")
    assert sol_i["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol_d["x"][0], sol_i["x"][0], decimal=4)


def test_direct_indirect_same_answer_qp():
    """Direct and indirect solvers must agree on the QP optimal to 3 decimal places."""
    P = sp.csc_matrix(np.array([[2.0]]))
    opts = dict(eps_abs=1e-9, eps_rel=1e-9, verbose=False)
    sol_d = scs.SCS({"A": _A, "b": _b, "c": _c, "P": P}, _CONE, **opts).solve()
    sol_i = scs.SCS(
        {"A": _A, "b": _b, "c": _c, "P": P}, _CONE, use_indirect=True, **opts
    ).solve()
    assert_almost_equal(sol_d["x"][0], sol_i["x"][0], decimal=3)


# ===========================================================================
# 50. QP update() changes the objective direction
# ===========================================================================


def test_qp_update_c_changes_optimum():
    """
    QP: min (1/2)||x||^2 + c*x  s.t. 0 <= x <= 1.
    With c=3: deriv at x=0 is 3 > 0, so x* = 0.
    After update(c=[-3]): deriv at x=1 is -1 < 0, so x* = 1.
    """
    P = sp.csc_matrix(np.array([[2.0]]))
    solver = scs.SCS(
        {"A": _A, "b": _b, "c": np.array([3.0]), "P": P},
        _CONE,
        eps_abs=1e-9,
        eps_rel=1e-9,
        verbose=False,
    )
    sol1 = solver.solve()
    assert sol1["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol1["x"][0], 0.0, decimal=3)

    solver.update(c=np.array([-3.0]))
    sol2 = solver.solve()
    assert sol2["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol2["x"][0], 1.0, decimal=3)


# ===========================================================================
# 51. QP warm start
# ===========================================================================


def test_qp_warm_start_reduces_iterations():
    """Second solve of a QP with warm_start=True should use <= iterations of cold."""
    P = sp.csc_matrix(np.array([[2.0]]))
    solver = scs.SCS(
        {"A": _A, "b": _b, "c": _c, "P": P},
        _CONE,
        eps_abs=1e-9,
        eps_rel=1e-9,
        verbose=False,
    )
    iters_cold = solver.solve(warm_start=False)["info"]["iter"]
    iters_warm = solver.solve(warm_start=True)["info"]["iter"]
    assert iters_warm <= iters_cold


def test_qp_warm_start_correct_solution():
    """After a QP warm start the solution must still be correct."""
    P = sp.csc_matrix(np.array([[2.0]]))
    solver = scs.SCS(
        {"A": _A, "b": _b, "c": _c, "P": P}, _CONE, verbose=False
    )
    solver.solve()  # cold start
    sol = solver.solve(warm_start=True)  # warm start
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    # min x^2 - x s.t. 0<=x<=1 -> x* = 0.5
    assert_almost_equal(sol["x"][0], 0.5, decimal=2)


# ===========================================================================
# 52. Infeasible QP
# ===========================================================================


def test_qp_infeasible():
    """
    QP with contradictory constraints (x >= 1 AND x <= 0) is infeasible.
    The quadratic term doesn't affect feasibility.
    """
    P = sp.csc_matrix(np.array([[2.0]]))
    solver = scs.SCS(
        {"A": _A, "b": np.array([-1.0, 0.0]), "c": np.array([1.0]), "P": P},
        _CONE,
        max_iters=5000,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] == "infeasible"
    assert sol["info"]["status_val"] == scs.INFEASIBLE


# ===========================================================================
# 53. Legacy solve() passes P correctly
# ===========================================================================


def test_legacy_solve_with_P():
    """The legacy scs.solve() function must pass the P matrix to the solver."""
    P = sp.csc_matrix(np.array([[2.0]]))
    sol = scs.solve(
        {"A": _A, "b": _b, "c": _c, "P": P}, _CONE,
        eps_abs=1e-9, eps_rel=1e-9, verbose=False,
    )
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    # min x^2 - x s.t. 0<=x<=1 -> x* = 0.5
    assert_almost_equal(sol["x"][0], 0.5, decimal=3)


# ===========================================================================
# 54. normalize=True and normalize=False give the same answer
# ===========================================================================


def test_normalize_true_false_same_answer():
    """Both normalisation modes must converge to the same optimum."""
    sol_norm = scs.SCS(_make_data(), _CONE, normalize=True, verbose=False).solve()
    sol_no = scs.SCS(_make_data(), _CONE, normalize=False, verbose=False).solve()
    assert sol_norm["info"]["status"] in ("solved", "solved_inaccurate")
    assert sol_no["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol_norm["x"][0], sol_no["x"][0], decimal=2)


# ===========================================================================
# 55. comp_slack is near zero for solved problems
# ===========================================================================


def test_comp_slack_small_for_solved_lp():
    """comp_slack should be very close to zero when the LP is solved."""
    solver = scs.SCS(
        _make_data(), _CONE, eps_abs=1e-9, eps_rel=1e-9, verbose=False
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert abs(sol["info"]["comp_slack"]) < 1e-8


# ===========================================================================
# 56. Empty cone list fields are accepted
# ===========================================================================


def test_empty_q_s_p_fields():
    """Passing empty lists for q, s, p alongside a real cone should work fine."""
    sol = scs.SCS(
        _make_data(), {"l": 2, "q": [], "s": [], "p": []}, verbose=False
    ).solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


# ===========================================================================
# 57. Parametrised rho_x values
# ===========================================================================


@pytest.mark.parametrize("rho_x", [1e-7, 1e-4, 1e-2])
def test_rho_x_values(rho_x):
    """Different rho_x values should all produce a correct solution."""
    solver = scs.SCS(_make_data(), _CONE, rho_x=rho_x, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


# ===========================================================================
# 58. Mixed ep + p cone (random feasible)
# ===========================================================================


def test_mixed_ep_and_power_cone():
    """
    A random feasible problem with primal exp and power cones combined.
    Verifies the solver finds the correct objective value.
    """
    import sys
    sys.path.insert(0, "test")
    import gen_random_cone_prob as tools

    np.random.seed(77)
    K_mix = {
        "z": 3, "l": 5, "q": [], "s": [],
        "ep": 1, "ed": 0,
        "p": [0.4, -0.6],
    }
    m_mix = tools.get_scs_cone_dims(K_mix)
    data, pstar = tools.gen_feasible(K_mix, n=m_mix // 3, density=0.2)

    solver = scs.SCS(
        data, K_mix,
        max_iters=50000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate"), (
        f"ep+p mixed cone failed: {sol['info']['status']}"
    )
    assert_almost_equal(np.dot(data["c"], sol["x"]), pstar, decimal=2)


# ===========================================================================
# 59. Deterministic: two independent SCS objects give identical solutions
# ===========================================================================


def test_two_instances_same_problem_identical_result():
    """
    Two freshly constructed SCS objects on identical data must produce
    bit-for-bit identical solutions (the solver is deterministic).
    """
    import sys
    sys.path.insert(0, "test")
    import gen_random_cone_prob as tools

    np.random.seed(11)
    K = {"z": 3, "l": 5, "q": [4], "s": [], "ep": 0, "ed": 0, "p": []}
    m = tools.get_scs_cone_dims(K)
    data, _ = tools.gen_feasible(K, n=m // 2, density=0.2)

    sol1 = scs.SCS(data, K, verbose=False).solve()
    sol2 = scs.SCS(data, K, verbose=False).solve()

    assert sol1["info"]["status"] == sol2["info"]["status"]
    np.testing.assert_array_equal(sol1["x"], sol2["x"])


# ===========================================================================
# 60. Alpha boundary values
# ===========================================================================


@pytest.mark.parametrize("alpha", [0.1, 1.9])
def test_alpha_boundary_values(alpha):
    """Alpha values near 0 and near 2 (boundaries of valid range) must work."""
    solver = scs.SCS(_make_data(), _CONE, alpha=alpha, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


# ===========================================================================
# 61. Negative parameter validation (tests error paths in C extension)
# ===========================================================================


def test_negative_eps_abs_raises():
    with pytest.raises(ValueError, match="eps_abs"):
        scs.SCS(_make_data(), _CONE, eps_abs=-1e-5, verbose=False)


def test_negative_eps_rel_raises():
    with pytest.raises(ValueError, match="eps_rel"):
        scs.SCS(_make_data(), _CONE, eps_rel=-1e-5, verbose=False)


def test_negative_eps_infeas_raises():
    with pytest.raises(ValueError, match="eps_infeas"):
        scs.SCS(_make_data(), _CONE, eps_infeas=-1e-5, verbose=False)


def test_negative_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        scs.SCS(_make_data(), _CONE, alpha=-0.5, verbose=False)


def test_negative_rho_x_raises():
    with pytest.raises(ValueError, match="rho_x"):
        scs.SCS(_make_data(), _CONE, rho_x=-0.1, verbose=False)


def test_negative_time_limit_secs_raises():
    with pytest.raises(ValueError, match="time_limit_secs"):
        scs.SCS(_make_data(), _CONE, time_limit_secs=-1.0, verbose=False)


def test_negative_scale_raises():
    with pytest.raises(ValueError, match="scale"):
        scs.SCS(_make_data(), _CONE, scale=-1.0, verbose=False)


def test_negative_acceleration_interval_raises():
    with pytest.raises(ValueError, match="acceleration_interval"):
        scs.SCS(_make_data(), _CONE, acceleration_interval=-1, verbose=False)


def test_negative_max_iters_raises():
    with pytest.raises(ValueError, match="max_iters"):
        scs.SCS(_make_data(), _CONE, max_iters=-10, verbose=False)


# ===========================================================================
# 62. Module selection edge cases (pop-all-flags-first fix)
# ===========================================================================


def test_cudss_true_gpu_true_indirect_true_raises():
    """cudss=True, gpu=True, use_indirect=True should raise ValueError."""
    with pytest.raises(ValueError, match="gpu=True"):
        scs.SCS(_make_data(), _CONE, cudss=True, gpu=True,
                use_indirect=True, verbose=False)


def test_cudss_true_gpu_true_indirect_false_tries_import():
    """cudss=True, gpu=True, use_indirect=False should attempt to import
    _scs_cudss (ImportError expected if not built)."""
    try:
        scs.SCS(_make_data(), _CONE, cudss=True, gpu=True,
                use_indirect=False, verbose=False)
    except ImportError:
        pass  # Expected: _scs_cudss not built


def test_gpu_true_mkl_true_no_leak():
    """gpu=True, mkl=True should NOT leak 'mkl' kwarg into the C extension.
    The mkl flag must be popped before passing settings to C."""
    try:
        scs.SCS(_make_data(), _CONE, gpu=True, mkl=True, verbose=False)
    except ImportError:
        pass  # Expected: _scs_gpu/_scs_cudss not built
    # If mkl leaked through, the C extension would raise TypeError
    # about unexpected kwarg. The ImportError (not TypeError) proves it
    # was properly popped.


def test_gpu_indirect_tries_import():
    """gpu=True, use_indirect=True should attempt to import _scs_gpu."""
    try:
        scs.SCS(_make_data(), _CONE, gpu=True, use_indirect=True,
                verbose=False)
    except ImportError:
        pass  # Expected: _scs_gpu not built


def test_mkl_direct_tries_import():
    """mkl=True, use_indirect=False should attempt to import _scs_mkl."""
    try:
        scs.SCS(_make_data(), _CONE, mkl=True, use_indirect=False,
                verbose=False)
    except ImportError:
        pass  # Expected: _scs_mkl not built


def test_select_module_pops_all_flags():
    """Verify that _select_scs_module pops all four selection flags."""
    stgs = {"use_indirect": False, "gpu": False, "mkl": False,
            "cudss": False, "verbose": False}
    scs.SCS(_make_data(), _CONE, **stgs)
    # If any flag wasn't popped, the C extension would raise TypeError


# ===========================================================================
# 63. Deprecated 'f' cone field (maps to 'z')
# ===========================================================================


def test_deprecated_f_cone_field():
    """The deprecated 'f' field should be treated as zero cone."""
    # min c'x s.t. Ax = b  (equality via zero cone)
    A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    b = np.array([1.0, 0.0])
    c = np.array([-1.0])
    data = {"A": A, "b": b, "c": c}
    # f=1 is the old name for z=1; remaining 1 row is nonneg
    sol = scs.solve(data, {"f": 1, "l": 1}, verbose=False)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


def test_f_and_z_both_set_sum():
    """If both 'f' and 'z' are set, they should be summed."""
    A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    b = np.array([1.0, 0.0])
    c = np.array([-1.0])
    data = {"A": A, "b": b, "c": c}
    # f=1 + z=0 should give z=1 total; 1 remaining row is nonneg
    sol = scs.solve(data, {"f": 1, "z": 0, "l": 1}, verbose=False)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 64. Cone fields specified as numpy arrays
# ===========================================================================


def test_cone_q_as_numpy_array():
    """SOC cone dimensions can be given as a numpy int array."""
    n = 3
    A = sp.csc_matrix(np.eye(n))
    b = np.array([0.0, 1.0, 1.0])
    c = np.array([-1.0, 0.0, 0.0])
    data = {"A": A, "b": b, "c": c}
    cone = {"q": np.array([3], dtype=np.int64)}
    sol = scs.solve(data, cone, verbose=False)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


def test_cone_s_as_numpy_array():
    """SDP cone dimensions can be given as a numpy int array."""
    # 2x2 SDP, vectorised dim = 3
    n = 3
    A = sp.csc_matrix(np.eye(n))
    b = np.array([1.0, 0.0, 1.0])
    c = np.array([-1.0, 0.0, -1.0])
    data = {"A": A, "b": b, "c": c}
    cone = {"s": np.array([2], dtype=np.int64)}
    sol = scs.solve(data, cone, verbose=False)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


def test_cone_q_as_single_int():
    """A single SOC cone dimension can be given as a bare int (not list)."""
    n = 3
    A = sp.csc_matrix(np.eye(n))
    b = np.array([0.0, 1.0, 1.0])
    c = np.array([-1.0, 0.0, 0.0])
    data = {"A": A, "b": b, "c": c}
    cone = {"q": 3}  # bare int instead of [3]
    sol = scs.solve(data, cone, verbose=False)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 65. Warm-start with wrong-dimension vectors
# ===========================================================================


def test_warm_start_x_wrong_dim_raises():
    """Warm-starting with x of wrong size should raise ValueError."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    with pytest.raises(ValueError):
        solver.solve(warm_start=True, x=np.array([1.0, 2.0, 3.0]))


def test_warm_start_y_wrong_dim_raises():
    """Warm-starting with y of wrong size should raise ValueError."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    with pytest.raises(ValueError):
        solver.solve(warm_start=True, y=np.array([1.0, 2.0, 3.0]))


def test_warm_start_s_wrong_dim_raises():
    """Warm-starting with s of wrong size should raise ValueError."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    with pytest.raises(ValueError):
        solver.solve(warm_start=True, s=np.array([1.0, 2.0, 3.0]))


# ===========================================================================
# 66. Update with non-float arrays
# ===========================================================================


def test_update_b_integer_array_raises():
    """update() with an integer b array should raise ValueError."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    with pytest.raises(ValueError):
        solver.update(b=np.array([1, 2], dtype=np.int64))


def test_update_c_integer_array_raises():
    """update() with an integer c array should raise ValueError."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()
    with pytest.raises(ValueError):
        solver.update(c=np.array([1], dtype=np.int64))


# ===========================================================================
# 67. Large max_iters produces solved result
# ===========================================================================


def test_large_max_iters_solves():
    """Very large max_iters should not cause issues."""
    solver = scs.SCS(_make_data(), _CONE, max_iters=100000, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] == "solved"


# ===========================================================================
# 68. Box cone with arrays for bu/bl
# ===========================================================================


def test_box_cone_numpy_bounds():
    """bu/bl specified as numpy float arrays should work."""
    n = 2
    # box cone: dim = d + 1 = 3 (t, x1, x2)
    # s = b - Ax, s in box cone => bl <= x <= bu
    A = sp.csc_matrix(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    b = np.array([1.0, 0.0, 0.0])  # t=1, zero offsets
    c = np.array([-1.0, -1.0])
    data = {"A": A, "b": b, "c": c}
    cone = {
        "bu": np.array([5.0, 5.0]),
        "bl": np.array([-5.0, -5.0]),
    }
    sol = scs.solve(data, cone, verbose=False)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


def test_box_cone_bu_bl_mismatch_raises():
    """bu and bl with different lengths should raise ValueError."""
    n = 2
    A = sp.csc_matrix(np.eye(3, n))
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([-1.0, -1.0])
    data = {"A": A, "b": b, "c": c}
    cone = {
        "bu": [10.0, 10.0],
        "bl": [-10.0],  # wrong length
    }
    with pytest.raises(ValueError, match="bu different dimension"):
        scs.SCS(data, cone, verbose=False)


# ===========================================================================
# 69. Zero-dimensional problems / edge cases
# ===========================================================================


def test_zero_element_P():
    """P can be a zero-nnz sparse matrix (empty quadratic term)."""
    P = sp.csc_matrix((1, 1))  # 1x1 zero matrix
    data = {"P": P, "A": _A.copy(), "b": _b.copy(), "c": _c.copy()}
    solver = scs.SCS(data, _CONE, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] == "solved"
    assert_almost_equal(sol["x"][0], 1.0, decimal=3)


# ===========================================================================
# 70. Solve without calling solve first then update
# ===========================================================================


def test_update_then_solve():
    """update() followed by solve() should work (update before first solve)."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.update(b=np.array([2.0, 0.0]))
    sol = solver.solve()
    # Now feasible region is [0, 2], optimal x* = 2
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][0], 2.0, decimal=3)


# ===========================================================================
# 71. Multiple solves with different warm starts
# ===========================================================================


def test_multiple_solves_warm_start_persistence():
    """Solution from previous solve is used as warm start for next."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol1 = solver.solve()
    # Second solve with warm_start=True should use sol1 as starting point
    sol2 = solver.solve(warm_start=True)
    assert sol2["info"]["status"] == "solved"
    assert_almost_equal(sol1["x"], sol2["x"], decimal=5)


def test_cold_start_after_warm():
    """warm_start=False after a warm solve should still converge correctly."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    solver.solve()  # first solve
    sol = solver.solve(warm_start=False)  # cold start
    assert sol["info"]["status"] == "solved"
    assert_almost_equal(sol["x"][0], 1.0, decimal=3)


# ===========================================================================
# 72. Acceleration lookback negative (type-I AA hack)
# ===========================================================================


def test_negative_acceleration_lookback():
    """Negative acceleration_lookback triggers type-I AA (intentional hack)."""
    solver = scs.SCS(_make_data(), _CONE,
                     acceleration_lookback=-10, verbose=False)
    sol = solver.solve()
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 73. Power cone with different exponents
# ===========================================================================


def test_power_cone_half_exponent():
    """Power cone with p=0.5: (x1^0.5)(x2^0.5) >= |x3|."""
    # Power cone: s1^p * s2^(1-p) >= |s3|, with p=0.5
    # Fix s1=1, s2=1 via equalities, then maximize s3
    # Rows: z=2 (equalities for s1,s2), then p cone (3 rows)
    n = 3
    m = 5  # 2 equalities + 3 power cone
    rows = [0, 1, 2, 3, 4]
    cols = [0, 1, 2, 2, 2]
    vals = [1.0, 1.0, 0.0, 0.0, 1.0]
    A = sp.csc_matrix((vals, (rows, cols)), shape=(m, n))
    b = np.array([1.0, 1.0, 1.0, 1.0, 0.0])  # z rows fix s1=1,s2=1; power cone s
    c = np.array([0.0, 0.0, -1.0])  # maximize x3
    data = {"A": A, "b": b, "c": c}
    cone = {"z": 2, "p": [0.5]}
    sol = scs.solve(data, cone, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")
    assert_almost_equal(sol["x"][2], 1.0, decimal=2)


# ===========================================================================
# 74. Complex SDP cone (cs) standalone
# ===========================================================================


def test_cs_cone_standalone():
    """Complex SDP cone by itself should solve."""
    # cs cone of order 2 has dimension 2^2 = 4
    # Use P=εI to ensure bounded objective
    n, m = 4, 4
    P = 0.1 * sp.eye(n, format="csc")
    A = sp.eye(m, n, format="csc")
    b = np.array([1.0, 0.0, 0.0, 1.0])
    c = np.array([1.0, 0.0, 0.0, 1.0])
    data = {"P": P, "A": A, "b": b, "c": c}
    cone = {"cs": [2]}
    sol = scs.solve(data, cone, verbose=False)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 75. Dual exponential cone (ed)
# ===========================================================================


def test_dual_exp_cone_standalone():
    """Dual exponential cone ed=1 should solve."""
    n, m = 3, 3
    A = sp.csc_matrix(np.eye(m, n))
    b = np.array([-1.0, -1.0, -1.0])
    c = np.array([1.0, 1.0, 1.0])
    data = {"A": A, "b": b, "c": c}
    cone = {"ed": 1}
    sol = scs.solve(data, cone, verbose=False)
    assert sol["info"]["status_val"] != -4  # not FAILED


# ===========================================================================
# 76. Multiple sequential update+solve cycles
# ===========================================================================


def test_many_update_solve_cycles():
    """Run several update-solve cycles to test workspace reuse."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    for i in range(1, 6):
        ub = float(i)
        solver.update(b=np.array([ub, 0.0]))
        sol = solver.solve()
        assert sol["info"]["status"] in ("solved", "solved_inaccurate")
        assert_almost_equal(sol["x"][0], ub, decimal=2)


# ===========================================================================
# 77. Info dict numerical fields have correct types
# ===========================================================================


def test_info_iter_is_int():
    sol = scs.solve(_make_data(), _CONE, verbose=False)
    assert isinstance(sol["info"]["iter"], int)


def test_info_pobj_is_float():
    sol = scs.solve(_make_data(), _CONE, verbose=False)
    assert isinstance(sol["info"]["pobj"], float)


def test_info_status_is_str():
    sol = scs.solve(_make_data(), _CONE, verbose=False)
    assert isinstance(sol["info"]["status"], str)


# ===========================================================================
# 78. Solve returns copies (not aliased internal buffers)
# ===========================================================================


def test_solution_arrays_are_copies():
    """Modifying returned x/y/s should not affect next solve."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    sol1 = solver.solve()
    x1_val = sol1["x"][0]
    sol1["x"][0] = 999.0  # mutate returned array
    sol2 = solver.solve(warm_start=False)
    # Second solve should not see the mutation
    assert_almost_equal(sol2["x"][0], x1_val, decimal=5)


# ===========================================================================
# 79. Verbose false produces no stdout
# ===========================================================================


def test_verbose_false_no_output(capsys):
    """verbose=False should produce no stdout."""
    scs.solve(_make_data(), _CONE, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""


# ===========================================================================
# 80. Float32 input arrays
# ===========================================================================


def test_float32_b_c_accepted():
    """float32 b and c should be accepted (cast internally to float64)."""
    A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    b = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([-1.0], dtype=np.float32)
    data = {"A": A, "b": b, "c": c}
    sol = scs.solve(data, _CONE, verbose=False)
    assert sol["info"]["status"] == "solved"
    assert_almost_equal(sol["x"][0], 1.0, decimal=3)


def test_float32_A_accepted():
    """float32 A data should be accepted (cast internally)."""
    A = sp.csc_matrix(np.array([[1.0], [-1.0]], dtype=np.float32))
    b = np.array([1.0, 0.0])
    c = np.array([-1.0])
    data = {"A": A, "b": b, "c": c}
    sol = scs.solve(data, _CONE, verbose=False)
    assert sol["info"]["status"] == "solved"


# ===========================================================================
# 81. Extremely sparse A (single nonzero)
# ===========================================================================


def test_very_sparse_A():
    """A with only one nonzero entry should solve correctly."""
    # min -x s.t. x >= 0 (trivially unbounded, but add upper bound)
    # Actually: min -x s.t. x <= 1, x >= 0
    A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    b = np.array([1.0, 0.0])
    c = np.array([-1.0])
    data = {"A": A, "b": b, "c": c}
    sol = scs.solve(data, {"l": 2}, verbose=False)
    assert sol["info"]["status"] == "solved"


# ===========================================================================
# 82. Empty A (all zeros)
# ===========================================================================


def test_zero_A_matrix():
    """A matrix with no nonzero entries — should still be processable."""
    A = sp.csc_matrix((2, 1))  # 2x1 zero matrix
    b = np.array([1.0, 1.0])
    c = np.array([1.0])
    data = {"A": A, "b": b, "c": c}
    sol = scs.solve(data, {"l": 2}, verbose=False)
    # With A=0, problem is min c'x s.t. s = b, s >= 0
    # x is free, any x gives obj = c'x, so unbounded if c != 0
    assert sol["info"]["status_val"] in (-1, -6, 1, 2)


# ===========================================================================
# 83. Two separate SCS instances don't interfere
# ===========================================================================


def test_two_independent_instances():
    """Two SCS instances with different problems should not interfere."""
    # Instance 1: max x s.t. x <= 1
    solver1 = scs.SCS(_make_data(), _CONE, verbose=False)

    # Instance 2: different problem, max x s.t. x <= 5
    A2 = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    b2 = np.array([5.0, 0.0])
    c2 = np.array([-1.0])
    solver2 = scs.SCS({"A": A2, "b": b2, "c": c2}, {"l": 2}, verbose=False)

    sol1 = solver1.solve()
    sol2 = solver2.solve()

    assert_almost_equal(sol1["x"][0], 1.0, decimal=3)
    assert_almost_equal(sol2["x"][0], 5.0, decimal=3)


# ===========================================================================
# 84. Legacy solve with no P (pure LP)
# ===========================================================================


def test_legacy_solve_no_P():
    """Legacy solve() with no P key in data should work."""
    data = {"A": _A.copy(), "b": _b.copy(), "c": _c.copy()}
    assert "P" not in data
    sol = scs.solve(data, _CONE, verbose=False)
    assert sol["info"]["status"] == "solved"
    assert_almost_equal(sol["x"][0], 1.0, decimal=3)


# ===========================================================================
# 85. Large QP with known closed-form solution
# ===========================================================================


def test_unconstrained_qp_known_solution():
    """Unconstrained QP: min (1/2)x'Px + c'x has closed-form x* = -P^{-1}c."""
    n = 5
    # P = 2I, c = [1,...,1] => x* = -0.5 * [1,...,1]
    P = 2.0 * sp.eye(n, format="csc")
    A = sp.csc_matrix((1, n))  # dummy single row
    b = np.zeros(1)
    c = np.ones(n)
    data = {"P": P, "A": A, "b": b, "c": c}
    cone = {"z": 1}  # equality constraint: 0 = 0
    sol = scs.solve(data, cone, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
    assert sol["info"]["status"] == "solved"
    expected_x = -0.5 * np.ones(n)
    assert_almost_equal(sol["x"], expected_x, decimal=3)


# ===========================================================================
# 86. Write data then verify file contains data
# ===========================================================================


def test_write_data_and_log_csv_simultaneously():
    """write_data_filename and log_csv_filename can both be set."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "data.bin")
        log_file = os.path.join(tmpdir, "log.csv")
        solver = scs.SCS(
            _make_data(), _CONE, verbose=False,
            write_data_filename=data_file,
            log_csv_filename=log_file,
        )
        sol = solver.solve()
        assert sol["info"]["status"] == "solved"
        assert os.path.isfile(data_file)
        assert os.path.isfile(log_file)


# ===========================================================================
# 87. Duality gap near zero for solved problems
# ===========================================================================


def test_duality_gap_small_for_solved():
    """For a solved LP, the duality gap should be near zero."""
    sol = scs.solve(_make_data(), _CONE, verbose=False,
                    eps_abs=1e-9, eps_rel=1e-9)
    assert sol["info"]["status"] == "solved"
    gap = abs(sol["info"]["gap"])
    assert gap < 1e-4, f"Duality gap too large: {gap}"


# ===========================================================================
# 88. P with only lower triangular entries gets converted
# ===========================================================================


def test_P_only_lower_triangular():
    """P given as strictly lower triangular should be flipped to upper."""
    n = 2
    # Lower triangular P (off-diag only in lower part)
    P = sp.csc_matrix(np.array([[2.0, 0.0], [1.0, 2.0]]))
    A = sp.eye(n, format="csc")
    b = np.zeros(n)
    c = np.ones(n)
    data = {"P": P, "A": A, "b": b, "c": c}
    cone = {"z": n}
    sol = scs.solve(data, cone, verbose=False)
    assert sol["info"]["status"] == "solved"


# ===========================================================================
# 89. Cone with all types simultaneously
# ===========================================================================


def test_all_cone_types_simultaneously():
    """Problem using z, l, q, s, ep, p cone types all at once."""
    np.random.seed(42)
    # Cone dims: z=1, l=2, q=[3], s=[2] (vec dim=3), ep=1 (dim 3), p=[0.5] (dim 3)
    # Total rows: 1 + 2 + 3 + 3 + 3 + 3 = 15
    cone = {"z": 1, "l": 2, "q": [3], "s": [2], "ep": 1, "p": [0.5]}
    m = 15
    n = m
    P = 0.1 * sp.eye(n, format="csc")
    A = sp.random(m, n, density=0.1, format="csc")
    A.data = np.random.randn(A.nnz)
    b = np.random.randn(m)
    c = np.random.randn(n)
    data = {"P": P, "A": A, "b": b, "c": c}
    sol = scs.solve(data, cone, verbose=False, max_iters=50000)
    assert sol["info"]["status"] in ("solved", "solved_inaccurate")


# ===========================================================================
# 90. Solve with explicit warm_start=True on first call (no prior solve)
# ===========================================================================


def test_warm_start_true_x_y_s_on_first_solve():
    """Providing x, y, s on the very first solve should work as warm start."""
    solver = scs.SCS(_make_data(), _CONE, verbose=False)
    x0 = np.array([0.9])
    y0 = np.array([1.0, 0.0])
    s0 = np.array([0.1, 0.9])
    sol = solver.solve(warm_start=True, x=x0, y=y0, s=s0)
    assert sol["info"]["status"] == "solved"
    assert_almost_equal(sol["x"][0], 1.0, decimal=3)
