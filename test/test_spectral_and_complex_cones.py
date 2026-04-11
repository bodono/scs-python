"""Tests for spectral cones (ell1, nuclear norm, logdet, sum-of-largest)
and complex PSD cone support in SCS.

Spectral cones require building with -Duse_spectral_cones=true.
Complex PSD cone requires LAPACK (use_lapack=true, on by default).
"""
import numpy as np
import scipy.sparse as sp
import pytest
import scs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sd_cone_size(n):
    """SDP cone variable size: n*(n+1)/2 (packed upper triangle, scaled)."""
    return n * (n + 1) // 2


def _csd_cone_size(n):
    """Complex SDP cone variable size: n*n."""
    return n * n


def _cone_dims(cone):
    """Compute total number of rows m for a cone specification dict."""
    m = cone.get("z", 0) + cone.get("l", 0)
    for qi in cone.get("q", []):
        m += qi
    for si in cone.get("s", []):
        m += _sd_cone_size(si)
    for ci in cone.get("cs", []):
        m += _csd_cone_size(ci)
    m += cone.get("ep", 0) * 3
    m += cone.get("ed", 0) * 3
    m += len(cone.get("p", [])) * 3
    # spectral cones
    for di in cone.get("d", []):
        m += _sd_cone_size(di) + 2
    nuc_m = cone.get("nuc_m", [])
    nuc_n = cone.get("nuc_n", [])
    for i in range(len(nuc_m)):
        m += nuc_m[i] * nuc_n[i] + 1
    for ei in cone.get("ell1", []):
        m += ei + 1
    sl_n = cone.get("sl_n", [])
    for i in range(len(sl_n)):
        m += _sd_cone_size(sl_n[i]) + 1
    return m


def _gen_feasible_qp(cone, n=None, p_scale=1.0):
    """Generate a random feasible QP over the given cone.

    Uses P = p_scale * I (strong regularisation) and dense-ish A to
    guarantee boundedness and feasibility.
    """
    m = _cone_dims(cone)
    if n is None:
        n = m
    P = p_scale * sp.eye(n, format="csc")
    A = sp.random(m, n, density=0.5, format="csc")
    A.data = np.random.randn(A.nnz)
    c = np.random.randn(n)
    b = A @ np.random.randn(n) + np.abs(np.random.randn(m))
    return dict(P=P, A=A, b=b, c=c)


# ---------------------------------------------------------------------------
# Detect which modules are available
# ---------------------------------------------------------------------------

_spectral_available = False
try:
    _probe_data = dict(
        A=sp.eye(2, format="csc"),
        b=np.ones(2),
        c=np.ones(2),
    )
    _probe_sol = scs.solve(_probe_data, {"ell1": [1]}, verbose=False, max_iters=1)
    _spectral_available = True
except Exception:
    pass

_dense_available = False
try:
    from scs import _scs_dense
    _dense_available = True
except ImportError:
    pass

_solver_configs = [
    {"linear_solver": scs.LinearSolver.AUTO},
    {"linear_solver": scs.LinearSolver.QDLDL},
    {"linear_solver": scs.LinearSolver.INDIRECT},
]
if _dense_available:
    _solver_configs.append({"linear_solver": scs.LinearSolver.DENSE})

skip_no_spectral = pytest.mark.skipif(
    not _spectral_available,
    reason="SCS not built with spectral cone support (-Duse_spectral_cones=true)",
)


def _assert_solved(sol):
    assert sol["info"]["status_val"] in (1, 2), (
        f"Expected solved, got: {sol['info']['status']}"
    )


# ===================================================================
# Complex PSD cone tests (does NOT require spectral cones build flag)
# ===================================================================


class TestComplexPSDCone:

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_complex_psd_feasibility(self, solver_opts):
        np.random.seed(42)
        cone = {"cs": [3]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_complex_psd_multiple(self, solver_opts):
        np.random.seed(123)
        cone = {"cs": [2, 3]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_mixed_real_complex_psd(self, solver_opts):
        np.random.seed(456)
        cone = dict(z=1, l=2, s=[3], cs=[3])
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)


# ===================================================================
# Ell-1 cone tests
# ===================================================================


@skip_no_spectral
class TestEll1Cone:

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_ell1_simple(self, solver_opts):
        np.random.seed(10)
        cone = {"ell1": [4]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_ell1_multiple(self, solver_opts):
        np.random.seed(20)
        cone = {"ell1": [3, 5]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_ell1_mixed_with_standard_cones(self, solver_opts):
        np.random.seed(30)
        cone = dict(z=1, l=2, q=[3], ell1=[4])
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    def test_ell1_norm_bound(self):
        """Verify s in ell1 cone satisfies s[0] >= ||s[1:]||_1."""
        np.random.seed(35)
        cone = {"ell1": [5]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
        if sol["info"]["status_val"] == 1:
            s = sol["s"]
            assert s[0] >= np.sum(np.abs(s[1:])) - 1e-4


# ===================================================================
# Nuclear norm cone tests
# ===================================================================


@skip_no_spectral
class TestNuclearNormCone:

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_nuclear_simple(self, solver_opts):
        np.random.seed(40)
        cone = {"nuc_m": [3], "nuc_n": [2]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_nuclear_multiple(self, solver_opts):
        np.random.seed(50)
        cone = {"nuc_m": [3, 4], "nuc_n": [2, 3]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_nuclear_mixed(self, solver_opts):
        np.random.seed(60)
        cone = dict(l=3, nuc_m=[3], nuc_n=[2])
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False)
        _assert_solved(sol)

    def test_nuclear_mismatched_lengths_raises(self):
        m = 10
        data = dict(A=sp.eye(m, format="csc"), b=np.ones(m), c=np.ones(m))
        cone = {"nuc_m": [3, 4], "nuc_n": [2]}
        with pytest.raises(ValueError, match="nuc_m and nuc_n must have the same length"):
            scs.solve(data, cone, verbose=False)

    def test_nuclear_norm_bound(self):
        """Verify s in nuclear norm cone satisfies ||X||_* <= t."""
        np.random.seed(65)
        rows, cols = 4, 3
        cone = {"nuc_m": [rows], "nuc_n": [cols]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
        if sol["info"]["status_val"] == 1:
            s = sol["s"]
            t_val = s[0]
            # SCS stores the matrix in column-major (Fortran) order
            X = s[1:].reshape(rows, cols, order="F")
            nuc_norm = np.sum(np.linalg.svd(X, compute_uv=False))
            assert t_val >= nuc_norm - 1e-4


# ===================================================================
# Logdet cone tests
# ===================================================================


@skip_no_spectral
class TestLogdetCone:

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_logdet_simple(self, solver_opts):
        np.random.seed(70)
        cone = {"d": [3]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=10000)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_logdet_multiple(self, solver_opts):
        np.random.seed(80)
        cone = {"d": [2, 3]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=10000)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_logdet_mixed(self, solver_opts):
        np.random.seed(90)
        cone = dict(l=2, d=[2])
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=10000)
        _assert_solved(sol)


# ===================================================================
# Sum-of-largest eigenvalues cone tests
# ===================================================================


@skip_no_spectral
class TestSumLargestCone:

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_sum_largest_simple(self, solver_opts):
        np.random.seed(100)
        cone = {"sl_n": [4], "sl_k": [2]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=10000)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_sum_largest_multiple(self, solver_opts):
        np.random.seed(110)
        cone = {"sl_n": [3, 4], "sl_k": [1, 2]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=10000)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_sum_largest_mixed(self, solver_opts):
        np.random.seed(120)
        cone = dict(z=1, l=2, sl_n=[3], sl_k=[1])
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=10000)
        _assert_solved(sol)

    def test_sum_largest_mismatched_raises(self):
        m = 20
        data = dict(A=sp.eye(m, format="csc"), b=np.ones(m), c=np.ones(m))
        cone = {"sl_n": [3, 4], "sl_k": [1]}
        with pytest.raises(ValueError, match="sl_n and sl_k must have the same length"):
            scs.solve(data, cone, verbose=False)


# ===================================================================
# Combined / kitchen-sink tests
# ===================================================================


@skip_no_spectral
class TestAllConesCombined:

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_kitchen_sink(self, solver_opts):
        """Problem with every cone type present."""
        np.random.seed(200)
        cone = dict(
            z=1,
            l=2,
            q=[3],
            s=[2],
            cs=[2],
            ep=1,
            ed=1,
            p=[0.5],
            d=[2],
            nuc_m=[3],
            nuc_n=[2],
            ell1=[3],
            sl_n=[3],
            sl_k=[1],
        )
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=10000)
        _assert_solved(sol)

    @pytest.mark.parametrize("solver_opts", _solver_configs)
    def test_spectral_and_complex_psd(self, solver_opts):
        np.random.seed(210)
        cone = dict(cs=[3], ell1=[5], nuc_m=[4], nuc_n=[3])
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, **solver_opts, verbose=False, max_iters=5000)
        _assert_solved(sol)


# ===================================================================
# Edge cases
# ===================================================================


@skip_no_spectral
class TestEdgeCases:

    def test_ell1_dim_one(self):
        np.random.seed(300)
        cone = {"ell1": [1]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, verbose=False)
        _assert_solved(sol)

    def test_nuclear_square(self):
        np.random.seed(310)
        cone = {"nuc_m": [3], "nuc_n": [3]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, verbose=False)
        _assert_solved(sol)

    def test_logdet_dim_one(self):
        np.random.seed(320)
        cone = {"d": [1]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, verbose=False, max_iters=10000)
        _assert_solved(sol)

    def test_sum_largest_k_one(self):
        np.random.seed(330)
        cone = {"sl_n": [3], "sl_k": [1]}
        data = _gen_feasible_qp(cone)
        sol = scs.solve(data, cone, verbose=False, max_iters=10000)
        _assert_solved(sol)

    def test_warm_start_with_spectral(self):
        np.random.seed(340)
        cone = {"ell1": [4]}
        data = _gen_feasible_qp(cone)
        solver = scs.SCS(data, cone, verbose=False, max_iters=500)
        sol1 = solver.solve()
        sol2 = solver.solve(warm_start=True)
        assert sol2["info"]["status_val"] in (1, 2)
