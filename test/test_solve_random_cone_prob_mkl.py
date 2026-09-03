from __future__ import print_function, division
import sys
import platform
import scs
import numpy as np
from scipy import sparse
import pytest
import gen_random_cone_prob as tools

#############################################
#  Uses scs to solve a random cone problem  #
#############################################

# The MKL backend ships in the x86-64 Linux wheels (MKL linked statically
# into _scs_mkl) and in source builds with link_mkl / mkl_static_prefix.
# Skip on platforms where it is never available, and when the extension is
# absent (musllinux, aarch64, macOS).
if sys.platform == "darwin":
    pytest.skip("MKL is not available on macOS", allow_module_level=True)
if sys.platform == "linux" and platform.machine() != "x86_64":
    pytest.skip("MKL is not available on this architecture", allow_module_level=True)

try:
    from scs import _scs_mkl  # noqa: E402
except ImportError:
    pytest.skip("MKL backend not importable", allow_module_level=True)

# cone:
K = {
    "z": 10,
    "l": 15,
    "q": [5, 10, 0, 1],
    "s": [3, 4, 0, 0, 1, 10],
    "ep": 10,
    "ed": 10,
    "p": [-0.25, 0.5, 0.75, -0.33],
}
m = tools.get_scs_cone_dims(K)
params = {"verbose": True, "eps_abs": 1e-7, "eps_rel": 1e-7, "eps_infeas": 1e-7}


def test_solve_feasible():
    rng = np.random.RandomState(3000)
    data, p_star = tools.gen_feasible(K, n=m // 3, density=0.1, rng=rng)
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.MKL, **params)
    sol = solver.solve()
    assert sol["info"]["lin_sys_solver"] == "sparse-direct-mkl-pardiso"
    x = sol["x"]
    y = sol["y"]
    s = sol["s"]
    np.testing.assert_almost_equal(np.dot(data["c"], x), p_star, decimal=3)
    np.testing.assert_array_less(
        np.linalg.norm(data["A"] @ x - data["b"] + s), 1e-3
    )
    np.testing.assert_array_less(
        np.linalg.norm(data["A"].T @ y + data["c"]), 1e-3
    )
    np.testing.assert_almost_equal(s.T @ y, 0.0)
    np.testing.assert_almost_equal(s, tools.proj_cone(s, K), decimal=4)
    np.testing.assert_almost_equal(y, tools.proj_dual_cone(y, K), decimal=4)


def test_solve_infeasible():
    rng = np.random.RandomState(3001)
    data = tools.gen_infeasible(K, n=m // 2, rng=rng)
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.MKL, **params)
    sol = solver.solve()
    y = sol["y"]
    np.testing.assert_array_less(np.linalg.norm(data["A"].T @ y), 1e-3)
    np.testing.assert_array_less(data["b"].T @ y, -0.1)
    np.testing.assert_almost_equal(y, tools.proj_dual_cone(y, K), decimal=4)


def test_solve_unbounded():
    rng = np.random.RandomState(3002)
    data = tools.gen_unbounded(K, n=m // 2, rng=rng)
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.MKL, **params)
    sol = solver.solve()
    x = sol["x"]
    s = sol["s"]
    np.testing.assert_array_less(np.linalg.norm(data["A"] @ x + s), 1e-3)
    np.testing.assert_array_less(data["c"].T @ x, -0.1)
    np.testing.assert_almost_equal(s, tools.proj_cone(s, K), decimal=4)
