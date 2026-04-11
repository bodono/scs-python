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

# MKL is shipped in manylinux x86_64 and Windows wheels, but not in
# musllinux or macOS or aarch64 wheels. Skip on platforms where MKL
# is never available; on MKL platforms fail hard if the import is missing.
if sys.platform == "darwin":
    pytest.skip("MKL is not available on macOS", allow_module_level=True)
if sys.platform == "linux" and platform.machine() != "x86_64":
    pytest.skip("MKL is not available on this architecture", allow_module_level=True)

try:
    from scs import _scs_mkl  # noqa: E402
except ImportError:
    # musllinux x86_64 ships openblas, not MKL
    pytest.skip("MKL module not installed", allow_module_level=True)

np.random.seed(1)

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
    data, p_star = tools.gen_feasible(K, n=m // 3, density=0.1)
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.MKL, **params)
    sol = solver.solve()
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
    data = tools.gen_infeasible(K, n=m // 2)
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.MKL, **params)
    sol = solver.solve()
    y = sol["y"]
    np.testing.assert_array_less(np.linalg.norm(data["A"].T @ y), 1e-3)
    np.testing.assert_array_less(data["b"].T @ y, -0.1)
    np.testing.assert_almost_equal(y, tools.proj_dual_cone(y, K), decimal=4)


def test_solve_unbounded():
    data = tools.gen_unbounded(K, n=m // 2)
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.MKL, **params)
    sol = solver.solve()
    x = sol["x"]
    s = sol["s"]
    np.testing.assert_array_less(np.linalg.norm(data["A"] @ x + s), 1e-3)
    np.testing.assert_array_less(data["c"].T @ x, -0.1)
    np.testing.assert_almost_equal(s, tools.proj_cone(s, K), decimal=4)
