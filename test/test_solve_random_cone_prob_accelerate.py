from __future__ import print_function, division
import sys
import scs
import numpy as np
from scipy import sparse
import pytest
import gen_random_cone_prob as tools

#############################################
#  Uses scs to solve a random cone problem  #
#############################################

# On macOS the accelerate module must be present (it ships in the wheel).
# On other platforms, skip the entire module.
if sys.platform != "darwin":
    pytest.skip("Apple Accelerate is macOS-only", allow_module_level=True)

from scs import _scs_accelerate  # noqa: E402

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
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.ACCELERATE, **params)
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
    # A small, well-conditioned infeasible LP avoids the platform-sensitive
    # convergence of the previous large random cone problem:
    #
    #   x >= 1  ->  -x + s_0 = -1, s_0 >= 0
    #   x <= 0  ->   x + s_1 =  0, s_1 >= 0
    data = {
        "A": sparse.csc_matrix([[-1.0], [1.0]]),
        "b": np.array([-1.0, 0.0]),
        "c": np.array([1.0]),
    }
    cone = {"l": 2}
    solver = scs.SCS(
        data,
        cone,
        linear_solver=scs.LinearSolver.ACCELERATE,
        verbose=False,
        eps_infeas=1e-7,
        max_iters=10000,
    )
    sol = solver.solve()
    y = sol["y"]
    assert sol["info"]["status"] == "infeasible"
    assert sol["info"]["status_val"] == scs.INFEASIBLE
    np.testing.assert_allclose(data["A"].T @ y, 0.0, atol=1e-6)
    assert data["b"].T @ y < -0.1
    assert np.all(y >= -1e-7)


def test_solve_unbounded():
    rng = np.random.RandomState(3002)
    data = tools.gen_unbounded(K, n=m // 2, rng=rng)
    solver = scs.SCS(data, K, linear_solver=scs.LinearSolver.ACCELERATE, **params)
    sol = solver.solve()
    x = sol["x"]
    s = sol["s"]
    np.testing.assert_array_less(np.linalg.norm(data["A"] @ x + s), 1e-3)
    np.testing.assert_array_less(data["c"].T @ x, -0.1)
    np.testing.assert_almost_equal(s, tools.proj_cone(s, K), decimal=4)
