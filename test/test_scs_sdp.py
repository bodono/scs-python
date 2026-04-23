from __future__ import print_function
import platform

## import utilities to generate random cone probs:
import sys
import gen_random_cone_prob as tools


def import_error(msg):
    print()
    print("## IMPORT ERROR:" + msg)
    print()


try:
    import pytest
except ImportError:
    import_error("Please install pytest to run tests.")
    raise

try:
    import scs
except ImportError:
    import_error("You must install the scs module before running tests.")
    raise

try:
    import numpy as np
    from numpy.testing import assert_almost_equal
except ImportError:
    import_error("Please install numpy.")
    raise

try:
    import scipy.sparse as sp
except ImportError:
    import_error("Please install scipy.")
    raise


def assert_(str1, str2):
    if str1 != str2:
        print("assert failure: %s != %s" % (str1, str2))
    assert str1 == str2


def check_infeasible(sol):
    assert_(sol["info"]["status"], "infeasible")


def check_unbounded(sol):
    assert_(sol["info"]["status"], "unbounded")


num_feas = 50
num_unb = 10
num_infeas = 10

opts = {
    "max_iters": 10000,
    "eps_abs": 1e-5,
    "eps_infeas": 1e-5,
}
K = {
    "z": 10,
    "l": 25,
    "q": [5, 10, 0, 1],
    "s": [2, 1, 2, 0, 1, 10, 8],
    "ep": 0,
    "ed": 0,
    "p": [0.25, -0.75, 0.33, -0.33, 0.2],
}
m = tools.get_scs_cone_dims(K)


_dense_available = False
try:
    from scs import _scs_dense
    _dense_available = True
except ImportError:
    pass

_solver_configs = [
    {"linear_solver": scs.LinearSolver.AUTO},
    {"linear_solver": scs.LinearSolver.QDLDL},
    {"linear_solver": scs.LinearSolver.CPU_INDIRECT},
]
if _dense_available:
    _solver_configs.append({"linear_solver": scs.LinearSolver.CPU_DENSE})


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_feasible(solver_opts):
    rng = np.random.RandomState(2000)
    for i in range(num_feas):
        data, p_star = tools.gen_feasible(K, n=m // 3, density=0.1, rng=rng)
        solver = scs.SCS(data, K, **solver_opts, **opts)
        sol = solver.solve()
        assert_almost_equal(np.dot(data["c"], sol["x"]), p_star, decimal=2)
        assert_almost_equal(np.dot(-data["b"], sol["y"]), p_star, decimal=2)


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_infeasible(solver_opts):
    rng = np.random.RandomState(2001)
    for i in range(num_infeas):
        data = tools.gen_infeasible(K, n=m // 2, rng=rng)
        solver = scs.SCS(data, K, **solver_opts, **opts)
        sol = solver.solve()
        check_infeasible(sol)


# TODO: indirect solver has trouble in this test, so disable for now
_unbounded_configs = [{"linear_solver": scs.LinearSolver.QDLDL}]
if _dense_available:
    _unbounded_configs.append({"linear_solver": scs.LinearSolver.CPU_DENSE})


@pytest.mark.parametrize("solver_opts", _unbounded_configs)
def test_unbounded(solver_opts):
    rng = np.random.RandomState(2002)
    for i in range(num_unb):
        data = tools.gen_unbounded(K, n=m // 2, rng=rng)
        solver = scs.SCS(data, K, **solver_opts, **opts)
        sol = solver.solve()
        check_unbounded(sol)
