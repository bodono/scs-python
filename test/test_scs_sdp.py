from __future__ import print_function
import os
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


# Several PSD blocks of comparable size, with enough work for SCS to project
# them in parallel when it is built for that (-Duse_openmp=true
# -Duse_parallel_psd=true). Every other build projects them one after another.
# The answer must be the same.
K_blocks = {
    "z": 4,
    "l": 6,
    "q": [5],
    "s": [24, 24, 22, 24],
    "ep": 0,
    "ed": 0,
    "p": [],
}
m_blocks = tools.get_scs_cone_dims(K_blocks)


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_feasible_several_psd_blocks(solver_opts):
    rng = np.random.RandomState(2003)
    for i in range(3):
        data, p_star = tools.gen_feasible(
            K_blocks, n=m_blocks // 3, density=0.1, rng=rng
        )
        solver = scs.SCS(data, K_blocks, **solver_opts, **opts)
        sol = solver.solve()
        assert_almost_equal(np.dot(data["c"], sol["x"]), p_star, decimal=2)
        assert_almost_equal(np.dot(-data["b"], sol["y"]), p_star, decimal=2)


# Whether those blocks are projected in parallel is decided by the build, and
# a verbose setup says so in its header. CI sets SCS_EXPECT_PARALLEL_PSD to
# "1" for a build with -Duse_parallel_psd=true and to "0" for one without,
# both with several OpenMP threads and the BLAS on one thread
# (OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1), so that each build is seen to
# take the path it is meant to take. Without the variable there is nothing to
# compare with, and the test is skipped.
_expect_parallel_psd = os.environ.get("SCS_EXPECT_PARALLEL_PSD")


@pytest.mark.skipif(
    _expect_parallel_psd not in ("0", "1"),
    reason="SCS_EXPECT_PARALLEL_PSD is not set to 0 or 1",
)
@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_parallel_psd_matches_build(solver_opts, capsys):
    rng = np.random.RandomState(2004)
    data, _ = tools.gen_feasible(
        K_blocks, n=m_blocks // 3, density=0.1, rng=rng
    )
    capsys.readouterr()
    scs.SCS(data, K_blocks, verbose=True, **solver_opts, **opts)
    header = capsys.readouterr().out
    parallel = "psd cones projected in parallel by" in header
    assert parallel == (_expect_parallel_psd == "1"), header


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
