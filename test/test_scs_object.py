from __future__ import print_function
import platform


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
    from numpy.testing import assert_almost_equal, assert_array_less
except ImportError:
    import_error("Please install numpy.")
    raise

try:
    import scipy.sparse as sp
except ImportError:
    import_error("Please install scipy.")
    raise

# global data structures for problem
c = np.array([-1.0])
b = np.array([1.0, 0.0])
A = sp.csc_matrix([1.0, -1.0]).T.tocsc()
data = {"A": A, "b": b, "c": c}
cone = {"l": 2}


@pytest.mark.parametrize("use_indirect", [False, True])
def test_update(use_indirect):
    # max x
    # s.t 0 <= x <= 1
    solver = scs.SCS(data, cone, use_indirect=use_indirect, verbose=False)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)

    # min x
    # s.t 0 <= x <= 1
    c_new = np.array([1.0])
    solver.update(c=c_new)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 0.0, decimal=2)

    # max x
    # s.t -1 <= x <= 1
    b_new = np.array([1.0, 1.0])
    solver.update(b=b_new)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], -1.0, decimal=2)


@pytest.mark.parametrize("use_indirect", [False, True])
def test_warm_start(use_indirect):
    # max x
    # s.t 0 <= x <= 1
    solver = scs.SCS(data, cone, use_indirect=use_indirect, verbose=False)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)

    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)
    assert_array_less(sol["info"]["iter"], 10)
