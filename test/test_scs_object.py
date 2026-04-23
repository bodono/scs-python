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
def test_backwards_compatibility(solver_opts):
    # max x
    # s.t 0 <= x <= 1
    sol = scs.solve(data, cone, verbose=False, **solver_opts)
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_update(solver_opts):
    # max x
    # s.t 0 <= x <= 1
    solver = scs.SCS(data, cone, verbose=False, **solver_opts)
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


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_warm_start(solver_opts):
    # max x
    # s.t 0 <= x <= 1
    solver = scs.SCS(data, cone, verbose=False, **solver_opts)
    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)

    sol = solver.solve()
    assert_almost_equal(sol["x"][0], 1.0, decimal=2)
    assert_array_less(sol["info"]["iter"], 10)

    sol = solver.solve(x=None)
    sol = solver.solve(x=np.array([7.0]), y=None, s=None)
    sol = solver.solve(
        x=np.array([7.0]), y=np.array([1.0, 2.0]), s=np.array([3.0, 4.0])
    )

    with pytest.raises(ValueError):
        sol = solver.solve(x=np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "bad_kwarg,expected_in_msg",
    [
        ({"max_iters": "not_an_int"}, "integer"),
        ({"scale": "not_a_float"}, "real number"),
        ({"eps_abs": object()}, "real number"),
        ({"time_limit_secs": []}, "real number"),
        ({"acceleration_lookback": 1.5}, "integer"),
    ],
)
def test_init_type_error_is_informative(bad_kwarg, expected_in_msg):
    """Bad-typed settings should surface the native TypeError from
    PyArg_ParseTupleAndKeywords naming the expected type, not the old
    catch-all 'Error parsing inputs' ValueError."""
    with pytest.raises(TypeError) as exc_info:
        scs.SCS(data, cone, verbose=False, **bad_kwarg)
    msg = str(exc_info.value)
    assert expected_in_msg in msg, (
        f"expected {expected_in_msg!r} in error message, got: {msg!r}"
    )
    assert "Error parsing inputs" not in msg
