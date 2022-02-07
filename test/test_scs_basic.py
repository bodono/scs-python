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
    from numpy.testing import assert_almost_equal
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

FAIL = "failure"  # scs code for failure


@pytest.mark.parametrize(
    "cone,use_indirect,expected",
    [
        ({"q": [], "l": 2}, False, 1),
        ({"q": [], "l": 2}, True, 1),
        ({"q": [2], "l": 0}, False, 0.5),
        ({"q": [2], "l": 0}, True, 0.5),
    ],
)
def test_problems(cone, use_indirect, expected):
    sol = scs.solve(data, cone=cone, use_indirect=use_indirect, verbose=False)
    assert_almost_equal(sol["x"][0], expected, decimal=2)


if platform.python_version_tuple() < ("3", "0", "0"):

    @pytest.mark.parametrize(
        "cone,use_indirect,expected",
        [
            ({"q": [], "l": long(2)}, False, 1),
            ({"q": [], "l": long(2)}, True, 1),
            ({"q": [long(2)], "l": 0}, False, 0.5),
            ({"q": [long(2)], "l": 0}, True, 0.5),
        ],
    )
    def test_problems_with_longs(cone, use_indirect, expected):
        sol = scs.solve(
            data, cone=cone, use_indirect=use_indirect, verbose=False
        )
        assert_almost_equal(sol["x"][0], expected, decimal=2)


def test_failures():
    with pytest.raises(TypeError):
        scs.solve()

    with pytest.raises(ValueError):
        scs.solve(data, {"q": [4], "l": -2})

    # disable this until win64 types figured out
    # with pytest.raises(ValueError)
    # scs.solve(data, {'q': [], 'l': 2}, max_iters=-1)

    # python 2.6 and before just cast float to int
    if platform.python_version_tuple() >= ("2", "7", "0"):
        with pytest.raises(ValueError):
            scs.solve(data, {"q": [], "l": 2}, max_iters=1.1)

    with pytest.raises(ValueError):
        sol = scs.solve(data, {"q": [1], "l": 0}, verbose=False)
