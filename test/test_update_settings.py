"""Tests of changing settings after construction with SCS.update(**settings).

The problem is a random feasible LP (n = 100, m = 300) that takes well over
a few iterations from a cold start, so a time limit of 1e-10 s always stops
it at the first convergence check.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import scs

_dense_available = False
try:
    from scs import _scs_dense  # noqa: F401
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

_INACCURATE = (scs.SOLVED_INACCURATE, scs.INFEASIBLE_INACCURATE,
               scs.UNBOUNDED_INACCURATE)


def _random_lp(seed=0, n=100, m=300, z=30):
    """A feasible, bounded LP with a known primal-dual solution."""
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=0.1, random_state=rng, format="csc")
    A = A + sp.eye(m, n, format="csc")  # no empty columns
    x = rng.standard_normal(n)
    # complementary slack s and dual y on the cone {0}^z x R_+^(m - z)
    s = np.zeros(m)
    y = np.zeros(m)
    y[:z] = rng.standard_normal(z)
    pos = rng.random(m - z) < 0.5
    s[z:][pos] = rng.random(pos.sum())
    y[z:][~pos] = rng.random((~pos).sum())
    data = {"A": A.tocsc(), "b": A @ x + s, "c": -A.T @ y}
    return data, {"z": z, "l": m - z}


DATA, CONE = _random_lp()


def _hit_time_limit(sol):
    info = sol["info"]
    return (info["status_val"] in _INACCURATE
            and "time_limit_secs" in info["status"])


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_time_limit_after_setup(solver_opts):
    solver = scs.SCS(DATA, CONE, verbose=False, **solver_opts)
    sol = solver.solve(warm_start=False)
    assert sol["info"]["status_val"] == scs.SOLVED
    setup_time = sol["info"]["setup_time"]

    solver.update(time_limit_secs=1e-10)
    sol = solver.solve(warm_start=False)
    assert _hit_time_limit(sol), sol["info"]["status"]
    # changing only settings does not touch the setup time
    assert sol["info"]["setup_time"] == setup_time
    # the new limit stays in force
    sol = solver.solve(warm_start=False)
    assert _hit_time_limit(sol), sol["info"]["status"]

    solver.update(time_limit_secs=0)  # 0 = no limit
    sol = solver.solve(warm_start=False)
    assert sol["info"]["status_val"] == scs.SOLVED


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_time_limit_from_construction_removed(solver_opts):
    solver = scs.SCS(DATA, CONE, verbose=False, time_limit_secs=1e-10,
                     **solver_opts)
    solver.update(time_limit_secs=0)
    sol = solver.solve(warm_start=False)
    assert sol["info"]["status_val"] == scs.SOLVED


@pytest.mark.parametrize("solver_opts", _solver_configs)
def test_tolerance_override(solver_opts):
    tight = dict(eps_abs=1e-7, eps_rel=1e-7, eps_infeas=1e-9, alpha=1.8)
    ref = scs.SCS(DATA, CONE, verbose=False, **tight, **solver_opts).solve()

    solver = scs.SCS(DATA, CONE, verbose=False, eps_abs=1e-3, eps_rel=1e-3,
                     **solver_opts)
    solver.update(**tight)
    sol = solver.solve()
    # Before the first solve, changing settings is the same as constructing
    # with them, up to rounding: results are compared to a tolerance on every
    # backend, since an exact match can hold by chance (MKL's BLAS, which the
    # QDLDL and indirect modules also use on Linux, can round a later solve
    # differently from an earlier identical one).
    assert sol["info"]["status_val"] == ref["info"]["status_val"] == scs.SOLVED
    for key in ("x", "y", "s"):
        np.testing.assert_allclose(sol[key], ref[key], atol=1e-5)
    assert sol["info"]["res_pri"] < 1e-5

    solver.update(max_iters=3)
    sol = solver.solve(warm_start=False)
    assert sol["info"]["iter"] <= 3
    assert sol["info"]["status_val"] != scs.SOLVED


@pytest.mark.parametrize(
    "setting",
    [
        {"scale": 0.5},
        {"rho_x": 1e-3},
        {"normalize": False},
        {"adaptive_scale": False},
        {"acceleration_lookback": 3},
        {"acceleration_interval": 7},
        {"write_data_filename": "never_written"},
        {"linear_solver": scs.LinearSolver.QDLDL},
    ],
    ids=lambda s: next(iter(s)),
)
def test_refused_setup_setting(setting):
    solver = scs.SCS(DATA, CONE, verbose=False)
    ref = solver.solve(warm_start=False)
    b_new = 2 * DATA["b"]
    with pytest.raises(ValueError, match="used to set up the solver"):
        solver.update(b=b_new, max_iters=1, **setting)
    # nothing changed: not max_iters, not b
    sol = solver.solve(warm_start=False)
    assert sol["info"]["iter"] > 1
    np.testing.assert_allclose(sol["x"], ref["x"], atol=1e-3)


def test_refused_by_core():
    # The extension object takes complete settings; the core refuses a setup
    # setting that differs from construction and changes nothing.
    solver = scs.SCS(DATA, CONE, verbose=False)
    with pytest.raises(ValueError, match="refused the settings update"):
        solver._solver.update_settings(verbose=False, scale=0.5, max_iters=1)
    sol = solver.solve(warm_start=False)
    assert sol["info"]["iter"] > 1


@pytest.mark.parametrize("name", ["not_a_setting", "warm_start", "x"])
def test_unknown_setting(name):
    solver = scs.SCS(DATA, CONE, verbose=False)
    with pytest.raises(TypeError, match="unknown setting"):
        solver.update(max_iters=1, **{name: 1})
    sol = solver.solve(warm_start=False)
    assert sol["info"]["iter"] > 1


@pytest.mark.parametrize(
    "setting, msg",
    [
        ({"time_limit_secs": -1.0}, "time_limit_secs must be"),
        ({"time_limit_secs": float("nan")}, "time_limit_secs must be"),
        ({"max_iters": 0}, "max_iters must be positive"),
        ({"alpha": 2.0}, "alpha must be"),
        ({"eps_abs": -1e-3}, "eps_abs must be"),
    ],
)
def test_invalid_value(setting, msg):
    solver = scs.SCS(DATA, CONE, verbose=False)
    with pytest.raises(ValueError, match=msg):
        solver.update(**setting)
    sol = solver.solve(warm_start=False)
    assert sol["info"]["status_val"] == scs.SOLVED


def test_wrong_type():
    solver = scs.SCS(DATA, CONE, verbose=False)
    with pytest.raises(TypeError):
        solver.update(verbose=1)  # must be a bool, as at construction


def test_update_b_and_settings_together():
    solver = scs.SCS(DATA, CONE, verbose=False)
    solver.solve()
    b_new = DATA["b"] + 0.1
    solver.update(b=b_new, eps_abs=1e-6, eps_rel=1e-6)
    sol = solver.solve()
    ref = scs.SCS(dict(DATA, b=b_new), CONE, verbose=False, eps_abs=1e-6,
                  eps_rel=1e-6).solve()
    assert sol["info"]["status_val"] == scs.SOLVED
    np.testing.assert_allclose(sol["x"], ref["x"], atol=1e-4)


def test_bad_b_with_settings_changes_nothing():
    solver = scs.SCS(DATA, CONE, verbose=False)
    with pytest.raises(ValueError, match="incompatible dimension"):
        solver.update(b=np.ones(3), max_iters=1)
    sol = solver.solve(warm_start=False)
    assert sol["info"]["iter"] > 1
    assert sol["info"]["status_val"] == scs.SOLVED
