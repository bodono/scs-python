"""Tests for thread safety of concurrent solve calls.

When Py_BEGIN_ALLOW_THREADS releases the GIL during scs_solve, multiple
threads sharing a single SCS instance can race on the workspace and
solution buffers. These tests verify that the per-instance lock correctly
serializes access and produces correct results.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_almost_equal

import scs


NUM_THREADS = 4


def _make_simple_lp():
    """Simple LP: max x s.t. 0 <= x <= 1. Optimal x=1."""
    A = sp.csc_matrix([1.0, -1.0]).T.tocsc()
    b = np.array([1.0, 0.0])
    c = np.array([-1.0])
    data = {"A": A, "b": b, "c": c}
    cone = {"l": 2}
    return data, cone, 1.0  # expected x[0]


class TestThreadSafety:
    """Multiple threads call solve() on the same SCS object.

    The per-instance lock serializes access to the underlying C workspace,
    so results should be correct even under concurrent access.
    """

    def test_shared_instance_concurrent_solve(self):
        """Multiple threads solve on the same SCS instance."""
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)

        def worker():
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            return sol["x"][0]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in futures]

        for r in results:
            assert_almost_equal(r, expected, decimal=2)

    def test_shared_instance_repeated_concurrent_solve(self):
        """Repeated rounds of concurrent solves on a shared instance.

        This is the pattern that triggered the race condition: warm-start
        state from a previous solve gets corrupted by concurrent access.
        """
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)

        for _ in range(5):
            def worker():
                sol = solver.solve()
                assert sol["info"]["status_val"] == 1
                return sol["x"][0]

            with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
                futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
                results = [f.result(timeout=30) for f in futures]

            for r in results:
                assert_almost_equal(r, expected, decimal=2)

    def test_independent_instances_concurrent(self):
        """Independent instances solve fully in parallel (no contention)."""
        data, cone, expected = _make_simple_lp()

        def worker():
            solver = scs.SCS(data, cone, verbose=False)
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            return sol["x"][0]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in futures]

        for r in results:
            assert_almost_equal(r, expected, decimal=2)

    def test_concurrent_solve_update(self):
        """Each thread does solve -> update -> solve on its own instance."""
        data, cone, _ = _make_simple_lp()

        def worker():
            solver = scs.SCS(data, cone, verbose=False)

            sol1 = solver.solve()
            assert sol1["info"]["status_val"] == 1
            assert_almost_equal(sol1["x"][0], 1.0, decimal=2)

            # Update c to minimize: min x s.t. 0 <= x <= 1 => x=0
            solver.update(c=np.array([1.0]))
            sol2 = solver.solve()
            assert sol2["info"]["status_val"] == 1
            assert_almost_equal(sol2["x"][0], 0.0, decimal=2)

            return True

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in futures]

        assert all(results)
