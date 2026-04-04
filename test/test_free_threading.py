"""Tests for free-threading (no-GIL) support.

These tests verify that scs works correctly when called concurrently from
multiple threads. On GIL-enabled builds, the tests still run but the
concurrency is serialized by the GIL. On free-threaded builds (3.13t+),
these tests exercise true parallel execution.
"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_almost_equal

import scs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_lp():
    """Simple LP: max x s.t. 0 <= x <= 1. Optimal x=1."""
    A = sp.csc_matrix([1.0, -1.0]).T.tocsc()
    b = np.array([1.0, 0.0])
    c = np.array([-1.0])
    data = {"A": A, "b": b, "c": c}
    cone = {"l": 2}
    return data, cone, 1.0  # expected x[0]


def _make_socp():
    """SOCP: max x s.t. |x| <= 1 (SOC constraint). Optimal x=0.5."""
    A = sp.csc_matrix([1.0, -1.0]).T.tocsc()
    b = np.array([1.0, 0.0])
    c = np.array([-1.0])
    data = {"A": A, "b": b, "c": c}
    cone = {"q": [2]}
    return data, cone, 0.5  # expected x[0]


def _make_larger_lp(n=20, seed=42):
    """A larger LP with n variables for more substantial computation.

    max  c'x
    s.t. Ax <= b, x >= 0

    Constructed so optimal solution has x[0] > 0.
    """
    rng = np.random.RandomState(seed)
    m = 3 * n
    # Feasible LP: random A, b = A @ ones(n) + slack
    A_dense = rng.randn(m, n)
    x_feas = np.abs(rng.randn(n)) + 0.1
    b_ineq = A_dense @ x_feas + np.abs(rng.randn(m)) + 0.1

    # Stack: inequality constraints (Ax <= b) and nonnegativity (x >= 0)
    A_full = sp.vstack([
        sp.csc_matrix(A_dense),
        sp.eye(n, format="csc") * -1.0,
    ], format="csc")
    b_full = np.concatenate([b_ineq, np.zeros(n)])
    c = -np.abs(rng.randn(n))  # negative for maximization

    data = {"A": A_full, "b": b_full, "c": c}
    cone = {"l": m + n}
    return data, cone


def _solve_and_check(solver, expected_x0, decimal=2):
    """Call solve() and check result. Returns the solution dict."""
    sol = solver.solve()
    assert sol["info"]["status_val"] in (1, 2), (
        f"Unexpected status: {sol['info']['status']}"
    )
    if expected_x0 is not None:
        assert_almost_equal(sol["x"][0], expected_x0, decimal=decimal)
    return sol


# ---------------------------------------------------------------------------
# Test: concurrent solves on INDEPENDENT instances (the primary use case)
# ---------------------------------------------------------------------------

NUM_THREADS = 8


class TestConcurrentIndependentInstances:
    """Each thread creates and solves its own SCS instance.

    This is the main use case: different optimization problems solved in
    parallel, each with its own data and workspace.
    """

    def test_concurrent_lp_solves(self):
        """Multiple threads each solve the same simple LP independently."""
        data, cone, expected = _make_simple_lp()

        def worker():
            solver = scs.SCS(data, cone, verbose=False)
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], expected, decimal=2)
            return sol["x"][0]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in futures]

        assert len(results) == NUM_THREADS
        for r in results:
            assert_almost_equal(r, expected, decimal=2)

    def test_concurrent_socp_solves(self):
        """Multiple threads each solve the same SOCP independently."""
        data, cone, expected = _make_socp()

        def worker():
            solver = scs.SCS(data, cone, verbose=False)
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], expected, decimal=2)
            return sol["x"][0]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in futures]

        assert len(results) == NUM_THREADS

    def test_concurrent_mixed_problems(self):
        """Threads solve different problem types concurrently."""
        lp_data, lp_cone, lp_expected = _make_simple_lp()
        socp_data, socp_cone, socp_expected = _make_socp()

        def lp_worker(i):
            solver = scs.SCS(lp_data, lp_cone, verbose=False)
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], lp_expected, decimal=2)
            return ("lp", i, sol["x"][0])

        def socp_worker(i):
            solver = scs.SCS(socp_data, socp_cone, verbose=False)
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], socp_expected, decimal=2)
            return ("socp", i, sol["x"][0])

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = []
            for i in range(NUM_THREADS):
                if i % 2 == 0:
                    futures.append(pool.submit(lp_worker, i))
                else:
                    futures.append(pool.submit(socp_worker, i))
            results = [f.result(timeout=30) for f in futures]

        assert len(results) == NUM_THREADS

    def test_concurrent_direct_and_indirect(self):
        """Threads use different solver backends (direct vs indirect)."""
        data, cone, expected = _make_simple_lp()

        def worker(use_indirect):
            solver = scs.SCS(
                data, cone, use_indirect=use_indirect, verbose=False
            )
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], expected, decimal=2)
            return sol["x"][0]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [
                pool.submit(worker, i % 2 == 0) for i in range(NUM_THREADS)
            ]
            results = [f.result(timeout=30) for f in futures]

        assert len(results) == NUM_THREADS

    def test_concurrent_larger_problems(self):
        """Threads solve larger problems that take more computation."""
        data, cone = _make_larger_lp(n=20)

        def worker(seed):
            solver = scs.SCS(
                data, cone, verbose=False, max_iters=5000
            )
            sol = solver.solve()
            assert sol["info"]["status_val"] in (1, 2)
            return sol["info"]["status_val"]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker, i) for i in range(NUM_THREADS)]
            results = [f.result(timeout=60) for f in futures]

        assert len(results) == NUM_THREADS

    def test_many_sequential_solves_per_thread(self):
        """Each thread creates and solves multiple problems sequentially."""
        data, cone, expected = _make_simple_lp()
        solves_per_thread = 10

        def worker():
            results = []
            for _ in range(solves_per_thread):
                solver = scs.SCS(data, cone, verbose=False)
                sol = solver.solve()
                assert sol["info"]["status_val"] == 1
                results.append(sol["x"][0])
            return results

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            all_results = [f.result(timeout=60) for f in futures]

        assert len(all_results) == NUM_THREADS
        for thread_results in all_results:
            assert len(thread_results) == solves_per_thread
            for r in thread_results:
                assert_almost_equal(r, expected, decimal=2)


# ---------------------------------------------------------------------------
# Test: concurrent solves on a SHARED instance
# ---------------------------------------------------------------------------

class TestConcurrentSharedInstance:
    """Multiple threads call solve() on the same SCS object.

    The per-instance PyMutex (in free-threaded builds) serializes access
    to the underlying C workspace, so results should still be correct.
    """

    def test_shared_instance_concurrent_solve(self):
        """Multiple threads solve on the same SCS instance."""
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)

        def worker():
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], expected, decimal=2)
            return sol["x"][0]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker) for _ in range(4)]
            results = [f.result(timeout=30) for f in futures]

        assert len(results) == 4
        for r in results:
            assert_almost_equal(r, expected, decimal=2)

    def test_shared_instance_repeated_concurrent_solve(self):
        """Repeated rounds of concurrent solves on a shared instance."""
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)

        for round_num in range(5):
            def worker():
                sol = solver.solve()
                assert sol["info"]["status_val"] == 1
                return sol["x"][0]

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(worker) for _ in range(4)]
                results = [f.result(timeout=30) for f in futures]

            for r in results:
                assert_almost_equal(r, expected, decimal=2)


# ---------------------------------------------------------------------------
# Test: concurrent solve + update sequences
# ---------------------------------------------------------------------------

class TestConcurrentSolveUpdate:
    """Each thread does a solve-update-solve sequence on its own instance."""

    def test_concurrent_update_sequences(self):
        """Multiple threads each do solve -> update -> solve independently."""
        data, cone, _ = _make_simple_lp()

        def worker():
            solver = scs.SCS(data, cone, verbose=False)

            # Initial solve: max x s.t. 0 <= x <= 1 => x=1
            sol1 = solver.solve()
            assert sol1["info"]["status_val"] == 1
            assert_almost_equal(sol1["x"][0], 1.0, decimal=2)

            # Update c to minimize: min x s.t. 0 <= x <= 1 => x=0
            solver.update(c=np.array([1.0]))
            sol2 = solver.solve()
            assert sol2["info"]["status_val"] == 1
            assert_almost_equal(sol2["x"][0], 0.0, decimal=2)

            # Update b: max x s.t. -1 <= x <= 1 => x=-1 (still minimizing)
            solver.update(b=np.array([1.0, 1.0]))
            sol3 = solver.solve()
            assert sol3["info"]["status_val"] == 1
            assert_almost_equal(sol3["x"][0], -1.0, decimal=2)

            return True

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=60) for f in futures]

        assert all(results)
        assert len(results) == NUM_THREADS

    def test_concurrent_warm_start(self):
        """Multiple threads each do warm-started solves independently."""
        data, cone, expected = _make_simple_lp()

        def worker():
            solver = scs.SCS(data, cone, verbose=False)

            # Cold solve
            sol1 = solver.solve()
            assert_almost_equal(sol1["x"][0], expected, decimal=2)

            # Warm-started solve (uses previous solution)
            sol2 = solver.solve()
            assert_almost_equal(sol2["x"][0], expected, decimal=2)
            # Warm start should converge faster
            assert sol2["info"]["iter"] <= sol1["info"]["iter"]

            # Explicit warm start vectors
            sol3 = solver.solve(
                x=np.array([0.9]),
                y=sol2["y"],
                s=sol2["s"],
            )
            assert_almost_equal(sol3["x"][0], expected, decimal=2)

            return True

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=60) for f in futures]

        assert all(results)


# ---------------------------------------------------------------------------
# Test: legacy solve() function under concurrency
# ---------------------------------------------------------------------------

class TestConcurrentLegacySolve:
    """Test the legacy scs.solve() function from multiple threads."""

    def test_concurrent_legacy_solve(self):
        """Multiple threads use the legacy solve() interface."""
        data, cone, expected = _make_simple_lp()

        def worker():
            sol = scs.solve(data, cone, verbose=False)
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], expected, decimal=2)
            return sol["x"][0]

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in futures]

        assert len(results) == NUM_THREADS


# ---------------------------------------------------------------------------
# Test: thread creation and destruction stress test
# ---------------------------------------------------------------------------

class TestThreadStress:
    """Stress tests with many short-lived threads."""

    def test_rapid_thread_creation(self):
        """Rapidly create and destroy threads, each doing a quick solve."""
        data, cone, expected = _make_simple_lp()
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                solver = scs.SCS(data, cone, verbose=False)
                sol = solver.solve()
                assert_almost_equal(sol["x"][0], expected, decimal=2)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = []
        for _ in range(50):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Thread did not complete"

        assert len(errors) == 0, f"Errors in threads: {errors}"

    def test_concurrent_construction_and_solve(self):
        """Threads concurrently construct SCS objects and solve."""
        data, cone, expected = _make_simple_lp()

        barrier = threading.Barrier(NUM_THREADS)

        def worker():
            # All threads start constructing at roughly the same time
            barrier.wait(timeout=10)
            solver = scs.SCS(data, cone, verbose=False)
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], expected, decimal=2)
            return True

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker) for _ in range(NUM_THREADS)]
            results = [f.result(timeout=30) for f in futures]

        assert all(results)


# ---------------------------------------------------------------------------
# Test: result isolation (no cross-thread data corruption)
# ---------------------------------------------------------------------------

class TestResultIsolation:
    """Verify that results from concurrent solves don't get mixed up."""

    def test_result_vectors_are_independent(self):
        """Each thread's result arrays are independent memory."""
        data, cone, expected = _make_simple_lp()
        results_lock = threading.Lock()
        all_results = []

        def worker(thread_id):
            solver = scs.SCS(data, cone, verbose=False)
            sol = solver.solve()
            # Store the actual array objects
            with results_lock:
                all_results.append({
                    "id": thread_id,
                    "x": sol["x"].copy(),
                    "y": sol["y"].copy(),
                    "s": sol["s"].copy(),
                })
            return True

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(worker, i) for i in range(NUM_THREADS)]
            [f.result(timeout=30) for f in futures]

        # All results should be correct and identical in value
        assert len(all_results) == NUM_THREADS
        for r in all_results:
            assert_almost_equal(r["x"][0], expected, decimal=2)

    def test_different_problems_correct_results(self):
        """Threads solving different problems get their own correct answers."""
        # Problem 1: max x, 0 <= x <= 1 => x=1
        data1, cone1, exp1 = _make_simple_lp()
        # Problem 2: max x, |x| <= 1 (SOC) => x=0.5
        data2, cone2, exp2 = _make_socp()

        def worker_lp():
            solver = scs.SCS(data1, cone1, verbose=False)
            sol = solver.solve()
            assert_almost_equal(sol["x"][0], exp1, decimal=2)
            return ("lp", sol["x"][0])

        def worker_socp():
            solver = scs.SCS(data2, cone2, verbose=False)
            sol = solver.solve()
            assert_almost_equal(sol["x"][0], exp2, decimal=2)
            return ("socp", sol["x"][0])

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = []
            for i in range(NUM_THREADS):
                if i % 2 == 0:
                    futures.append(pool.submit(worker_lp))
                else:
                    futures.append(pool.submit(worker_socp))

            for f in as_completed(futures, timeout=30):
                kind, val = f.result()
                if kind == "lp":
                    assert_almost_equal(val, exp1, decimal=2)
                else:
                    assert_almost_equal(val, exp2, decimal=2)
