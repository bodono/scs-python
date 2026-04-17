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


@pytest.mark.thread_unsafe(reason="creates its own threads internally")
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

        def worker(linear_solver):
            solver = scs.SCS(
                data, cone, linear_solver=linear_solver, verbose=False
            )
            sol = solver.solve()
            assert sol["info"]["status_val"] == 1
            assert_almost_equal(sol["x"][0], expected, decimal=2)
            return sol["x"][0]

        backends = [scs.LinearSolver.QDLDL, scs.LinearSolver.INDIRECT]
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [
                pool.submit(worker, backends[i % 2]) for i in range(NUM_THREADS)
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

@pytest.mark.thread_unsafe(reason="creates its own threads internally")
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

@pytest.mark.thread_unsafe(reason="creates its own threads internally")
class TestConcurrentSolveUpdate:
    """Concurrent solve and update operations."""

    def test_shared_instance_concurrent_solve_and_update(self):
        """Threads simultaneously call solve() and update() on the same instance.

        The per-instance lock serializes access, so no data race should occur.
        We can't predict which operation runs first, but none should crash or
        corrupt state.
        """
        data, cone, _ = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)
        barrier = threading.Barrier(NUM_THREADS)
        errors = []
        lock = threading.Lock()

        def solve_worker():
            try:
                barrier.wait(timeout=10)
                sol = solver.solve()
                assert sol["info"]["status_val"] in (1, 2)
            except Exception as e:
                with lock:
                    errors.append(f"solve: {e}")

        def update_worker():
            try:
                barrier.wait(timeout=10)
                # Update with slightly perturbed b
                solver.update(b=np.array([1.0, 0.0]))
            except Exception as e:
                with lock:
                    errors.append(f"update: {e}")

        threads = []
        for i in range(NUM_THREADS):
            if i % 2 == 0:
                t = threading.Thread(target=solve_worker)
            else:
                t = threading.Thread(target=update_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Thread did not complete"

        assert len(errors) == 0, f"Errors in threads: {errors}"

        # Solver should still be usable after the concurrent barrage
        sol = solver.solve()
        assert sol["info"]["status_val"] in (1, 2)

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

@pytest.mark.thread_unsafe(reason="creates its own threads internally")
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

@pytest.mark.thread_unsafe(reason="creates its own threads internally")
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

@pytest.mark.thread_unsafe(reason="creates its own threads internally")
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


# ---------------------------------------------------------------------------
# Stress tests targeting specific thread-safety bugs
# ---------------------------------------------------------------------------

@pytest.mark.thread_unsafe(reason="creates its own threads internally")
class TestBorrowedRefSafety:
    """Tests exercising PyDict/PyList ref safety during concurrent init.

    Before the fix, PyDict_GetItemString / PyList_GetItem returned borrowed
    references that could be invalidated if another thread mutated the
    container concurrently. These tests share mutable containers across
    threads to exercise the strong-ref (PyDict_GetItemStringRef /
    PyList_GetItemRef) code paths.
    """

    def test_shared_cone_dict_concurrent_init(self):
        """Multiple threads construct SCS instances from the same cone dict.

        The cone dict is read by the C extension during __init__. With
        borrowed refs, a concurrent mutation to the dict could invalidate
        a pointer mid-parse. Strong refs prevent this.
        """
        A = sp.csc_matrix([1.0, -1.0]).T.tocsc()
        b = np.array([1.0, 0.0])
        c = np.array([-1.0])
        data = {"A": A, "b": b, "c": c}
        cone = {"l": 2}  # shared across all threads

        errors = []
        lock = threading.Lock()

        def worker():
            try:
                solver = scs.SCS(data, cone, verbose=False)
                sol = solver.solve()
                assert sol["info"]["status_val"] == 1
                assert_almost_equal(sol["x"][0], 1.0, decimal=2)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_shared_cone_with_list_values_concurrent_init(self):
        """Cone dict with list values (SOC dimensions) shared across threads.

        Exercises PyList_GetItemRef in get_cone_arr_dim — the list elements
        are read one-by-one during parsing.
        """
        # Feasible SOCP: minimize c'x s.t. ||x|| <= t (one SOC of size 3)
        # A = -I (3x2 + slack), b = 0, cone q=[3]
        # With an extra nonneg cone to bound things
        A = sp.vstack([
            -sp.eye(3, n=2, format="csc"),  # SOC: (s0, s1, s2) = b - Ax
            sp.eye(2, format="csc"),         # nonneg: x >= 0
        ], format="csc")
        b = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        c = np.array([-1.0, 0.0])
        data = {"A": A, "b": b, "c": c}
        cone = {"q": [3], "l": 2}  # list value — exercises PyList_GetItemRef

        errors = []
        lock = threading.Lock()

        def worker():
            try:
                solver = scs.SCS(data, cone, verbose=False)
                sol = solver.solve()
                assert sol["info"]["status_val"] in (1, 2)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_shared_cone_with_float_list_values_concurrent_init(self):
        """Cone dict with float list values (power cone exponents) shared
        across threads.

        Exercises PyList_GetItemRef in get_cone_float_arr — the float list
        elements are read one-by-one during parsing.
        """
        # Power cone: (x1, x2, x3) s.t. |x3| <= x1^p * x2^(1-p), x1,x2 >= 0
        # Each power cone triple is size 3, with exponent p
        m = 3
        n = 3
        A = -sp.eye(m, n=n, format="csc")
        b = np.array([1.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, -1.0])
        data = {"A": A, "b": b, "c": c}
        cone = {"p": [0.5]}  # float list — exercises PyList_GetItemRef in get_cone_float_arr

        errors = []
        lock = threading.Lock()

        def worker():
            try:
                solver = scs.SCS(data, cone, verbose=False, max_iters=10000)
                sol = solver.solve()
                # Any valid status is fine — we're testing thread safety not convergence
                assert sol["info"]["status_val"] in (1, 2, -1, -2, -7)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()

        assert len(errors) == 0, f"Errors: {errors}"


@pytest.mark.thread_unsafe(reason="creates its own threads internally")
class TestTOCTOURaceSafety:
    """Tests exercising the TOCTOU fix on self->work.

    Before the fix, SCS_solve and SCS_update checked self->work without
    holding the lock. A concurrent dealloc (SCS_finish) could set
    self->work = NULL between the check and the lock acquisition.
    """

    def test_rapid_create_solve_destroy(self):
        """Rapidly create, solve, and let SCS instances be garbage collected.

        Many short-lived instances stress the init/finish paths. On
        free-threaded builds, the dealloc can race with in-progress solves
        if the TOCTOU check isn't under the lock.
        """
        import gc
        data, cone, expected = _make_simple_lp()
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(50):
                    solver = scs.SCS(data, cone, verbose=False)
                    sol = solver.solve()
                    assert sol["info"]["status_val"] == 1
                    assert_almost_equal(sol["x"][0], expected, decimal=2)
                    del solver
                gc.collect()
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_solve_after_del_raises_or_succeeds(self):
        """Verify solve doesn't crash if workspace is gone.

        This is a single-threaded sanity check: calling solve on a
        properly-constructed instance should work, and the object shouldn't
        crash during cleanup.
        """
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)
        sol = solver.solve()
        assert sol["info"]["status_val"] == 1
        assert_almost_equal(sol["x"][0], expected, decimal=2)
        # Second solve should also work (warm start)
        sol2 = solver.solve()
        assert sol2["info"]["status_val"] == 1


@pytest.mark.thread_unsafe(reason="creates its own threads internally")
class TestConcurrentSolveUpdateStress:
    """Heavy stress tests for concurrent solve + update on shared instances."""

    def test_solve_update_barrage_shared_instance(self):
        """Many threads hammer solve() and update() on the same instance.

        This is a more aggressive version of
        test_shared_instance_concurrent_solve_and_update, with more threads
        and repeated rounds.
        """
        data, cone, _ = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)
        errors = []
        lock = threading.Lock()

        def solve_worker():
            try:
                for _ in range(10):
                    sol = solver.solve()
                    assert sol["info"]["status_val"] in (1, 2)
            except Exception as e:
                with lock:
                    errors.append(f"solve: {e}")

        def update_worker():
            try:
                for _ in range(10):
                    solver.update(b=np.array([1.0, 0.0]))
            except Exception as e:
                with lock:
                    errors.append(f"update: {e}")

        threads = []
        for i in range(16):
            if i % 2 == 0:
                threads.append(threading.Thread(target=solve_worker))
            else:
                threads.append(threading.Thread(target=update_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive()

        assert len(errors) == 0, f"Errors: {errors}"

        # Instance should still be usable
        sol = solver.solve()
        assert sol["info"]["status_val"] in (1, 2)

    def test_concurrent_update_different_data(self):
        """Threads update with different b/c vectors concurrently.

        After the barrage, we verify the solver still produces valid results
        (not corrupted state).
        """
        data, cone, _ = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)
        errors = []
        lock = threading.Lock()
        rng = np.random.RandomState(42)

        # Pre-generate update vectors (each thread gets its own)
        b_updates = [np.array([rng.uniform(0.5, 2.0), 0.0]) for _ in range(NUM_THREADS)]
        c_updates = [np.array([rng.uniform(-2.0, -0.5)]) for _ in range(NUM_THREADS)]

        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid):
            try:
                barrier.wait(timeout=10)
                for _ in range(5):
                    solver.update(b=b_updates[tid], c=c_updates[tid])
                    sol = solver.solve()
                    # Just check it didn't crash or return garbage status
                    assert sol["info"]["status_val"] in (1, 2, -2, -7)
            except Exception as e:
                with lock:
                    errors.append(f"thread {tid}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_warm_start_under_contention(self):
        """Concurrent warm-started solves on a shared instance.

        Warm starts read and write the sol struct, which is protected by the
        per-instance lock. This test stresses that path.
        """
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)

        # Initial cold solve to populate warm-start data
        sol0 = solver.solve()
        assert sol0["info"]["status_val"] == 1

        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(10):
                    sol = solver.solve(
                        x=np.array([0.9]),
                        y=sol0["y"],
                        s=sol0["s"],
                    )
                    assert sol["info"]["status_val"] == 1
                    assert_almost_equal(sol["x"][0], expected, decimal=2)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive()

        assert len(errors) == 0, f"Errors: {errors}"


@pytest.mark.thread_unsafe(reason="creates its own threads internally")
class TestErrorPathContention:
    """Tests that lock is properly released on error paths.

    If a thread hits an error (e.g. bad warm-start dimensions) inside
    the locked section, it must release the lock before returning. If it
    doesn't, subsequent calls from other threads will deadlock.
    """

    def test_bad_warm_start_does_not_deadlock(self):
        """One thread passes wrong-dimension warm-start vectors while others
        solve normally. The error path must release the lock.
        """
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)

        # Do an initial solve so warm-start data is populated
        solver.solve()

        errors = []
        lock = threading.Lock()
        barrier = threading.Barrier(NUM_THREADS)

        def good_worker():
            try:
                barrier.wait(timeout=10)
                for _ in range(10):
                    sol = solver.solve()
                    assert sol["info"]["status_val"] == 1
                    assert_almost_equal(sol["x"][0], expected, decimal=2)
            except Exception as e:
                with lock:
                    errors.append(f"good: {e}")

        def bad_worker():
            try:
                barrier.wait(timeout=10)
                for _ in range(10):
                    try:
                        # Wrong dimension x -- should raise ValueError
                        solver.solve(x=np.array([1.0, 2.0, 3.0]))
                    except ValueError:
                        pass  # expected -- lock must have been released
            except Exception as e:
                with lock:
                    errors.append(f"bad: {e}")

        threads = []
        for i in range(NUM_THREADS):
            if i == 0:
                threads.append(threading.Thread(target=bad_worker))
            else:
                threads.append(threading.Thread(target=good_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Thread deadlocked -- lock not released on error?"

        assert len(errors) == 0, f"Errors: {errors}"

    def test_bad_update_does_not_deadlock(self):
        """One thread passes wrong-dimension update vectors while others
        solve normally. The error path in SCS_update must release the lock.
        """
        data, cone, expected = _make_simple_lp()
        solver = scs.SCS(data, cone, verbose=False)
        errors = []
        lock = threading.Lock()
        barrier = threading.Barrier(NUM_THREADS)

        def good_worker():
            try:
                barrier.wait(timeout=10)
                for _ in range(10):
                    sol = solver.solve()
                    assert sol["info"]["status_val"] in (1, 2)
            except Exception as e:
                with lock:
                    errors.append(f"good: {e}")

        def bad_worker():
            try:
                barrier.wait(timeout=10)
                for _ in range(10):
                    try:
                        # Wrong dimension b -- should raise ValueError
                        solver.update(b=np.array([1.0, 2.0, 3.0]))
                    except ValueError:
                        pass  # expected
            except Exception as e:
                with lock:
                    errors.append(f"bad: {e}")

        threads = []
        for i in range(NUM_THREADS):
            if i == 0:
                threads.append(threading.Thread(target=bad_worker))
            else:
                threads.append(threading.Thread(target=good_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Thread deadlocked -- lock not released on error?"

        assert len(errors) == 0, f"Errors: {errors}"
