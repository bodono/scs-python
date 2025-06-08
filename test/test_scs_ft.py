import pytest
import threading
import numpy as np
import scipy.sparse as sp
import scs
import sys
from numpy.testing import assert_almost_equal
import time
import queue

# --- Global constant ---
FAIL = "failure"

SHARED_DATA_FOR_TEST = {
    "A": sp.csc_matrix([1.0, -1.0]).T.tocsc(),
    "b": np.array([1.0, 0.0]),
    "c": np.array([-1.0])
}
SHARED_CONE_CONFIG_FOR_TEST = {"q": [], "l": 2}
EXPECTED_X0_FOR_SHARED_PROBLEM_TEST = 1.0

# --- Worker function for threads ---
def worker_solve_on_shared_instance(test_id, shared_solver_instance, expected_x0, results_queue):
    """
    Attempts to call solve() on a shared SCS solver instance.
    Reports result or exception to the main thread via a queue.
    """
    print(f"[Thread {test_id}]: Attempting to call solve() on the shared solver instance.")
    try:
        sol = shared_solver_instance.solve(warm_start=False, x=None, y=None, s=None)
        if sol["info"]["status"] != "solved":
            # Report failure status
            results_queue.put({
                "id": test_id,
                "status": "solver_fail_status",
                "info": sol["info"],
                "x": sol.get("x")
            })
            print(f"[Thread {test_id}]: Solver status: {sol['info']['status']}.")
            return

        assert_almost_equal(sol["x"][0], expected_x0, decimal=2)
        results_queue.put({
            "id": test_id,
            "status": "success",
            "x0": sol["x"][0],
            "info": sol["info"]
        })
        print(f"[Thread {test_id}]: Call to solve() completed. Expected x[0] ~ {expected_x0}, Got x[0] ~ {sol['x'][0]:.2f}.")

    except AssertionError as e:
        results_queue.put({"id": test_id, "status": "assertion_error", "error": e})
        print(f"[Thread {test_id}]: TEST FAILED (result inconsistent). Assertion Error: {e}")
    except Exception as e:
        results_queue.put({"id": test_id, "status": "exception", "error": e, "type": type(e).__name__})
        print(f"[Thread {test_id}]: An unexpected error occurred: {type(e).__name__}: {e}.")

pytest.mark.skipif(sys._is_gil_enabled(), "Only for free threaded")
def test_concurrent_solve_on_single_scs_instance():
    """
    Tests concurrent calls to solve() on a SINGLE scs.SCS instance.
    """
    print("\npytest: Starting test: Concurrent calls to solve() on a SINGLE SCS instance.")
    print("pytest: WARNING: This test probes potentially unsafe behavior.\n")

    # ONE SCS solver instance that will be shared among threads
    print(f"pytest: Creating a single shared SCS solver instance with data and cone={SHARED_CONE_CONFIG_FOR_TEST}")
    try:
        shared_solver = scs.SCS(
            SHARED_DATA_FOR_TEST,
            cone=SHARED_CONE_CONFIG_FOR_TEST,
            verbose=False,
            normalize=False,
            max_iters=2000
        )
        print("pytest: Shared SCS solver instance created successfully.")
    except Exception as e:
        pytest.fail(f"pytest: Failed to create the shared SCS solver instance: {type(e).__name__}: {e}")


    num_concurrent_calls = 4
    threads = []
    results_queue = queue.Queue()

    print(f"\npytest: Launching {num_concurrent_calls} threads to call solve() on the shared instance...\n")

    for i in range(num_concurrent_calls):
        test_id = i + 1
        thread = threading.Thread(
            target=worker_solve_on_shared_instance,
            args=(test_id, shared_solver, EXPECTED_X0_FOR_SHARED_PROBLEM_TEST, results_queue)
        )
        threads.append(thread)
        thread.start()
        print(f"pytest: Launched thread {test_id} to call solve() on shared instance.")

    print("\npytest: All threads launched. Waiting for completion (timeout 10s per thread)...\n")

    for i, thread in enumerate(threads):
        thread.join(timeout=10.0)
        if thread.is_alive():
            print(f"pytest: WARNING - Thread {i+1} is still alive after join timeout. Test may hang or be inconclusive for this thread.")
        else:
            print(f"pytest: Thread {i+1} has finished.")

    print("\npytest: All threads have attempted to call solve() on the shared instance.")

    success_count = 0
    solver_fail_status_count = 0
    assertion_error_count = 0
    exception_count = 0

    results_summary = []

    while not results_queue.empty():
        try:
            result = results_queue.get_nowait()
            results_summary.append(result)
            if result["status"] == "success":
                success_count += 1
            elif result["status"] == "solver_fail_status":
                solver_fail_status_count += 1
            elif result["status"] == "assertion_error":
                assertion_error_count += 1
            elif result["status"] == "exception":
                exception_count += 1
        except queue.Empty:
            break
        except Exception as e:
            print(f"pytest: Error retrieving result from queue: {e}")

    print("\n--- Results Summary ---")
    for res_idx, res_item in enumerate(results_summary):
        print(f"Result {res_idx + 1}: {res_item}")
    print("-----------------------")
    print(f"Total threads launched: {num_concurrent_calls}")
    print(f"Threads reported results: {len(results_summary)}")
    print(f"Successful solves (matching expected): {success_count}")
    print(f"Solver reported non-success status: {solver_fail_status_count}")
    print(f"Assertion errors (result mismatch): {assertion_error_count}")
    print(f"Other Python exceptions during solve: {exception_count}")

    if exception_count > 0:
        exception_details = [res for res in results_summary if res["status"] == "exception"]
        pytest.fail(f"{exception_count} thread(s) raised an unexpected Python exception during solve(). Details: {exception_details}")

    if assertion_error_count > 0:
        assertion_details = [res for res in results_summary if res["status"] == "assertion_error"]
        pytest.fail(f"{assertion_error_count} thread(s) had an assertion error (result mismatch). This indicates inconsistency. Details: {assertion_details}")

    if solver_fail_status_count > 0:
        fail_status_details = [res for res in results_summary if res["status"] == "solver_fail_status"]
        pytest.fail(f"{solver_fail_status_count} thread(s) resulted in a non-success solver status. Details: {fail_status_details}")


    assert success_count == num_concurrent_calls, \
        f"Expected all {num_concurrent_calls} threads to succeed and match expected value, but only {success_count} did. " \
        f"Solver fails: {solver_fail_status_count}, Assertion errors: {assertion_error_count}, Exceptions: {exception_count}"
