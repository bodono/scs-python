import pytest
import threading
import numpy as np
import scipy.sparse as sp
import scs
import sys
from numpy.testing import assert_almost_equal
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import gen_random_cone_prob as tools

# --- Global constant ---
FAIL = "failure"

SHARED_DATA_FOR_TEST = {
    "A": sp.csc_matrix([1.0, -1.0]).T.tocsc(),
    "b": np.array([1.0, 0.0]),
    "c": np.array([-1.0])
}
SHARED_CONE_CONFIG_FOR_TEST = {"q": [], "l": 2}
EXPECTED_X0_FOR_SHARED_PROBLEM_TEST = 1.0
NUM_CONCURRENT_SOLVES=8

# Cone definition
K_CONFIG = {
    "z": 5,
    "l": 10,
    "q": [3, 4],
    "s": [2, 3],
    "ep": 4,
    "ed": 4,
    "p": [-0.25, 0.5],
}

SOLVER_PARAMS_CONFIG = {
    "verbose": False,
    "eps_abs": 1e-5,
    "eps_rel": 1e-5,
    "eps_infeas": 1e-5,
    "max_iters": 3500,
}

UPDATE_TEST_C_NEW = np.array([1.0])
UPDATE_TEST_B_NEW = np.array([1.0, 1.0])

EXPECTED_X1_UPDATE = 1.0
EXPECTED_X2_UPDATE = 0.0
EXPECTED_X3_UPDATE = -1.0

# --- Worker function executed by each thread ---
def solve_one_random_cone_problem(cone_def, solver_params_def, worker_id):
    """
    Generates a random feasible cone problem, solves it with SCS, and performs assertions.
    This function is intended to be run in a separate thread.
    Returns True on success, raises AssertionError on failure.
    """
    thread_name = threading.current_thread().name
    print(f"[Worker {worker_id} on {thread_name}]")

    m_dims = tools.get_scs_cone_dims(cone_def)
    n_vars = m_dims // 2
    if n_vars == 0: n_vars = 1

    # Generate a new feasible problem for each worker
    data, p_star_expected = tools.gen_feasible(cone_def, n=n_vars, density=0.2)

    print(f"[Worker {worker_id} on {thread_name}]: Problem generated. m={m_dims}, n={n_vars}. Expected p_star ~ {p_star_expected:.4f}")

    # Create and run the SCS solver
    solver = scs.SCS(data, cone_def, use_indirect=False, gpu=False, **solver_params_def)
    sol = solver.solve()
    x_sol = sol["x"]
    y_sol = sol["y"]
    s_sol = sol["s"]
    info = sol["info"]

    print(f"[Worker {worker_id} on {thread_name}]: Solved. Status: {info['status']}. Pobj: {info['pobj']:.4f}, Iters: {info['iter']}")

    # Assertions (similar to test_solve_feasible)
    # 1. Objective value
    np.testing.assert_almost_equal(np.dot(data["c"], x_sol), p_star_expected, decimal=2,
                                   err_msg=f"Worker {worker_id}: Objective value mismatch.")

    # 2. Primal feasibility (Ax - b + s = 0  => ||Ax - b + s|| ~ 0)
    # Relaxed tolerance from 1e-3 to 5e-3
    primal_residual_norm = np.linalg.norm(data["A"] @ x_sol - data["b"] + s_sol)
    np.testing.assert_array_less(primal_residual_norm, 5e-3,
                                  err_msg=f"Worker {worker_id}: Primal residual norm too high: {primal_residual_norm}")

    # 3. Dual feasibility (A'y + c = 0 => ||A'y + c|| ~ 0 for LP part, more complex for cones)
    # Relaxed tolerance from 1e-3 to 5e-3
    dual_residual_norm = np.linalg.norm(data["A"].T @ y_sol + data["c"])
    np.testing.assert_array_less(dual_residual_norm, 5e-3,
                                  err_msg=f"Worker {worker_id}: Dual residual norm too high: {dual_residual_norm}")

    # 4. Complementary slackness (s'y ~ 0)
    complementarity = s_sol.T @ y_sol
    np.testing.assert_almost_equal(complementarity, 0.0, decimal=3, # Check if close to zero
                                   err_msg=f"Worker {worker_id}: Complementary slackness violation: {complementarity}")

    # 5. Slack variable s in primal cone K (s = proj_K(s))
    projected_s = tools.proj_cone(s_sol, cone_def)
    np.testing.assert_almost_equal(s_sol, projected_s, decimal=3,
                                   err_msg=f"Worker {worker_id}: Slack variable s not in primal cone.")

    # 6. Dual variable y in dual cone K* (y = proj_K*(y))
    projected_y_dual = tools.proj_dual_cone(y_sol, cone_def)
    np.testing.assert_almost_equal(y_sol, projected_y_dual, decimal=3,
                                   err_msg=f"Worker {worker_id}: Dual variable y not in dual cone.")

    print(f"[Worker {worker_id} on {thread_name}]: All assertions passed.")
    return {"id": worker_id, "status": "success", "pobj": info['pobj'], "iters": info['iter']}

# --- Pytest test function using ThreadPoolExecutor ---
pytest.mark.skipif(sys._is_gil_enabled(), "Only for free threaded")
def test_concurrent_independent_cone_solves():
    """
    Tests running multiple independent SCS solves concurrently using ThreadPoolExecutor.
    Each solve uses the provided use_indirect and gpu flags.
    """
    completed_solves = 0
    failed_solves_details = []

    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_SOLVES) as executor:
        futures = []
        for i in range(NUM_CONCURRENT_SOLVES):
            worker_id = i + 1
            future = executor.submit(
                solve_one_random_cone_problem,
                K_CONFIG,
                SOLVER_PARAMS_CONFIG,
                worker_id
            )
            futures.append(future)
            print(f"pytest: Submitted task for worker {worker_id}.")

        print(f"\npytest: All {NUM_CONCURRENT_SOLVES} tasks submitted. Waiting for completion...\n")

        for future in as_completed(futures, timeout=NUM_CONCURRENT_SOLVES * 60.0):
            # Determine worker_id based on the future object's position in the original list.
            # This is a bit fragile if futures list were modified, but common for simple cases.
            # A more robust way would be to wrap future with its ID if needed for complex scenarios.
            worker_id_from_future = -1 # Default / placeholder
            for idx, f_item in enumerate(futures):
                if f_item == future:
                    worker_id_from_future = idx + 1
                    break

            try:
                result = future.result(timeout=60.0)
                print(f"pytest: Worker {result.get('id', worker_id_from_future)} completed successfully: {result}")
                completed_solves += 1
            except Exception as e:
                error_detail = f"Worker {worker_id_from_future} failed: {type(e).__name__}: {e}"
                print(f"pytest: ERROR - {error_detail}")
                failed_solves_details.append(error_detail)

    print(f"\npytest: Test execution finished.")
    print(f"Total solves attempted: {NUM_CONCURRENT_SOLVES}")
    print(f"Successful solves: {completed_solves}")
    print(f"Failed solves: {len(failed_solves_details)}")

    if failed_solves_details:
        pytest.fail(f"{len(failed_solves_details)} out of {NUM_CONCURRENT_SOLVES} concurrent solves failed.\n"
                    f"Failures:\n" + "\n".join(failed_solves_details))

    assert completed_solves == NUM_CONCURRENT_SOLVES, \
        f"Expected {NUM_CONCURRENT_SOLVES} successful concurrent solves, but got {completed_solves}."

    print(f"pytest: All {NUM_CONCURRENT_SOLVES} concurrent solves passed.")

def worker_perform_solve_update_sequence(solver_params_def, worker_id):
    """
    Performs a sequence of solve and update operations on an SCS instance.
    """
    thread_name = threading.current_thread().name
    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: Starting")

    solver = scs.SCS(SHARED_DATA_FOR_TEST, SHARED_CONE_CONFIG_FOR_TEST,
                     use_indirect=False, gpu=False, **solver_params_def)

    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: Performing initial solve.")
    sol1 = solver.solve()
    np.testing.assert_almost_equal(sol1["x"][0], EXPECTED_X1_UPDATE, decimal=2,
                                   err_msg=f"Worker {worker_id} (UpdateSeq): Initial solve failed.")
    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: Initial solve OK, x={sol1['x'][0]:.2f}")

    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: Updating c and solving.")
    solver.update(c=UPDATE_TEST_C_NEW)
    sol2 = solver.solve()
    np.testing.assert_almost_equal(sol2["x"][0], EXPECTED_X2_UPDATE, decimal=2,
                                   err_msg=f"Worker {worker_id} (UpdateSeq): Solve after c update failed.")
    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: Solve after c update OK, x={sol2['x'][0]:.2f}")

    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: Updating b and solving.")
    solver.update(b=UPDATE_TEST_B_NEW)
    sol3 = solver.solve()
    np.testing.assert_almost_equal(sol3["x"][0], EXPECTED_X3_UPDATE, decimal=2,
                                   err_msg=f"Worker {worker_id} (UpdateSeq): Solve after b update failed.")
    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: Solve after b update OK, x={sol3['x'][0]:.2f}")

    print(f"[Worker {worker_id} (UpdateSeq) on {thread_name}]: All update sequence assertions passed.")
    return {"id": worker_id, "type": "UpdateSeq", "status": "success"}


# --- Test for Concurrent Solve and Update Sequences ---
pytest.mark.skipif(sys._is_gil_enabled(), "Only for free threaded")
def test_concurrent_solve_update_sequences():
    """
    Tests running multiple SCS solve-update-solve sequences concurrently.
    """
    print(f"\npytest: Starting concurrent solve-update sequences test (use_indirect=False, gpu=False)")

    completed_jobs = 0
    failed_jobs_details = []

    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_SOLVES) as executor:
        futures = []
        for i in range(NUM_CONCURRENT_SOLVES):
            worker_id = i + 1
            future = executor.submit(
                worker_perform_solve_update_sequence,
                SOLVER_PARAMS_CONFIG, worker_id
            )
            futures.append(future)
            print(f"pytest: Submitted task for UpdateSeq worker {worker_id}.")

        print(f"\npytest: All {NUM_CONCURRENT_SOLVES} UpdateSeq tasks submitted. Waiting for completion...\n")
        for future in as_completed(futures, timeout=NUM_CONCURRENT_SOLVES * 30.0):
            worker_id_from_future = futures.index(future) + 1
            try:
                result = future.result(timeout=30.0)
                print(f"pytest: UpdateSeq Worker {result.get('id', worker_id_from_future)} completed successfully: {result}")
                completed_jobs += 1
            except Exception as e:
                error_detail = f"UpdateSeq Worker {worker_id_from_future} failed: {type(e).__name__}: {e}"
                print(f"pytest: ERROR - {error_detail}")
                failed_jobs_details.append(error_detail)

    print(f"\npytest: UpdateSeq test execution finished.")
    print(f"Total UpdateSeq jobs attempted: {NUM_CONCURRENT_SOLVES}, Successful: {completed_jobs}, Failed: {len(failed_jobs_details)}")

    if failed_jobs_details:
        pytest.fail(f"{len(failed_jobs_details)} out of {NUM_CONCURRENT_SOLVES} concurrent UpdateSeq jobs failed.\nFailures:\n" + "\n".join(failed_jobs_details))
    assert completed_jobs == NUM_CONCURRENT_SOLVES, f"Expected {NUM_CONCURRENT_SOLVES} successful UpdateSeq jobs, got {completed_jobs}."
    print(f"pytest: All {NUM_CONCURRENT_SOLVES} concurrent UpdateSeq jobs passed.")

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
