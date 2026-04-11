from __future__ import print_function, division
import scs
import numpy as np
import gen_random_cone_prob as tools

#############################################
#  Uses scs to solve a random cone problem  #
#############################################


def main():
    solvers = [scs.LinearSolver.AUTO, scs.LinearSolver.QDLDL, scs.LinearSolver.INDIRECT]
    try:
        from scs import _scs_gpu

        solvers.append(scs.LinearSolver.GPU)
    except ImportError:
        pass

    for linear_solver in solvers:
        np.random.seed(1)
        solve_feasible(linear_solver)
        solve_infeasible(linear_solver)
        solve_unbounded(linear_solver)


def solve_feasible(linear_solver):
    # cone:
    K = {
        "z": 10,
        "l": 15,
        "q": [5, 10, 0, 1],
        "s": [3, 4, 0, 0, 1],
        "ep": 10,
        "ed": 10,
        "p": [-0.25, 0.5, 0.75, -0.33],
    }
    m = tools.get_scs_cone_dims(K)
    data, p_star = tools.gen_feasible(K, n=m // 3, density=0.01)
    params = {"normalize": True, "scale": 5}

    sol = scs.solve(data, K, linear_solver=linear_solver, **params)
    x = sol["x"]
    y = sol["y"]
    print("p*  = ", p_star)
    print("pri error = ", (np.dot(data["c"], x) - p_star) / p_star)
    print("dual error = ", (-np.dot(data["b"], y) - p_star) / p_star)


def solve_infeasible(linear_solver):
    K = {
        "z": 10,
        "l": 15,
        "q": [5, 10, 0, 1],
        "s": [3, 4, 0, 0, 1],
        "ep": 10,
        "ed": 10,
        "p": [-0.25, 0.5, 0.75, -0.33],
    }
    m = tools.get_scs_cone_dims(K)
    data = tools.gen_infeasible(K, n=m // 3)
    params = {"normalize": True, "scale": 0.5}
    sol = scs.solve(data, K, linear_solver=linear_solver, **params)


def solve_unbounded(linear_solver):
    K = {
        "z": 10,
        "l": 15,
        "q": [5, 10, 0, 1],
        "s": [3, 4, 0, 0, 1],
        "ep": 10,
        "ed": 10,
        "p": [-0.25, 0.5, 0.75, -0.33],
    }
    m = tools.get_scs_cone_dims(K)
    data = tools.gen_unbounded(K, n=m // 3)
    params = {"normalize": True, "scale": 0.5}
    sol = scs.solve(data, K, linear_solver=linear_solver, **params)


if __name__ == "__main__":
    main()
