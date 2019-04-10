from __future__ import print_function
import platform
## import utilities to generate random cone probs:
import sys
import gen_random_cone_prob as tools

# if platform.system() == 'Windows':
#   print('Skipping Python linear system tests on Windows')
#   sys.exit(0)


def import_error(msg):
  print()
  print('## IMPORT ERROR:' + msg)
  print()


try:
  from nose.tools import assert_raises, assert_almost_equals
except ImportError:
  import_error('Please install nose to run tests.')
  raise

try:
  import scs
except ImportError:
  import_error('You must install the scs module before running tests.')
  raise

try:
  import numpy as np
except ImportError:
  import_error('Please install numpy.')
  raise

try:
  import scipy.sparse as sp
  import scipy.sparse.linalg as sla
except ImportError:
  import_error('Please install scipy.')
  raise


def check_solution(solution, expected):
  assert_almost_equals(solution, expected, places=2)


def assert_(str1, str2):
  if (str1 != str2):
    print('assert failure: %s != %s' % (str1, str2))
  assert str1 == str2


np.random.seed(0)
num_probs = 50

K = {
    'f': 10,
    'l': 25,
    'q': [5, 10, 0, 1],
    's': [2, 1, 2, 0, 1],
    'ep': 0,
    'ed': 0,
    'p': [0.25, -0.75, 0.33, -0.33, 0.2]
}
m = tools.get_scs_cone_dims(K)

def test_python_linsys():
  global Msolve, A, nz_cone, ncon_cone

  for i in range(num_probs):
    data, p_star = tools.gen_feasible(K, n=m // 3, density=0.1)

    A = data['A']
    ncon_cone, nz_cone = A.shape
    Msolve = None

    def init_lin_sys_work_cb(rho):
        global Msolve, nz_cone, ncon_cone, A
        M = sp.bmat([[rho*sp.eye(nz_cone), A.T], [A, -sp.eye(ncon_cone)]])
        Msolve = sla.factorized(M)

    def solve_lin_sys_cb(b, s, i):
        global Msolve
        b[:] = Msolve(b)

    def accum_by_a_cb(x, y):
        global A
        y += A.dot(x)

    def accum_by_atrans_cb(x, y):
        global A
        y += A.T.dot(x)

    def normalize_a_cb(boundaries, scale):
        global A, ncon_cone, nz_cone

        D_all = np.ones(ncon_cone)
        E_all = np.ones(nz_cone)

        min_scale, max_scale = (1e-4, 1e4)
        n_passes = 10

        for i in range(n_passes):
            D = np.sqrt(sla.norm(A, float('inf'), axis=1))
            E = np.sqrt(sla.norm(A, float('inf'), axis=0))
            D[D < min_scale] = 1.0
            E[E < min_scale] = 1.0
            D[D > max_scale] = max_scale
            E[E > max_scale] = max_scale
            start = boundaries[0]
            for delta in boundaries[1:]:
                D[start:start+delta] = D[start:start+delta].mean()
                start += delta
            A = sp.diags(1/D).dot(A).dot(sp.diags(1/E))
            D_all *= D
            E_all *= E

        mean_row_norm = sla.norm(A, 2, axis=1).mean()
        mean_col_norm = sla.norm(A, 2, axis=0).mean()
        A *= scale

        return D_all, E_all, mean_row_norm, mean_col_norm

    def un_normalize_a_cb(D, E, scale):
        global A
        A = sp.diags(D).dot(A).dot(sp.diags(E))/scale

    sol = scs.solve(
        data, K, verbose=False, use_indirect=False,
        normalize=False,
        linsys_cbs=(
            init_lin_sys_work_cb,
            solve_lin_sys_cb, accum_by_a_cb, accum_by_atrans_cb,
            normalize_a_cb, un_normalize_a_cb,
        ),
        max_iters=int(1e5), eps=1e-5,
    )

    yield check_solution, np.dot(data['c'], sol['x']), p_star
    yield check_solution, np.dot(-data['b'], sol['y']), p_star

    sol = scs.solve(
        data, K, verbose=False, use_indirect=False,
        normalize=True,
        linsys_cbs=(
            init_lin_sys_work_cb,
            solve_lin_sys_cb, accum_by_a_cb, accum_by_atrans_cb,
            normalize_a_cb, un_normalize_a_cb,
        ),
        max_iters=int(1e5), eps=1e-5,
    )

    yield check_solution, np.dot(data['c'], sol['x']), p_star
    yield check_solution, np.dot(-data['b'], sol['y']), p_star

# for i,c in zip(range(4), test_python_linsys()):
#     print(i,c)
