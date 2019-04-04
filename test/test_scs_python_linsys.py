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
  for i in range(num_probs):
    data, p_star = tools.gen_feasible(K, n=m // 3, density=0.1)

    A = data['A']
    ncon_cone, nz_cone = A.shape
    rho_x = 1e-3
    M = sp.bmat([[rho_x*sp.eye(nz_cone), A.T], [A, -sp.eye(ncon_cone)]])
    Msolve = sla.factorized(M)

    def solve_lin_sys_cb(b, s, i):
        b[:] = Msolve(b)

    def accum_by_a_cb(x, y):
        y += A.dot(x)

    def accum_by_atrans_cb(x, y):
        y += A.T.dot(x)

    sol = scs.solve(
        data, K, verbose=True, use_indirect=False,
        normalize=False, rho_x=rho_x,
        linsys_cbs=(solve_lin_sys_cb,accum_by_a_cb,accum_by_atrans_cb),
        max_iters=int(1e5), eps=1e-5,
    )

    yield check_solution, np.dot(data['c'], sol['x']), p_star
    yield check_solution, np.dot(-data['b'], sol['y']), p_star
