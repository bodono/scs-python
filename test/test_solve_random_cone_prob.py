from __future__ import print_function, division
import scs
import numpy as np
from scipy import sparse
import gen_random_cone_prob as tools

#############################################
#  Uses scs to solve a random cone problem  #
#############################################

def import_error(msg):
  print()
  print('## IMPORT ERROR:' + msg)
  print()


try:
  import pytest
except ImportError:
  import_error('Please install pytest to run tests.')
  raise

flags = [(False, False), (True, False)]
try:
  import _scs_gpu
  flags += [(True, True)]
except ImportError:
  pass

np.random.seed(1)

# cone:
K = {
    'f': 10,
    'l': 15,
    'q': [5, 10, 0, 1],
    's': [3, 4, 0, 0, 1],
    'ep': 10,
    'ed': 10,
    'p': [-0.25, 0.5, 0.75, -0.33]
}
m = tools.get_scs_cone_dims(K)
params = {'normalize': True, 'scale': 5, 'cg_rate': 2}

@pytest.mark.parametrize("use_indirect,gpu", flags)
def test_solve_feasible(use_indirect, gpu):
  data, p_star = tools.gen_feasible(K, n=m // 3, density=0.01)

  sol = scs.solve(data, K, use_indirect=use_indirect, gpu=gpu, **params)
  x = sol['x']
  y = sol['y']
  print('p*  = ', p_star)
  print('pri error = ', (np.dot(data['c'], x) - p_star) / p_star)
  print('dual error = ', (-np.dot(data['b'], y) - p_star) / p_star)


@pytest.mark.parametrize("use_indirect,gpu", flags)
def test_solve_infeasible(use_indirect, gpu):
  data = tools.gen_infeasible(K, n=m // 3)
  sol = scs.solve(data, K, use_indirect=use_indirect, gpu=gpu, **params)


@pytest.mark.parametrize("use_indirect,gpu", flags)
def test_solve_unbounded(use_indirect, gpu):
  data = tools.gen_unbounded(K, n=m // 3)
  sol = scs.solve(data, K, use_indirect=use_indirect, gpu=gpu, **params)
