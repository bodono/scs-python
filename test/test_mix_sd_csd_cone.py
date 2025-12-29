import numpy as np
import scs
import scipy
import pytest
    
def gen_feasible(m, n, p_scale = 0.1):
    P = p_scale * scipy.sparse.eye(n, format="csc")   
    A = scipy.sparse.random(m, n, density=0.05, format="csc")
    c = np.random.randn(n)
    b = np.random.randn(m)

    return (P, A, b, c)


@pytest.mark.parametrize("use_indirect", [False, True])
def test_mix_sd_csd_cones(use_indirect):
    seed = 1234
    np.random.seed(seed)
    cone = dict(z=1, l=2, s=[3, 4], cs=[5, 4])
    m = int(cone['z'] + cone['l'] + sum([j * (j+1) / 2 for j in cone['s']]) + sum([j * j for j in cone['cs']]))
    n = m
    (P, A, b, c) = gen_feasible(m, n)
    probdata = dict(P=P, A=A, b=b, c=c)
    sol = scs.solve(probdata, cone, use_indirect=use_indirect)
    np.testing.assert_equal(sol['info']['status'], 'solved')