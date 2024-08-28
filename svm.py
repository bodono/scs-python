import scs 
import numpy as np
import scipy as sp
from scipy import sparse
import time
import osqp

# Generate problem data
np.random.seed(123)
n = 50
m = 5000
N = int(m / 2)
gamma = 1.0
eps = 1e-11
b = np.hstack([np.ones(N), -np.ones(N)])
A_upp = sparse.random(N, n, density=0.5)
A_low = sparse.random(N, n, density=0.5)
Ad = sparse.vstack([
        A_upp / np.sqrt(n) + (A_upp != 0.).astype(float) / n,
        A_low / np.sqrt(n) - (A_low != 0.).astype(float) / n
     ], format='csc')

Im = sparse.eye(m)
P = sparse.block_diag([sparse.eye(n), sparse.csc_matrix((m, m))], format='csc')
q = np.hstack([np.zeros(n), gamma*np.ones(m)])
A = sparse.bmat([[sparse.diags(b)@Ad, -Im],
                 [None,                Im]], format='csc')
l = np.hstack([-np.inf*np.ones(m), np.zeros(m)])
u = np.hstack([-np.ones(m), np.inf*np.ones(m)])

data = dict(A=A, P=P, c=q, b=np.zeros(A.shape[0]))
cone = dict(lower=-u, upper=-l)

start = time.time()
# Create an SCS object
prob = scs.SCS(data, cone, eps_abs=eps, eps_rel=eps, verbose=False)
# Solve problem
res_scs = prob.solve()
print(f"time SCS {time.time() - start}")

start = time.time()
# Create an OSQP object
prob = osqp.OSQP()
# Setup workspace
prob.setup(P, q, A, l, u, eps_abs=eps, eps_rel=eps, max_iter=int(1e6), verbose=False)
# Solve problem
res_osqp = prob.solve()
print(f"time OSQP {time.time() - start}")

print("differences in primal/dual sols:")
print(np.linalg.norm(res_scs['x'] - res_osqp.x))
print(np.linalg.norm(res_scs['y'] - res_osqp.y))
