"""An ILP64 build must refuse to run in a process whose MKL (here NumPy's)
is already LP64: the core's interface-layer guard fails construction, and
prints why to stdout (the workflow greps for it)."""
import numpy as np, scipy.sparse as sp, scs

d = dict(P=sp.eye(2, format="csc"), A=sp.eye(2, format="csc"), b=np.ones(2), c=np.ones(2))
try:
    scs.SCS(d, dict(l=2), linear_solver=scs.LinearSolver.MKL, verbose=True)
except ValueError as e:
    print("guard refused construction:", e)
else:
    raise SystemExit("ILP64 build initialized against LP64 numpy-MKL: guard failed")
