# nost test suite copied initially from ECOS project
from __future__ import print_function
import platform

import numpy as np
import scs
import scipy.sparse as sp

c = np.array([-1.0])
b = np.array([1.0, -0.0])
A = sp.csc_matrix([1.0, -1.0]).T.tocsc()
P = sp.csc_matrix([1.2345]).tocsc()
data = {"A": A, "P": P, "b": b, "c": c}
cone = {"q": [], "l": 2}

sol = scs.solve(data, cone, use_indirect=False)
print(sol)
sol = scs.solve(data, cone, use_indirect=True)
print(sol)
