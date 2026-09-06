#!/usr/bin/env python
import enum
import sys
import numpy as np
from scipy import sparse
from scs import _scs_direct
import warnings

__version__ = _scs_direct.version()
__sizeof_int__ = _scs_direct.sizeof_int()
__sizeof_float__ = _scs_direct.sizeof_float()


# SCS return integers correspond to one of these flags:
# (copied from scs/include/glbopts.h)
INFEASIBLE_INACCURATE = -7  # SCS best guess infeasible
UNBOUNDED_INACCURATE = -6  # SCS best guess unbounded
SIGINT = -5  # interrupted by sig int
FAILED = -4  # SCS failed
INDETERMINATE = -3  # indeterminate (norm too small)
INFEASIBLE = -2  # primal infeasible, dual unbounded
UNBOUNDED = -1  # primal unbounded, dual infeasible
UNFINISHED = 0  # never returned, used as placeholder
SOLVED = 1  # problem solved to desired accuracy
SOLVED_INACCURATE = 2  # SCS best guess solved


class LinearSolver(enum.Enum):
  """Linear system solver backend for SCS.

  Pass a member (or its string value) as the `linear_solver` setting. Which
  members are importable depends on the build; `AUTO` picks the best one
  available on the platform.

  Example:

  >>> import scs
  >>> scs.LinearSolver.QDLDL.value
  'qdldl'
  >>> scs.LinearSolver("cpu_indirect") is scs.LinearSolver.CPU_INDIRECT
  True
  """
  AUTO = "auto"
  QDLDL = "qdldl"
  CPU_INDIRECT = "cpu_indirect"
  MKL = "mkl"
  ACCELERATE = "accelerate"
  CPU_DENSE = "cpu_dense"
  GPU_INDIRECT = "gpu_indirect"
  CUDSS = "cudss"


def _load_module(name):
  from importlib import import_module
  return import_module(f"scs.{name}")


def _resolve_auto():
  """Auto-detect the best available direct solver for this platform."""
  if sys.platform == "darwin":
    # Prefer the bundled QDLDL on macOS over Apple Accelerate.
    return _scs_direct
  try:
    return _load_module("_scs_mkl")
  except ImportError:
    pass
  return _scs_direct


_SOLVER_DISPATCH = {
    LinearSolver.AUTO: _resolve_auto,
    LinearSolver.QDLDL: lambda: _scs_direct,
    LinearSolver.CPU_INDIRECT: lambda: _load_module("_scs_indirect"),
    LinearSolver.MKL: lambda: _load_module("_scs_mkl"),
    LinearSolver.ACCELERATE: lambda: _load_module("_scs_accelerate"),
    LinearSolver.CPU_DENSE: lambda: _load_module("_scs_dense"),
    LinearSolver.GPU_INDIRECT: lambda: _load_module("_scs_gpu"),
    LinearSolver.CUDSS: lambda: _load_module("_scs_cudss"),
}


def _select_scs_module(stgs):
  """Choose which SCS C extension to import based on settings."""
  linear_solver = stgs.pop("linear_solver", LinearSolver.AUTO)
  if isinstance(linear_solver, str):
    linear_solver = LinearSolver(linear_solver)
  return _SOLVER_DISPATCH[linear_solver]()


def _has_lower_tri(P):
  """Fast check for strictly lower triangular entries in a sorted CSC matrix."""
  nnz_per_col = np.diff(P.indptr)
  nonempty = nnz_per_col > 0
  if not nonempty.any():
    return False
  last_row = P.indices[P.indptr[1:][nonempty] - 1]
  return bool(np.any(last_row > np.where(nonempty)[0]))


class SCS(object):
  """A solver instance bound to one problem's data and cone.

  Construct it once, then call `solve` as often as needed; `update` swaps in
  a new `b` or `c` without rebuilding the workspace.

  Example:

  >>> import numpy as np
  >>> import scipy.sparse as sp
  >>> import scs
  >>> # maximize x  subject to  0 <= x <= 1
  >>> data = {
  ...     "A": sp.csc_matrix(np.array([[1.0], [-1.0]])),
  ...     "b": np.array([1.0, 0.0]),
  ...     "c": np.array([-1.0]),
  ... }
  >>> solver = scs.SCS(data, {"l": 2}, verbose=False)
  >>> solver.solve()["info"]["status"]
  'solved'
  """

  def __init__(self, data, cone, **settings):
    """Initialize the SCS solver.

    @param data     Dictionary containing keys `P`, `A`, `b`, `c`.
    @param cone     Dictionary containing cone information.
    @param settings Settings as kwargs, see docs.

    Thread safety: construction is assumed to be thread-local. Calling
    `__init__` on a live SCS instance from another thread (i.e. while
    `solve` or `update` may be running on it) is undefined behavior.
    Use a fresh `SCS(...)` instance instead.
    """
    self._settings = settings
    if not data or not cone:
      raise ValueError("Missing data or cone information")

    if "b" not in data or "c" not in data:
      raise ValueError("Missing one of b, c from data dictionary")
    if "A" not in data:
      raise ValueError("Missing A from data dictionary")

    A = data["A"]
    b = data["b"]
    c = data["c"]

    if A is None or b is None or c is None:
      raise ValueError("Incomplete data specification")

    if not sparse.issparse(A):
      raise TypeError("A is required to be a sparse matrix")
    if not A.format == "csc":
      warnings.warn(
          "Converting A to a CSC (compressed sparse column) matrix;"
          " may take a while."
      )
      A = A.tocsc()

    # .todense() returns a 2-D np.matrix; the C layer requires ndim==1.
    # Flatten to a 1-D ndarray so a sparse b or c is actually accepted.
    if sparse.issparse(b):
      b = np.asarray(b.todense()).ravel()

    if sparse.issparse(c):
      c = np.asarray(c.todense()).ravel()

    m = len(b)
    n = len(c)

    # sorted_indices() returns a new matrix; sort_indices() would mutate
    # the caller's A in place (surprising, and a data race under the
    # free-threaded build if another thread reads the same matrix).
    if not A.has_sorted_indices:
      A = A.sorted_indices()
    Adata, Aindices, Acolptr = A.data, A.indices, A.indptr
    if A.shape != (m, n):
      raise ValueError("A shape not compatible with b,c")

    Pdata, Pindices, Pcolptr = None, None, None
    if "P" in data:
      P = data["P"]
      if P is not None:
        if not sparse.issparse(P):
          raise TypeError("P is required to be a sparse matrix")
        if P.shape != (n, n):
          raise ValueError("P shape not compatible with A,b,c")
        if not P.format == "csc":
          warnings.warn(
              "Converting P to a CSC (compressed sparse column) "
              "matrix; may take a while."
          )
          P = P.tocsc()
        # sorted_indices() returns a new matrix; see A above.
        if not P.has_sorted_indices:
          P = P.sorted_indices()
        # extract upper triangular component only
        if _has_lower_tri(P):
          P = sparse.triu(P, format="csc")
        Pdata, Pindices, Pcolptr = P.data, P.indices, P.indptr

    # Which scs are we using (scs_direct, scs_indirect, ...)
    _scs = _select_scs_module(self._settings)

    # Initialize solver
    self._solver = _scs.SCS(
        (m, n),
        Adata,
        Aindices,
        Acolptr,
        Pdata,
        Pindices,
        Pcolptr,
        b,
        c,
        cone,
        **self._settings,
    )

  def solve(self, warm_start=True, x=None, y=None, s=None):
    """Solve the optimization problem.

    @param warm_start   Whether to warm-start. By default the solution of
                        the previous problem is used as the warm-start. The
                        warm-start can be overridden to another value by
                        passing `x`, `y`, `s` args.
    @param x            Primal warm-start override.
    @param y            Dual warm-start override.
    @param s            Slack warm-start override.

    @return dictionary with solution with keys:
         'x' - primal solution
         's' - primal slack solution
         'y' - dual solution
         'info' - information dictionary (see docs)

    Example:

    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> import scs
    >>> # maximize x  subject to  0 <= x <= 1
    >>> data = {
    ...     "A": sp.csc_matrix(np.array([[1.0], [-1.0]])),
    ...     "b": np.array([1.0, 0.0]),
    ...     "c": np.array([-1.0]),
    ... }
    >>> solver = scs.SCS(data, {"l": 2}, eps_abs=1e-9, eps_rel=1e-9,
    ...                  verbose=False)
    >>> sol = solver.solve()
    >>> sorted(sol)
    ['info', 's', 'x', 'y']
    >>> sol["info"]["status"]
    'solved'
    >>> float(np.round(sol["x"][0], 6))
    1.0
    """
    return self._solver.solve(warm_start, x, y, s)

  def update(self, b=None, c=None):
    """Update the `b` vector, `c` vector, or both, before another solve.

    After a solve we can reuse the SCS workspace in another solve if the
    only problem data that has changed are the `b` and `c` vectors.

    @param  b   New `b` vector.
    @param  c   New `c` vector.

    Example:

    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> import scs
    >>> data = {
    ...     "A": sp.csc_matrix(np.array([[1.0], [-1.0]])),
    ...     "b": np.array([1.0, 0.0]),
    ...     "c": np.array([-1.0]),
    ... }
    >>> solver = scs.SCS(data, {"l": 2}, eps_abs=1e-9, eps_rel=1e-9,
    ...                  verbose=False)
    >>> float(np.round(solver.solve()["x"][0], 6))   # maximize x
    1.0
    >>> solver.update(c=np.array([1.0]))             # now minimize x
    >>> float(np.round(abs(solver.solve()["x"][0]), 6))
    0.0
    """
    self._solver.update(b, c)


def solve(data, cone, **settings):
  """Solve a problem in one call, without holding on to a solver instance.

  Backwards compatible helper that simply calls the main API. Warm-start
  vectors are read out of `data` under the keys `x`, `y` and `s`, which is
  how the pre-`SCS`-class API passed them.

  @param data     Dictionary containing keys `A`, `b`, `c`, optionally `P`,
                  and optionally the warm-start vectors `x`, `y`, `s`.
  @param cone     Dictionary containing cone information.
  @param settings Settings as kwargs, see docs.

  @return the same dictionary `SCS.solve` returns.

  Example:

  >>> import numpy as np
  >>> import scipy.sparse as sp
  >>> import scs
  >>> data = {
  ...     "A": sp.csc_matrix(np.array([[1.0], [-1.0]])),
  ...     "b": np.array([1.0, 0.0]),
  ...     "c": np.array([-1.0]),
  ... }
  >>> sol = scs.solve(data, {"l": 2}, eps_abs=1e-9, eps_rel=1e-9,
  ...                 verbose=False)
  >>> sol["info"]["status"]
  'solved'
  >>> float(np.round(sol["x"][0], 6))
  1.0
  """
  solver = SCS(data, cone, **settings)

  # Hack out the warm start data from old API
  x = y = s = None
  if "x" in data:
    x = data["x"]
  if "y" in data:
    y = data["y"]
  if "s" in data:
    s = data["s"]

  return solver.solve(warm_start=True, x=x, y=y, s=s)
