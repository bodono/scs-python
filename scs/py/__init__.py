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
  """Linear system solver backend for SCS."""
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


# The boolean selection flags SCS accepted before 3.3.0. They were consumed
# here, so this is also where they are recognised on the way out.
_LEGACY_SOLVER_FLAGS = ("cudss", "gpu", "mkl", "use_indirect")


def _pop_legacy_solver_flags(stgs, explicit_linear_solver=None):
  """Translate the pre-3.3.0 boolean selection flags into a `LinearSolver`.

  Returns `None` when no legacy flag was passed. Otherwise the flags are
  removed from `stgs`, one `DeprecationWarning` naming the replacement is
  emitted, and the equivalent enum member is returned.

  When `explicit_linear_solver` is given the caller already chose a backend,
  so the flags are consumed and warned about but not translated: an explicit
  `linear_solver` wins even over a flag combination that was never valid.

  The mapping reproduces what 3.2.11 did, including its two surprises:
  `gpu=True` without `use_indirect` selected cuDSS rather than the CUDA
  indirect solver, and `mkl=True` with `use_indirect=True` was refused.
  The one deliberate difference is `cudss=True`, which 3.2.11 rejected
  outright ("To use cuDSS set gpu=True and use_indirect=False.") and which
  now maps to the backend its name plainly means.
  """
  present = {k: stgs.pop(k) for k in _LEGACY_SOLVER_FLAGS if k in stgs}
  if not present:
    return None

  named = ", ".join("`%s`" % k for k in sorted(present))
  if explicit_linear_solver is not None:
    warnings.warn(
        "The boolean solver-selection flags were removed in SCS 3.3.0."
        " You passed %s; they are ignored because `linear_solver` was given."
        % named,
        DeprecationWarning,
        stacklevel=4,
    )
    return None

  use_indirect = bool(present.get("use_indirect", False))
  if present.get("cudss"):
    replacement = LinearSolver.CUDSS
  elif present.get("gpu"):
    replacement = (
        LinearSolver.GPU_INDIRECT if use_indirect else LinearSolver.CUDSS
    )
  elif present.get("mkl"):
    if use_indirect:
      raise NotImplementedError(
          "There is no MKL indirect solver; pass"
          " `linear_solver=scs.LinearSolver.CPU_INDIRECT` for the iterative"
          " solver, or `LinearSolver.MKL` for MKL Pardiso."
      )
    replacement = LinearSolver.MKL
  else:
    replacement = (
        LinearSolver.CPU_INDIRECT if use_indirect else LinearSolver.QDLDL
    )

  warnings.warn(
      "The boolean solver-selection flags were removed in SCS 3.3.0."
      " You passed %s; pass `linear_solver=scs.LinearSolver.%s` instead."
      % (named, replacement.name),
      DeprecationWarning,
      # _pop_legacy_solver_flags <- _select_scs_module <- SCS.__init__ <- user
      stacklevel=4,
  )
  return replacement


def _select_scs_module(stgs):
  """Choose which SCS C extension to import based on settings."""
  # An explicit `linear_solver` wins: the legacy flags only supply a default,
  # and are not translated at all (so a never-valid combination such as
  # mkl=True, use_indirect=True cannot raise) when a choice was made.
  explicit = stgs.pop("linear_solver", None)
  legacy = _pop_legacy_solver_flags(stgs, explicit)
  linear_solver = explicit if explicit is not None else (legacy or LinearSolver.AUTO)
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
    """
    return self._solver.solve(warm_start, x, y, s)

  def update(self, b=None, c=None):
    """Update the `b` vector, `c` vector, or both, before another solve.

    After a solve we can reuse the SCS workspace in another solve if the
    only problem data that has changed are the `b` and `c` vectors.

    @param  b   New `b` vector.
    @param  c   New `c` vector.
    """
    self._solver.update(b, c)


# Backwards compatible helper function that simply calls the main API.
def solve(data, cone, **settings):
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
