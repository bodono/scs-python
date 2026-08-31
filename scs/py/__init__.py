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


def _preload_intel_mkl():
  """Preload MKL from Intel's official ``mkl`` PyPI wheels (``scs[mkl]``).

  The pre-built Linux x86-64 wheels link the _scs_mkl extension against MKL
  without vendoring it: MKL loads its CPU dispatch kernels via dlopen, which
  wheel-repair tools cannot see, so a vendored MKL is incomplete and aborts
  the process at solve time (cvxgrp/scs#423). The ``scs[mkl]`` extra instead
  installs Intel's own ``mkl`` and ``intel-openmp`` wheels into
  ``<prefix>/lib``.

  The primary lookup mechanism is a ``$ORIGIN``-relative RUNPATH baked into
  the wheel's extension (site-packages/scs is four levels below the prefix
  in every standard layout), which lets the loader resolve MKL's mutually
  referencing component libraries as one group. This fallback covers
  non-standard layouts where that relative path misses: it dlopens the
  libraries RTLD_LAZY | RTLD_LOCAL so the extension's DT_NEEDED entries
  resolve from the link map. LAZY is required -- the components cannot be
  eagerly bound one at a time -- and ctypes.CDLL always forces RTLD_NOW,
  so this calls dlopen(3) directly. RTLD_LOCAL keeps MKL's BLAS from
  interposing on other libraries (e.g. NumPy's vendored OpenBLAS).
  Returns True if anything was loaded.
  """
  if not sys.platform.startswith("linux"):
    return False
  import ctypes
  import glob
  import os
  from importlib import metadata

  libdirs = []
  for pkg in ("mkl", "intel-openmp"):
    try:
      dist = metadata.distribution(pkg)
    except metadata.PackageNotFoundError:
      continue
    for f in dist.files or ():
      if f.name.startswith(("libmkl_", "libiomp5")):
        d = os.path.dirname(os.fspath(dist.locate_file(f)))
        if d not in libdirs and os.path.isdir(d):
          libdirs.append(d)
  if not libdirs:
    # Fallback for installers that do not record RECORD data files.
    for prefix in dict.fromkeys((sys.prefix, sys.base_prefix, sys.exec_prefix)):
      d = os.path.join(prefix, "lib")
      if glob.glob(os.path.join(d, "libmkl_core.so*")):
        libdirs.append(d)
  if not libdirs:
    return False

  # Dependency-safe order: OpenMP runtime, then MKL core, threading layer,
  # interface layer, and the single-dynamic-library runtime (used by the
  # extension's interface-layer check).
  patterns = (
      "libiomp5.so",
      "libmkl_core.so*",
      "libmkl_sequential.so*",
      "libmkl_intel_thread.so*",
      "libmkl_intel_lp64.so*",
      "libmkl_intel_ilp64.so*",
      "libmkl_rt.so*",
  )
  dlopen = ctypes.CDLL(None).dlopen
  dlopen.restype = ctypes.c_void_p
  dlopen.argtypes = (ctypes.c_char_p, ctypes.c_int)
  loaded = False
  for pattern in patterns:
    for d in libdirs:
      for path in sorted(glob.glob(os.path.join(d, pattern))):
        if dlopen(os.fsencode(path), os.RTLD_LAZY | os.RTLD_LOCAL):
          loaded = True
  return loaded


def _load_module(name):
  from importlib import import_module
  try:
    return import_module(f"scs.{name}")
  except ImportError:
    # The wheel _scs_mkl extension resolves MKL from the `mkl` PyPI package
    # (scs[mkl]) rather than vendored libraries; make those loadable and
    # retry. Without them the ImportError propagates and AUTO falls back.
    if name != "_scs_mkl" or not _preload_intel_mkl():
      raise
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
