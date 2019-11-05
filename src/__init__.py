#!/usr/bin/env python
from __future__ import print_function
from warnings import warn
from scipy import sparse
import _scs_direct

__version__ = _scs_direct.version()
__sizeof_int__ = _scs_direct.sizeof_int()
__sizeof_float__ = _scs_direct.sizeof_float()

_USE_INDIRECT_DEFAULT = False


def solve(probdata, cone, **kwargs):
  """Solves convex cone problems.

    @return dictionary with solution with keys:
         'x' - primal solution
         's' - primal slack solution
         'y' - dual solution
         'info' - information dictionary
  """
  if not probdata or not cone:
    raise TypeError('Missing data or cone information')

  if 'b' not in probdata or 'c' not in probdata:
    raise TypeError('Missing one or more of b, c from data dictionary')

  b = probdata['b']
  c = probdata['c']

  m = len(b)
  n = len(c)

  warm = {}
  if 'x' in probdata:
    warm['x'] = probdata['x']
  if 'y' in probdata:
    warm['y'] = probdata['y']
  if 's' in probdata:
    warm['s'] = probdata['s']

  if b is None or c is None:
    raise TypeError('Incomplete data specification')

  linsys_cbs = kwargs.get('linsys_cbs', None)
  if linsys_cbs:
    # Create an empty placeholder A matrix that is never used.
    A = sparse.csc_matrix((m,n))
  else:
    if 'A' not in probdata:
      raise TypeError('Missing A from data dictionary')
    A = probdata['A']

    if not sparse.issparse(A):
      raise TypeError('A is required to be a sparse matrix')
    if not sparse.isspmatrix_csc(A):
      warn('Converting A to a CSC (compressed sparse column) matrix; may take a '
           'while.')
      A = A.tocsc()

  if sparse.issparse(b):
    b = b.todense()

  if sparse.issparse(c):
    c = c.todense()

  Adata, Aindices, Acolptr = A.data, A.indices, A.indptr
  if kwargs.pop('gpu', False):  # False by default
    if not kwargs.pop('use_indirect', _USE_INDIRECT_DEFAULT):
      raise NotImplementedError(
          'GPU direct solver not yet available, pass `use_indirect=True`.')
    import _scs_gpu
    return _scs_gpu.csolve((m, n), Adata, Aindices, Acolptr, b, c, cone, warm,
                           **kwargs)

  if kwargs.pop('use_indirect', _USE_INDIRECT_DEFAULT):
    import _scs_indirect
    return _scs_indirect.csolve((m, n), Adata, Aindices, Acolptr, b, c, cone,
                                warm, **kwargs)

  if linsys_cbs:
    import _scs_python
    return _scs_python.csolve(
        (m, n), Adata, Aindices, Acolptr, b, c, cone,
        warm, **kwargs)

  return _scs_direct.csolve((m, n), Adata, Aindices, Acolptr, b, c, cone, warm,
                            **kwargs)
