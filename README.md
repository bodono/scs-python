scs-python
===

[![Build Status](https://github.com/bodono/scs-python/actions/workflows/build.yml/badge.svg)](https://github.com/bodono/scs-python/actions/workflows/build.yml)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://www.cvxgrp.org/scs/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/scs.svg?label=PyPI%20downloads&cacheSeconds=86400)](https://pypistats.org/packages/scs)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/scs.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/scs)

Python interface for [SCS](https://github.com/cvxgrp/scs) 3.0.0 and higher.
The full documentation is available [here](https://www.cvxgrp.org/scs/).

## Installation

```bash
pip install scs
```

To install from source:
```bash
git clone --recursive https://github.com/bodono/scs-python.git
cd scs-python
pip install .
```

### Linear solver backends

SCS supports several linear solver backends. The default is `AUTO`, which
selects the best available solver for the platform:
- **macOS**: Apple Accelerate if available, otherwise QDLDL
- **Linux / Windows**: MKL Pardiso if available, otherwise QDLDL

```python
# Auto-detect best backend (default)
solver = scs.SCS(data, cone)

# Explicitly select a solver
solver = scs.SCS(data, cone, linear_solver=scs.LinearSolver.QDLDL)
```

Available values: `AUTO`, `QDLDL`, `INDIRECT`, `MKL`, `ACCELERATE`, `DENSE`,
`GPU`, `CUDSS`.

The pre-built wheels (`pip install scs`) include MKL on x86_64 Linux and
Windows, and Apple Accelerate on macOS. When installing from source, additional
backends can be enabled with build-time flags:

```bash
# MKL Pardiso direct solver
pip install . -Csetup-args=-Dlink_mkl=true

# Use 64-bit BLAS/LAPACK integers (ILP64 / BLAS64)
pip install . -Csetup-args=-Duse_blas64=true

# GPU direct solver (cuDSS)
pip install . -Csetup-args=-Dlink_cudss=true -Csetup-args=-Dint32=true

# Dense direct solver (LAPACK)
pip install . -Csetup-args=-Duse_lapack=true

# Spectral cones (logdet, nuclear norm, ell-1, sum-of-largest)
pip install . -Csetup-args=-Duse_spectral_cones=true
```

Notes:
- Linux x86_64 wheels are built and tested against threaded MKL, and CI asserts a `libiomp5` dependency on the packaged `_scs_mkl` extension. Windows currently falls back to sequential MKL because Intel's conda `pkg-config` metadata for the threaded variant is still broken.
- `BLAS64` is a general SCS build mode for ILP64 BLAS/LAPACK libraries, not an MKL-only feature.
- For the MKL Pardiso backend specifically, `BLAS64` must be paired with 64-bit SCS integers (`DLONG` / `int32=false`), and SCS now fails early if another library in the process has already fixed MKL to an incompatible LP64/ILP64 interface layer.

## Usage

```python
import numpy as np
import scipy.sparse as sp
import scs

m, n = 4, 2
A = sp.random(m, n, density=0.5, format="csc")
b = np.random.randn(m)
c = np.random.randn(n)
P = sp.eye(n, format="csc")

cone = {"l": m}  # non-negative cone
data = {"P": P, "A": A, "b": b, "c": c}

solver = scs.SCS(data, cone, verbose=False)
sol = solver.solve()

print(sol["info"]["status"])  # 'solved'
print(sol["x"])               # primal solution
```

### Cone types

The `cone` dict supports the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `z` | `int` | Zero cone |
| `l` | `int` | Non-negative cone |
| `bu`, `bl` | `array` | Box cone bounds |
| `q` | `list[int]` | Second-order cone lengths |
| `s` | `list[int]` | PSD cone matrix dimensions |
| `cs` | `list[int]` | Complex PSD cone matrix dimensions |
| `ep` | `int` | Primal exponential cone triples |
| `ed` | `int` | Dual exponential cone triples |
| `p` | `list[float]` | Power cone parameters |

With `-Duse_spectral_cones=true`:

| Key | Type | Description |
|-----|------|-------------|
| `d` | `list[int]` | Log-determinant cone matrix dimensions |
| `nuc_m`, `nuc_n` | `list[int]` | Nuclear norm cone row/column dimensions |
| `ell1` | `list[int]` | ell-1 norm cone dimensions |
| `sl_n`, `sl_k` | `list[int]` | Sum-of-largest-eigenvalues dimensions and k values |

See the [cone documentation](https://www.cvxgrp.org/scs/api/cones.html) for
mathematical definitions and data layout details.

## Testing

```bash
pip install pytest
pytest test/
```
