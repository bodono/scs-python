# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`scs-python` is a Python wrapper around the SCS (Splitting Conic Solver) C library — a high-performance solver for large-scale convex cone programming. It exposes multiple C extension modules built from a git submodule (`scs_source/`), selectable at build time.

## Build Commands

The project uses Meson via the `meson-python` PEP 517 backend, with Pixi for the dev environment.

```bash
# Install (default CPU direct solver)
pip install -v . --no-build-isolation

# Install with OpenMP
pip install -v . --no-build-isolation -Csetup-args=-Duse_openmp=true

# Install with Intel MKL backend
pip install -v . --no-build-isolation -Csetup-args=-Dlink_mkl=true

# Install with spectral cone support (logdet, nuclear norm, ell1, sum-of-largest)
pip install -v . --no-build-isolation -Csetup-args=-Duse_spectral_cones=true

# Build source distribution
pipx run build --sdist -Csetup-args=-Dsdist_mode=true
```

With Pixi:
```bash
pixi run python -m pip install -v . --no-build-isolation
```

## Testing

```bash
pytest test/

# Run a single test file
pytest test/test_scs_basic.py

# Run a specific test
pytest test/test_scs_basic.py::test_name
```

MKL, GPU, and dense tests (`test_solve_random_cone_prob_mkl.py`, `test_solve_random_cone_prob_cudss.py`, `test_solve_random_cone_prob_dense.py`) require the corresponding build variants. Spectral cone tests (`test_spectral_and_complex_cones.py`) require `-Duse_spectral_cones=true`.

## Architecture

### Layered Structure

```
Python API (scs/py/__init__.py)
    └── SCS class + legacy solve() function
        └── Dynamically selected C extension module
            ├── _scs_direct   (always built — direct solver via QDLDL)
            ├── _scs_indirect (always built — iterative/CG solver)
            ├── _scs_dense    (built if use_lapack=true — dense LU via LAPACK)
            ├── _scs_mkl      (built if link_mkl=true)
            ├── _scs_gpu      (built if use_gpu=true, requires int32=true + CUDA)
            └── _scs_cudss    (built if link_cudss=true, requires int32=true + cuDSS)
                └── scs_source/ (git submodule — core SCS C library)
```

### Key Files

- `scs/py/__init__.py` — All Python-layer logic: input validation, sparse matrix conversion (enforces CSC format), module selection, warm-start handling, and result post-processing.
- `scs/scspy.c` + `scs/include/scsmodule.h` + `scs/include/scsobject.h` — Thin C wrapper that bridges Python/NumPy to the SCS C API.
- `scs/scsobject.h` — C-level cone dict parsing and SCS workspace init/solve/dealloc.
- `meson.build` / `meson.options` — Build configuration and build options that control which extension modules are compiled.
- `scs_source/` — Git submodule containing the full SCS C library: `src/` (ADMM loop, cone projections, Anderson acceleration), `linsys/` (pluggable linear system solvers), `include/`, `external/` (AMD ordering, QDLDL).

### Data Flow

1. User passes `data` dict (`A`, `b`, `c`, optional `P`) and `cone` dict to `scs.SCS()` or `scs.solve()`.
2. Python layer validates and converts `A`/`P` to sparse CSC, converts `b`/`c` to dense float arrays.
3. The appropriate C extension is selected based on settings (`use_indirect`, `gpu`, `mkl`, `cudss`, `dense`).
4. `.solve()` (with optional warm-start vectors `x`, `y`, `s`) calls into the C extension.
5. The C library runs ADMM iterations with Anderson acceleration, delegates linear system solves to the configured backend, and returns a dict with keys `x`, `s`, `y`, `info`.

### Solver Variants

The Python module selection logic in `scs/py/__init__.py` loads the right `_scs_*` module at runtime based on which build options were enabled at compile time and which runtime settings are passed.

## Important Notes

- `scs_source/` is a git submodule — run `git submodule update --init` after cloning.
- The C extension modules are compiled objects; any change to `scs_source/` or the C wrapper requires a rebuild (`pip install -v . --no-build-isolation`).
- `A` must be a sparse CSC matrix; `P` (optional quadratic term) must be sparse CSC upper triangular. This is enforced in Python before passing to C.
- GPU variants require `int32=true` at build time.
- macOS uses Accelerate BLAS automatically; Linux defaults to OpenBLAS; Windows detects MKL.
