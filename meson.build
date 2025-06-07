project('scs', 'c')

py = import('python').find_installation(pure: false)

cc = meson.get_compiler('c')
fs = import('fs')
# Check for submodule before doing anything else
if not fs.exists('scs_source/README.md')
  error('Missing the `scs_source` submodule! Run `git submodule update --init` to fix this.')
endif

# Simpler check for numpy
incdir_numpy = run_command(py,['-c',
'''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
'''], check: true).stdout().strip()

# Get BLAS
blas_deps = []
if get_option('link_mkl')
    blas_deps = [cc.find_library('mkl_rt', required : false)]
    if not blas_deps[0].found()
        blas_deps = [dependency('mkl-sdl', required : false)]
    endif
else
    if host_machine.system() == 'darwin'
        blas_deps = [dependency('Accelerate')]
    else
        blas_deps = [cc.find_library('openblas', static: get_option('link_blas_statically'), required : false)]
        if not blas_deps[0].found()
            blas_deps = [dependency(['openblas', 'OpenBLAS'], static: get_option('link_blas_statically'), required : false)]
        endif
    endif

    # try to find blas/cblas (e.g., Linux)
    if not blas_deps[0].found()
        blas_deps = [cc.find_library('blas', static: get_option('link_blas_statically'), required : false)]
        if not blas_deps[0].found()
            blas_deps = [dependency('blas', static: get_option('link_blas_statically'), required : false)]
        endif
        lapack_dep = cc.find_library('lapack', static: get_option('link_blas_statically'), required : false)
        if not lapack_dep.found()
            lapack_dep = dependency('lapack', static: get_option('link_blas_statically'), required : false)
        endif
        if lapack_dep.found()
            blas_deps += lapack_dep
        endif
        cblas_dep = cc.find_library('cblas', static: get_option('link_blas_statically'), required : false)
        if not cblas_dep.found()
            cblas_dep = dependency('cblas', static: get_option('link_blas_statically'), required : false)
        endif
        if cblas_dep.found()
            blas_deps += cblas_dep
        endif
    endif
endif

# Since nothing above was required, we stop here if failed.
# When creating an sdist we are not compiling / linking, so don't need to fail.
if not blas_deps[0].found() and not get_option('sdist_mode')
    error('OpenBLAS or Netlib BLAS/CBLAS is required on all platforms, and was not found.')
endif

# Common
common_c_args = cc.get_supported_arguments('-Wno-unused-result') + [
  '-DPYTHON', '-DCTRLC=1', '-DUSE_LAPACK=1', '-DDLONG=1'
]
common_includes = [
  'scs',
  'scs_source/include',
  'scs_source/linsys',
  incdir_numpy
]
_deps = [blas_deps]
if get_option('use_openmp')
  _deps += dependency('openmp')
endif

# Sources
scs_core_sources = files(
  'scs_source/src/aa.c',
  'scs_source/src/cones.c',
  'scs_source/src/ctrlc.c',
  'scs_source/src/exp_cone.c',
  'scs_source/src/linalg.c',
  'scs_source/src/normalize.c',
  'scs_source/src/rw.c',
  'scs_source/src/scs_version.c',
  'scs_source/src/scs.c',
  'scs_source/src/util.c'
)

common_linsys_sources = files(
  'scs_source/linsys/scs_matrix.c',
  'scs_source/linsys/csparse.c'
)

amd_sources = files(
  'scs_source/linsys/external/amd/amd_1.c',
  'scs_source/linsys/external/amd/amd_2.c',
  'scs_source/linsys/external/amd/amd_aat.c',
  'scs_source/linsys/external/amd/amd_control.c',
  'scs_source/linsys/external/amd/amd_defaults.c',
  'scs_source/linsys/external/amd/amd_dump.c',
  'scs_source/linsys/external/amd/amd_global.c',
  'scs_source/linsys/external/amd/amd_info.c',
  'scs_source/linsys/external/amd/amd_order.c',
  'scs_source/linsys/external/amd/amd_post_tree.c',
  'scs_source/linsys/external/amd/amd_postorder.c',
  'scs_source/linsys/external/amd/amd_preprocess.c',
  'scs_source/linsys/external/amd/amd_valid.c',
  'scs_source/linsys/external/amd/SuiteSparse_config.c'
)

py.extension_module(
  '_scs_direct',
  'scs/scspy.c',
  'scs_source/linsys/cpu/direct/private.c',
  'scs_source/linsys/external/qdldl/qdldl.c',
  common_linsys_sources,
  scs_core_sources,
  amd_sources,
  c_args: common_c_args,
  include_directories: common_includes + [
    'scs_source/linsys/cpu/direct',
    'scs_source/linsys/external/qdldl',
    'scs_source/linsys/external/amd'
  ],
  dependencies: _deps,
  install: true,
)

py.extension_module(
  '_scs_indirect',
  'scs/scspy.c',
  'scs_source/linsys/cpu/indirect/private.c',
  common_linsys_sources,
  scs_core_sources,
  c_args: common_c_args + ['-DPY_INDIRECT', '-DINDIRECT=1'],
  include_directories: common_includes + ['scs_source/linsys/cpu/indirect'],
  dependencies: _deps,
  install: true,
)

if get_option('link_mkl')
  py.extension_module(
    '_scs_mkl',
    'scs/scspy.c',
    'scs_source/linsys/mkl/direct/private.c',
    common_linsys_sources,
    scs_core_sources,
    c_args: common_c_args + ['-DPY_MKL'],
    include_directories: common_includes + ['scs_source/linsys/mkl/direct'],
  dependencies: _deps,
    install: true,
  )
endif

py.install_sources('scs/__init__.py', subdir: 'scs')
