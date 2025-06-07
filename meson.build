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
  '-DPYTHON', '-DCTRLC=1'
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

is_linux = host_machine.system() == 'linux'
if is_linux
  _deps += cc.find_library('rt', required : true)
endif

if get_option('use_lapack')
  common_c_args += '-DUSE_LAPACK=1'
endif

if get_option('use_singleprec')
  common_c_args += '-DSFLOAT=1'
endif

if get_option('use_blas64')
  common_c_args += '-DBLAS64=1'
endif

if get_option('use_extraverbose')
  common_c_args += '-DVERBOSITY=999'
endif

is_gpu_build = get_option('use_gpu')
use_32bit_ints = get_option('int32')

if is_gpu_build and not use_32bit_ints
  error('GPU builds require 32-bit integers. Please re-run Meson with -Dint32=true')
endif

if not use_32bit_ints
  common_c_args += '-DDLONG=1'
endif

cuda_dep = dependency('cuda', required: is_gpu_build)
if is_gpu_build and not cuda_dep.found()
  error('GPU build requested but CUDA toolkit was not found.')
endif

if is_gpu_build
  gpu_c_args = c_args + ['-DPY_GPU=1', '-DINDIRECT=1']
  deps +=
  if get_option('gpu_atrans')
    gpu_c_args += '-DGPU_TRANSPOSE_MAT=1'
  endif
  ext_modules += py.extension_module(
    '_scs_gpu',
    common_sources,
    # In Meson, sources with a `.cu` extension are compiled with nvcc by default.
    # To compile `.c` files with nvcc, they must be explicitly targeted.
    # It is strongly recommended to rename CUDA-C files to `.cu`.
    fs.glob('scs_source/linsys/gpu/*.c'),
    fs.glob('scs_source/linsys/gpu/indirect/*.c'),
    c_args: gpu_c_args,
    dependencies: [blas_dep, cuda_dep],
    include_directories: [
      common_includes, 'scs_source/linsys/gpu/', 'scs_source/linsys/gpu/indirect'
    ],
    install: true,
  )
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

# Build modules
scs_dir = py.get_install_dir() / 'scs'
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
  install_dir: scs_dir,
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
  install_dir: scs_dir,
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
    install_dir: scs_dir,
    install: true,
  )
endif

py.install_sources('scs/py/__init__.py', subdir: 'scs')
