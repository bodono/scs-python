project('scs', 'c')

py = import('python').find_installation(pure: false)

cc = meson.get_compiler('c')
blas_deps = []
if host_machine.system() == 'darwin'
  blas_deps = [dependency('Accelerate')]
else
  blas_deps = [cc.find_library('openblas', static: get_option('link_blas_statically'), required : false)]
endif
if not blas_deps[0].found()
    blas_deps = [cc.find_library('blas')]
    cblas_dep = cc.find_library('cblas', static: get_option('link_blas_statically'), required : false)
    if cblas_dep.found()
        blas_deps += cblas_dep
    endif
endif

fs = import('fs')
if not fs.exists('scs_source/README.md')
  error('Missing the `scs_source` submodule! Run `git submodule update --init` to fix this.')
endif

# This uses the path as is, and avoids running the interpreter.
incdir_numpy = run_command(py,['-c',
'''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
'''], check: true).stdout().strip()

# rw.c emits a lot of -Wunused-result warnings, silence them for now:
c_args = cc.get_supported_arguments('-Wno-unused-result')

# meson doesn't support wildcards 
# https://mesonbuild.com/FAQ.html#why-cant-i-specify-target-files-with-a-wildcard

py.extension_module(
    '_scs_direct',

    'scs/scspy.c',
    'scs_source/linsys/cpu/direct/private.c',

    # scs_source/src:
    'scs_source/src/aa.c',
    'scs_source/src/cones.c',
    'scs_source/src/ctrlc.c',
    'scs_source/src/exp_cone.c',
    'scs_source/src/linalg.c',
    'scs_source/src/normalize.c',
    'scs_source/src/rw.c',
    'scs_source/src/scs_version.c',
    'scs_source/src/scs.c',
    'scs_source/src/util.c',

    # scs_source/linsys:
    'scs_source/linsys/scs_matrix.c',
    'scs_source/linsys/csparse.c',

    # scs_source/linsys/external/qdldl:
    'scs_source/linsys/external/qdldl/qdldl.c',

    # scs_source/linsys/external/amd:
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
    'scs_source/linsys/external/amd/SuiteSparse_config.c',

    include_directories : [
        'scs',
        'scs_source/include',
        'scs_source/linsys',
        'scs_source/linsys/cpu/direct',
        'scs_source/linsys/external/qdldl',
        'scs_source/linsys/external/amd',
        incdir_numpy],
    install: true,
    c_args: c_args + ['-DPYTHON', '-DCTRLC=1', '-DUSE_LAPACK=1', '-DDLONG=1'],
    dependencies: blas_deps,
)

py.extension_module(
    '_scs_indirect',

    'scs/scspy.c',
    'scs_source/linsys/cpu/indirect/private.c',

    # scs_source/src:
    'scs_source/src/aa.c',
    'scs_source/src/cones.c',
    'scs_source/src/ctrlc.c',
    'scs_source/src/exp_cone.c',
    'scs_source/src/linalg.c',
    'scs_source/src/normalize.c',
    'scs_source/src/rw.c',
    'scs_source/src/scs_version.c',
    'scs_source/src/scs.c',
    'scs_source/src/util.c',

    # scs_source/linsys:
    'scs_source/linsys/scs_matrix.c',
    'scs_source/linsys/csparse.c',

    include_directories : [
        'scs',
        'scs_source/include',
        'scs_source/linsys',
        'scs_source/linsys/cpu/indirect',
        incdir_numpy],
    install: true,
    c_args: c_args + ['-DPYTHON', '-DCTRLC=1', '-DPY_INDIRECT', '-DINDIRECT=1',
         '-DUSE_LAPACK=1', '-DDLONG=1'],
    dependencies: blas_deps,
)

py.install_sources('scs/__init__.py', subdir: 'scs')
