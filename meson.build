project('scs', 'c')

py = import('python').find_installation(pure: false)

# This uses the path as is, and avoids running the interpreter.
incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
incdir = np.get_include()
print(incdir)
'''
    ],
    check: true
  ).stdout().strip()

incdir = include_directories('scs_source/include')
pyincdir = include_directories('src')
linsysinc = include_directories('scs_source/linsys')
directinc = include_directories('scs_source/linsys/cpu/direct')
qdldlinc = include_directories('scs_source/linsys/external/qdldl')
amdinc = include_directories('scs_source/linsys/external/amd')


# meson doesn't support wildcards 
# https://mesonbuild.com/FAQ.html#why-cant-i-specify-target-files-with-a-wildcard

py.extension_module(
	'_scs_direct',
	
    'src/scspy.c',
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
		incdir, pyincdir, linsysinc, directinc, qdldlinc, amdinc, incdir_numpy],
    install: true,
)