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

incdir = include_directories('scs/include')
pyincdir = include_directories('src')
linsysinc = include_directories('scs/linsys')
indirectinc = include_directories('scs/linsys/cpu/indirect')

py.extension_module(
	'_scs_direct',
    'src/scspy.c',
	'scs/src/aa.c',
	'scs/src/cones.c',
	'scs/src/ctrlc.c',
	'scs/src/exp_cone.c',
	'scs/src/linalg.c',
	'scs/src/normalize.c',
	'scs/src/rw.c',
	'scs/src/scs_version.c',
	'scs/src/scs.c',
	'scs/src/util.c',
	'scs/linsys/scs_matrix.c',
	'scs/linsys/csparse.c',
	'scs/linsys/cpu/indirect/private.c',
	include_directories : [
		incdir, pyincdir, linsysinc, indirectinc, incdir_numpy],
    install: true,
)