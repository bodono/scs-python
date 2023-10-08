project('scs', 'c')


# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py3_dep = py3.dependency()

# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

# We do need an absolute path to feed to `cc.find_library` below
_incdir_numpy_abs = run_command(py3,
['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
check: true
).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif

py = import('python').find_installation(pure: false)


incdir = include_directories('scs/include')
pyincdir = include_directories('src')


py.extension_module(
	'scs',
    'src/scspy.c',
	include_directories : [incdir, pyincdir, _incdir_numpy_abs],
    install: true,
)