scs-python
===

[![Build Status](https://travis-ci.org/bodono/scs-python.svg?branch=master)](https://travis-ci.org/bodono/scs-python)
[![Build status](https://ci.appveyor.com/api/projects/status/ep9hnbkgd70t9yv1/branch/master?svg=true)](https://ci.appveyor.com/project/bodono/scs-python/branch/master)

Python interface for [SCS](https://github.com/cvxgrp/scs) 2.0.0 and higher.


To install using pip (recommended) use:
```shell
pip install scs
```
To install SCS from source:
```shell
git clone --recursive https://github.com/bodono/scs-python.git
cd scs-python
python setup.py install
```
You may need `sudo` privileges for a global installation. Running SCS requires
numpy and scipy to be installed. You can install the gpu interface using
```shell
python setup.py install --scs --gpu
```

To test that SCS installed correctly and you have `nose` installed, run
```shell
nosetests
```
you can also solve a random cone problem (without `nose`) using
```shell
python test/solve_random_cone_prob.py
```

After installing the SCS interface, you import SCS using
```python
import scs
```
This module provides a single function `solve` with the following call signature:
```python
sol = scs.solve(data,
                cone,
                use_indirect=False,
                gpu=False,
                verbose=True,
                normalize=True,
                max_iters=5000,
                scale=1,
                eps=1e-5,
                cg_rate=2,
                alpha=1.5,
                rho_x=1e-3,
                acceleration_lookback=10,
                write_data_filename=None)
```
The argument `data` is a python dictionary with three elements `A`, `b`, and `c`
where `b` and `c` are `1d` NUMPY arrays and `A` is a SCIPY **sparse matrix in
CSC format**; if they are not of the proper format, SCS will attempt to convert
them.

The argument `cone` is a dictionary with fields `f`, `l`, `q`, `s`, `ep`, `ed`,
and `p` (all of which are optional) corresponding to the supported cone types.

The returned object is a dictionary containing the keys `'x'`, `'y'`, `'s'`, and
`'info'`.  The first three are NUMPY arrays containing the respective solution
vector. The `'info'` value is a dictionary with solver information.

### Warm-starting

Warm-starting SCS with a guess of the primal-dual solution can reduce the total
solve time. This is useful, for example, when solving several similar problems
sequentially. To do this simply add fields to the `data` dictionary passed to
`scs.solve` with additional fields `x`, `y`, and `s` (or any subset thereof)
where `x` and `s` correspond to NUMPY arrays containing the primal solution
guesses and `y` corresponds to a NUMPY array containing the dual solution guess.

