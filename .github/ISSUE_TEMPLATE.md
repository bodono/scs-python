**Please read this first**

If you are have a problem that SCS struggles with you can pass a string to the
`write_data_filename` argument and SCS will dump a file containing the problem
to disk. Zip the file and send it to us or attach it to this bug report for easy
reproduction of the issue.

A common cause of issues is not linking BLAS/LAPACK libraries correctly. 
SCS relies on Numpy to tell it where these libraries are on your system, using
the `get_info` commands:
```python
from numpy.distutils.system_info import get_info
print(get_info("blas_opt"))  # best blas install
print(get_info("lapack_opt"))  # best lapack install
print(get_info("blas"))  # fall back blas install
print(get_info("lapack"))  # fall back lapack install
```
If you are having this issue please search for resources on installing and
linking these libraries in python first. Also please make sure that the output
of the `get_info` commands above make sense for your system, ie, that the
`library_dirs` they return actually exist. You can try openblas if you need a
BLAS library.


## Specifications
- OS:
- SCS version:
- Python version:
- Numpy version:
- Scipy version:
- Output of `get_info` print commands as above (if applicable):

## Description
A clear and concise description of the problem.

## How to reproduce
Ideally a minimal snippet of code that reproduces the problem if possible.

## Additional information
Extra context.

## Output
Entire SCS output including the entire stack trace (if applicable).
