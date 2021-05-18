#!/bin/sh
# distributing wheels broken due to blas linking issues
#python setup.py sdist bdist_wheel
python setup.py sdist
twine upload dist/*

