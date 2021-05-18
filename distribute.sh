#!/bin/sh
# distributing wheels broken due to blas linking issues
pip install build
#python -m build --sdist --wheel
python -m build --sdist
twine upload dist/*

