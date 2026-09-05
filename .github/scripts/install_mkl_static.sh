#!/bin/bash
# Lay out Intel's static oneMKL archives (the mkl-static and mkl-include PyPI
# wheels) under a prefix for the static _scs_mkl link; see mkl_static_prefix
# in meson.options. Runs inside the cibuildwheel manylinux container.
set -euo pipefail
prefix=${1:?usage: install_mkl_static.sh <prefix>}
py=$(ls -d /opt/python/cp3*-cp3*/bin/python | head -1)
tmp=$(mktemp -d)
"$py" -m pip download --quiet --no-deps --only-binary=:all: \
  --platform manylinux_2_28_x86_64 -d "$tmp" mkl-static==2026.1.0 mkl-include==2026.1.0
for whl in "$tmp"/*.whl; do
  "$py" -m zipfile -e "$whl" "$tmp/unpacked"
done
mkdir -p "$prefix"
cp -r "$tmp"/unpacked/*.data/data/. "$prefix"/   # {lib,include}
rm -rf "$tmp"
