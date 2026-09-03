#!/bin/bash
# Lay out Intel's static oneMKL archives under a prefix for the static
# _scs_mkl link (see mkl_static_prefix in meson.options). The archives come
# from Intel's own mkl-static PyPI wheel; pip download with an explicit
# platform tag so the build container's python and glibc are irrelevant.
set -euo pipefail
prefix=${1:?usage: install_mkl_static.sh <prefix>}
ver=${MKL_STATIC_VERSION:-2026.1.0}
py=$(ls -d /opt/python/cp3*-cp3*/bin/python 2>/dev/null | head -1 || command -v python3)
tmp=$(mktemp -d)
"$py" -m pip download --quiet --no-deps --only-binary=:all: \
  --platform manylinux_2_28_x86_64 -d "$tmp" "mkl-static==$ver" "mkl-include==$ver"
for whl in "$tmp"/*.whl; do
  "$py" -m zipfile -e "$whl" "$tmp/unpacked"
done
mkdir -p "$prefix"
# wheel data files live under <dist>.data/data/{lib,include}
cp -r "$tmp"/unpacked/*.data/data/. "$prefix"/
for a in libmkl_intel_lp64.a libmkl_sequential.a libmkl_core.a; do
  test -f "$prefix/lib/$a" || { echo "install_mkl_static: missing $prefix/lib/$a" >&2; exit 1; }
done
rm -rf "$tmp"
echo "install_mkl_static: oneMKL $ver archives under $prefix"
