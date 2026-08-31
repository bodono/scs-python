#!/bin/bash
# Add a prefix-relative RUNPATH to the repaired wheel's _scs_mkl extension.
#
# The wheel does not vendor MKL (its dlopen'd CPU dispatch kernels are
# invisible to auditwheel; cvxgrp/scs#423). The scs[mkl] extra installs
# Intel's official wheels into <prefix>/lib, and site-packages/scs sits
# exactly four levels below the prefix in every standard layout (venv,
# conda, user site, system), so $ORIGIN/../../../../lib lets the dynamic
# loader resolve the whole MKL component group in one dlopen -- with the
# correct mutual binding MKL's libraries require, and without widening
# any symbol scope (NumPy's vendored OpenBLAS is unaffected).
set -euo pipefail
dest_dir="$1"
whl=$(ls -t "$dest_dir"/scs-*.whl | head -1)
tmp=$(mktemp -d)
python -m pip install --quiet wheel
python -m wheel unpack --dest "$tmp" "$whl"
unpacked=$(ls -d "$tmp"/scs-*)
patched=0
for so in "$unpacked"/scs/_scs_mkl*.so; do
  [ -e "$so" ] || continue
  patchelf --add-rpath '$ORIGIN/../../../../lib' "$so"
  echo "add_mkl_rpath: $(basename "$so") rpath -> $(patchelf --print-rpath "$so")"
  patched=1
done
[ "$patched" -eq 1 ] || { echo "add_mkl_rpath: no _scs_mkl extension found" >&2; exit 1; }
python -m wheel pack --dest-dir "$dest_dir" "$unpacked"
rm -rf "$tmp"
