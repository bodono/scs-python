#!/bin/bash
# Normalize the RUNPATH of the repaired wheel's _scs_mkl extension.
#
# The wheel does not vendor MKL (its dlopen'd CPU dispatch kernels are
# invisible to auditwheel; cvxgrp/scs#423). The scs[mkl] extra installs
# Intel's official wheels into <prefix>/lib, and site-packages/scs sits
# exactly four levels below the prefix in every standard layout (venv,
# conda, user site, system), so $ORIGIN/../../../../lib lets the dynamic
# loader resolve the whole MKL component group in one dlopen -- with the
# correct mutual binding MKL's libraries require, and without widening
# any symbol scope (NumPy's vendored OpenBLAS is unaffected).
#
# --set-rpath (not --add-rpath): the meson build bakes the container's
# /opt/intel/... paths into the extension, and auditwheel preserves them.
# Shipping those would let a build-machine-layout MKL installation win
# over the scs[mkl] runtime, and would let container tests silently
# resolve the build MKL. The RUNPATH must be exactly the one relative
# entry, and the MKL-related NEEDED set must be exactly what the loader
# shim and the mkl PyPI pin (mkl>=2026,<2027 -- the .so.3 ABI) provide;
# both are asserted here so a drift fails the build, not a user.
set -euo pipefail
dest_dir="$1"
whl=$(ls -t "$dest_dir"/scs-*.whl | head -1)
tmp=$(mktemp -d)
python -m pip install --quiet wheel
python -m wheel unpack --dest "$tmp" "$whl"
unpacked=$(ls -d "$tmp"/scs-*)
want_rpath='$ORIGIN/../../../../lib'
want_needed='libiomp5.so libmkl_core.so.3 libmkl_intel_lp64.so.3 libmkl_intel_thread.so.3 libmkl_rt.so.3'
patched=0
for so in "$unpacked"/scs/_scs_mkl*.so; do
  [ -e "$so" ] || continue
  patchelf --set-rpath "$want_rpath" "$so"
  got_rpath=$(patchelf --print-rpath "$so")
  if [ "$got_rpath" != "$want_rpath" ]; then
    echo "add_mkl_rpath: unexpected RUNPATH '$got_rpath' on $(basename "$so")" >&2
    exit 1
  fi
  got_needed=$(patchelf --print-needed "$so" | grep -E '^lib(mkl|iomp)' | sort | tr '\n' ' ' | sed 's/ $//')
  if [ "$got_needed" != "$want_needed" ]; then
    echo "add_mkl_rpath: NEEDED drift on $(basename "$so")" >&2
    echo "  want: $want_needed" >&2
    echo "  got:  $got_needed" >&2
    exit 1
  fi
  echo "add_mkl_rpath: $(basename "$so") rpath='$got_rpath' needed ok"
  patched=1
done
[ "$patched" -eq 1 ] || { echo "add_mkl_rpath: no _scs_mkl extension found" >&2; exit 1; }
python -m wheel pack --dest-dir "$dest_dir" "$unpacked"
rm -rf "$tmp"
