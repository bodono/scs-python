#!/bin/bash
# Size experiment: link the scs MKL backend against STATIC oneMKL (Intel's
# mkl-static PyPI wheel) into one shared object, prove it solves in a
# clean environment with no MKL on the loader path, and report sizes.
set -euo pipefail
LAYER=${1:-sequential}            # sequential | intel_thread | tbb_thread
VENV=${VENV:-$PWD/venv}
L=$VENV/lib
SRC=scs_source
OUT=out-$LAYER
rm -rf "$OUT"; mkdir -p "$OUT"

CFLAGS="-O3 -fno-math-errno -fPIC -ffunction-sections -fdata-sections \
  -DUSE_LAPACK -DUSE_SPECTRAL_CONES=1 -DCTRLC=1 -DSCS_MKL=1 \
  -I$SRC -I$SRC/include -I$SRC/linsys -I$VENV/include"
# static MKL has no mkl_rt, so the SDL-only interface-layer call becomes a
# no-op: the LP64 interface is fixed at link time.
cat > "$OUT/mkl_stub.c" <<'STUB'
int MKL_Set_Interface_Layer(int layer) { return layer; }
STUB
objs=()
for f in $SRC/src/*.c $SRC/src/spectral_cones/*.c $SRC/src/spectral_cones/*/*.c \
         $SRC/linsys/scs_matrix.c $SRC/linsys/csparse.c \
         $SRC/linsys/mkl/direct/private.c "$OUT/mkl_stub.c"; do
  o="$OUT/$(echo "$f" | tr '/' '_' | sed 's/\.c$/.o/')"
  gcc $CFLAGS -c "$f" -o "$o"
  objs+=("$o")
done

case "$LAYER" in
  sequential)   RT="" ;;
  intel_thread) RT="-L$L -liomp5" ;;
  tbb_thread)   RT="-L$L -ltbb" ;;
esac
SO="$OUT/libscsmkl_static.so"
gcc -shared -o "$SO" "${objs[@]}" \
  -Wl,--start-group "$L/libmkl_intel_lp64.a" "$L/libmkl_$LAYER.a" "$L/libmkl_core.a" -Wl,--end-group \
  $RT -lpthread -lm -ldl \
  -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,-rpath,"$L"

echo "== NEEDED (must contain no libmkl) =="
readelf -d "$SO" | grep NEEDED
if readelf -d "$SO" | grep -q "libmkl"; then echo "FAIL: dynamic MKL dependency present"; exit 1; fi

echo "== exported symbols (MKL must be hidden) =="
nm -D --defined-only "$SO" | grep -c " T " | xargs echo "exported functions:"
nm -D --defined-only "$SO" | grep -ci "mkl\|pardiso\|dgemm" | xargs echo "of which MKL-looking (want 0):" || true

echo "== solve in a clean environment =="
gcc -O2 -DUSE_LAPACK -I$SRC -I$SRC/include -I$SRC/linsys \
  $SRC/test/random_socp_prob.c -L"$OUT" -lscsmkl_static -Wl,-rpath,"$PWD/$OUT" -lm -o "$OUT/demo"
env -i PATH=/usr/bin:/bin "$OUT/demo" 800 0.1 0.3 7 > "$OUT/demo.log" 2>&1 || { tail -20 "$OUT/demo.log"; echo "FAIL: demo"; exit 1; }
grep -m1 "lin-sys" "$OUT/demo.log"
grep -m1 -i "status" "$OUT/demo.log"
grep -qi "mkl-pardiso" "$OUT/demo.log" || { echo "FAIL: not the pardiso backend"; exit 1; }
grep -qi "solved" "$OUT/demo.log" || { echo "FAIL: not solved"; exit 1; }

echo "== sizes =="
cp "$SO" "$OUT/stripped.so"; strip --strip-unneeded "$OUT/stripped.so"
raw=$(stat -c %s "$SO"); strp=$(stat -c %s "$OUT/stripped.so")
gz=$(gzip -6 -c "$OUT/stripped.so" | wc -c); xz=$(xz -6 -c "$OUT/stripped.so" | wc -c)
extra=""
case "$LAYER" in
  intel_thread) extra="libiomp5.so gz=$(gzip -6 -c "$L/libiomp5.so" | wc -c)" ;;
  tbb_thread)   extra="libtbb gz=$(gzip -6 -c "$(ls "$L"/libtbb.so.* | head -1)" | wc -c)" ;;
esac
printf 'SIZE layer=%s raw=%dMB stripped=%dMB gzip=%dMB xz=%dMB %s\n' \
  "$LAYER" $((raw/1000000)) $((strp/1000000)) $((gz/1000000)) $((xz/1000000)) "$extra"
