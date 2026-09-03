#!/bin/bash
# Size experiment: link the scs MKL backend against STATIC oneMKL (Intel's
# mkl-static PyPI wheel) into one shared object, prove it solves in a
# clean environment with no MKL on the loader path, and report sizes.
set -euo pipefail
LAYER=${1:-sequential}            # sequential | intel_thread | tbb_thread
SPECTRAL=${2:-0}                  # 1 = -DUSE_SPECTRAL_CONES=1 (wheel default: 0)
VENV=${VENV:-$PWD/venv}
L=$VENV/lib
SRC=scs_source
OUT=out-$LAYER
OUT=out-$LAYER-spectral$SPECTRAL
rm -rf "$OUT"; mkdir -p "$OUT"

CFLAGS="-O3 -fno-math-errno -fPIC -ffunction-sections -fdata-sections \
  -DUSE_LAPACK -DCTRLC=1 -DSCS_MKL=1 \
  -I$SRC -I$SRC/include -I$SRC/linsys -I$VENV/include"
SPECTRAL_SRC=""
if [ "$SPECTRAL" = 1 ]; then
  CFLAGS="$CFLAGS -DUSE_SPECTRAL_CONES=1"
  SPECTRAL_SRC="$SRC/src/spectral_cones/*.c $SRC/src/spectral_cones/*/*.c"
fi
# static MKL has no mkl_rt, so the SDL-only interface-layer call becomes a
# no-op: the LP64 interface is fixed at link time.
cat > "$OUT/mkl_stub.c" <<'STUB'
int MKL_Set_Interface_Layer(int layer) { return layer; }
STUB
objs=()
for f in $SRC/src/*.c $SPECTRAL_SRC \
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
link() {  # $1 = output, rest = extra linker flags
  local out=$1; shift
  gcc -shared -o "$out" "${objs[@]}" \
    -Wl,--start-group "$L/libmkl_intel_lp64.a" "$L/libmkl_$LAYER.a" "$L/libmkl_core.a" -Wl,--end-group \
    $RT -lpthread -lm -ldl -Wl,--exclude-libs,ALL -Wl,-rpath,"$L" "$@"
}
SO="$OUT/libscsmkl_static.so"
link "$SO" -Wl,--gc-sections
link "$OUT/libscsmkl_nogc.so"

echo "== NEEDED (must contain no libmkl) =="
readelf -d "$SO" | grep NEEDED
if readelf -d "$SO" | grep -q "libmkl"; then echo "FAIL: dynamic MKL dependency present"; exit 1; fi

echo "== exported symbols (MKL must be hidden) =="
nm -D --defined-only "$SO" | grep -c " T " | xargs echo "exported functions:"
nm -D --defined-only "$SO" | grep -ci "mkl\|pardiso\|dgemm" | xargs echo "of which MKL-looking (want 0):" || true

echo "== sizes =="
for v in libscsmkl_static libscsmkl_nogc; do
  cp "$OUT/$v.so" "$OUT/$v.stripped"; strip --strip-unneeded "$OUT/$v.stripped"
  printf '  %-18s raw=%dMB stripped=%dMB gzip=%dMB\n' "$v" \
    $(( $(stat -c %s "$OUT/$v.so")/1000000 )) $(( $(stat -c %s "$OUT/$v.stripped")/1000000 )) \
    $(( $(gzip -6 -c "$OUT/$v.stripped" | wc -c)/1000000 ))
done
cp "$SO" "$OUT/stripped.so"; strip --strip-unneeded "$OUT/stripped.so"
raw=$(stat -c %s "$SO"); strp=$(stat -c %s "$OUT/stripped.so")
gz=$(gzip -6 -c "$OUT/stripped.so" | wc -c); xz=$(xz -6 -c "$OUT/stripped.so" | wc -c)
extra=""
case "$LAYER" in
  intel_thread) extra="libiomp5.so gz=$(gzip -6 -c "$L/libiomp5.so" | wc -c)" ;;
  tbb_thread)   extra="libtbb gz=$(gzip -6 -c "$(ls "$L"/libtbb.so.* | head -1)" | wc -c)" ;;
esac
printf 'SIZE layer=%s spectral=%s raw=%dMB stripped=%dMB gzip=%dMB xz=%dMB %s\n' \
  "$LAYER" "$SPECTRAL" $((raw/1000000)) $((strp/1000000)) $((gz/1000000)) $((xz/1000000)) "$extra"
echo "== solve in a clean environment =="
# the demo must see the SAME struct layouts as the library (USE_SPECTRAL_CONES
# adds ScsCone fields), so it gets the identical defines
run_demo() {  # $1 = library basename (without lib/.so)
  gcc $CFLAGS $SRC/test/random_socp_prob.c -L"$OUT" -l"$1" -Wl,-rpath,"$PWD/$OUT" -lm -o "$OUT/demo_$1"
  if env -i PATH=/usr/bin:/bin stdbuf -oL "$OUT/demo_$1" 800 0.1 0.3 7 > "$OUT/demo_$1.log" 2>&1 \
     && grep -qi "mkl-pardiso" "$OUT/demo_$1.log" && grep -qi "solved" "$OUT/demo_$1.log"; then
    echo "DEMO $1: ok  ($(grep -m1 'lin-sys' "$OUT/demo_$1.log" | tr -s ' '))"
  else
    echo "DEMO $1: FAILED (rc=$?)"; tail -12 "$OUT/demo_$1.log"; return 1
  fi
}
set +e
run_demo scsmkl_static; ok_gc=$?
run_demo scsmkl_nogc;   ok_nogc=$?
set -e
printf 'RESULT layer=%s gc_sections=%s plain=%s\n' "$LAYER" \
  "$([ $ok_gc -eq 0 ] && echo ok || echo FAIL)" "$([ $ok_nogc -eq 0 ] && echo ok || echo FAIL)"


echo "== full core test suite against the static library (clean environment) =="
gcc $CFLAGS -I$SRC/test $SRC/test/run_tests.c -L"$OUT" -lscsmkl_static -Wl,-rpath,"$PWD/$OUT" -lm -o "$OUT/run_tests"
set +e
( cd "$OUT" && env -i PATH=/usr/bin:/bin stdbuf -oL ./run_tests > run_tests.log 2>&1 ); rc=$?
set -e
grep -E "ALL TESTS PASSED|Tests run|FAIL|failed|Segmentation" "$OUT/run_tests.log" | head -12
if [ $rc -eq 0 ] && grep -q "ALL TESTS PASSED" "$OUT/run_tests.log"; then
  printf 'SUITE layer=%s spectral=%s: PASSED (%s)\n' "$LAYER" "$SPECTRAL" "$(grep -m1 'Tests run' "$OUT/run_tests.log")"
else
  printf 'SUITE layer=%s spectral=%s: FAILED rc=%s\n' "$LAYER" "$SPECTRAL" "$rc"; tail -25 "$OUT/run_tests.log"; exit 1
fi
