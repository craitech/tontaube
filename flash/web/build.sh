#!/bin/bash
# Builds the browser engine (WebAssembly, threads) into ../extension/:
#   tts.{js,wasm}          SIMD
#   tts_relaxed.{js,wasm}  SIMD + relaxed SIMD (fused multiply-add); the extension picks it where available
#   tts_scalar.{js,wasm}   no SIMD
# Sources: ../engine/ (the engine, shared with the native CPU build) and ../text/ (rule verbalizer + chunker).
#
#   ./build.sh [EMSDK_DIR]      EMSDK_DIR: an activated emsdk checkout (tested with Emscripten 6.0.13);
#                               without it, em++ must be on PATH (e.g. after `source emsdk_env.sh`)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../engine" && pwd)
TEXT=$(cd "$HERE/../text" && pwd)
OUT=$(cd "$HERE/../extension" && pwd)

if [ $# -ge 1 ]; then
    # shellcheck disable=SC1091
    source "$1/emsdk_env.sh" > /dev/null 2>&1
fi
command -v em++ > /dev/null || { echo "em++ not found: pass the emsdk folder or put em++ on PATH" >&2; exit 1; }

JOBS=$(nproc 2> /dev/null || echo 4)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

FLAGS=(-std=c++17 -O3 -pthread -fwasm-exceptions)
INC=(-I"$SRC" -I"$TEXT" -I"$TEXT/include" -I"$TEXT/src")

# the text front end once, objects in parallel
mapfile -t RV < <(cd "$TEXT" && find src/core src/rules -name '*.cpp' | sort)
printf '%s\n' "${RV[@]}" | xargs -P "$JOBS" -I{} sh -c \
    'mkdir -p "$(dirname "$0/$1")" && em++ '"${FLAGS[*]}"' -Wno-unused-function -I"$2/include" -I"$2/src" -c "$2/$1" -o "$0/$1.o"' \
    "$TMP" {} "$TEXT"
OBJ=()
for f in "${RV[@]}"; do OBJ+=("$TMP/$f.o"); done

LINK=(-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=1MB -sDEFAULT_PTHREAD_STACK_SIZE=1MB
      -sEXIT_RUNTIME=0 -sDYNAMIC_EXECUTION=0 -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createTTS -sENVIRONMENT=web,worker
      -sINVOKE_RUN=0 "-sPTHREAD_POOL_SIZE=Module.ttsPool" -sEXPORTED_RUNTIME_METHODS=UTF8ToString,ccall,wasmMemory
      -sEXPORTED_FUNCTIONS=_malloc,_free)
pids=()
for v in simd relaxed scalar; do
    case $v in
        simd) f=(-msimd128); name=tts ;;
        relaxed) f=(-msimd128 -mrelaxed-simd); name=tts_relaxed ;;
        scalar) f=(-DWEB_SCALAR); name=tts_scalar ;;
    esac
    em++ "${FLAGS[@]}" "${INC[@]}" "${f[@]}" "${LINK[@]}" -o "$TMP/$name.js" "$SRC/tts_web.cpp" "${OBJ[@]}" & pids+=($!)
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
[ $bad = 0 ] || { echo "a WebAssembly build failed" >&2; exit 1; }
for name in tts tts_relaxed tts_scalar; do cp "$TMP/$name.js" "$TMP/$name.wasm" "$OUT/"; done
ls -l "$OUT"/tts*.js "$OUT"/tts*.wasm
