#!/usr/bin/env bash
# Build DJATOM x265 3.4+70 (10-bit, X265_BUILD 198) and FFmpeg n7.1 with libx265.
# Usage: Scripts/build_ffmpeg_x265.sh <linux|iphoneos> <prefix>
set -euo pipefail

TARGET="${1:?target linux or iphoneos}"
PREFIX="$(mkdir -p "$2" && cd "$2" && pwd)"
X265_TAG="3.4+70"
FFMPEG_TAG="n7.1"
BUILD_ID="djatom-${X265_TAG}-ffmpeg-${FFMPEG_TAG}-${TARGET}-v1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${TMPDIR:-/tmp}/rife60-ffmpeg-x265-src-${TARGET}"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

if [[ -f "$PREFIX/BUILD-INFO.txt" && -f "$PREFIX/lib/libx265.a" && -f "$PREFIX/lib/libavcodec.a" ]]; then
  if grep -qx "build-id=${BUILD_ID}" "$PREFIX/BUILD-INFO.txt"; then
    echo "FFmpeg/x265 already built at $PREFIX ($BUILD_ID)"
    exit 0
  fi
fi

rm -rf "$PREFIX"
mkdir -p "$PREFIX" "$SRC"
cd "$SRC"

if [[ ! -d x265/.git ]]; then
  rm -rf x265
  git clone --filter=blob:none https://github.com/DJATOM/x265-aMod.git x265
fi
git -C x265 fetch --depth 1 origin "refs/tags/${X265_TAG}:refs/tags/${X265_TAG}"
git -C x265 checkout -f "$X265_TAG"
python3 - "$SRC/x265/source/CMakeLists.txt" "$SRC/x265/source/dynamicHDR10/CMakeLists.txt" << 'PY'
import pathlib, sys
for name in sys.argv[1:]:
    path = pathlib.Path(name)
    if not path.is_file():
        continue
    text = path.read_text()
    text = text.replace("cmake_policy(SET CMP0025 OLD)", "cmake_policy(SET CMP0025 NEW)")
    text = text.replace("cmake_policy(SET CMP0054 OLD)", "cmake_policy(SET CMP0054 NEW)")
    text = text.replace("cmake_minimum_required (VERSION 2.8.8)", "cmake_minimum_required(VERSION 3.5)")
    text = text.replace("cmake_minimum_required (VERSION 2.8.11)", "cmake_minimum_required(VERSION 3.5)")
    path.write_text(text)
PY

if [[ ! -d ffmpeg/.git ]]; then
  rm -rf ffmpeg
  git clone --depth 1 --branch "$FFMPEG_TAG" https://github.com/FFmpeg/FFmpeg.git ffmpeg
fi
git -C ffmpeg checkout -f "$FFMPEG_TAG"

python3 - "$SRC/x265/source/common/aarch64/asm.S" "$SRC/x265/source/common/aarch64" << 'PY'
import pathlib, sys
asm = pathlib.Path(sys.argv[1])
text = asm.read_text()
text = text.replace(
"""#ifdef PREFIX
#define EXTERN_ASM _
#else
#define EXTERN_ASM
#endif

#ifdef __ELF__
#define ELF
#else
#define ELF @
#endif

#define HAVE_AS_FUNC 1
""",
"""#ifdef __APPLE__
#define EXTERN_ASM _
#else
#ifdef PREFIX
#define EXTERN_ASM _
#else
#define EXTERN_ASM
#endif
#endif

#ifdef __ELF__
#define ELF
#else
#define ELF @
#endif

#ifdef __APPLE__
#define HAVE_AS_FUNC 0
#else
#define HAVE_AS_FUNC 1
#endif
""")
asm.write_text(text)
for path in pathlib.Path(sys.argv[2]).glob("*.S"):
    body = path.read_text()
    body = body.replace(
        ".section .rodata\n",
        "#ifdef __APPLE__\n.section __TEXT,__const\n#else\n.section .rodata\n#endif\n",
    )
    path.write_text(body)
PY

python3 - "$SRC/x265/source/CMakeLists.txt" << 'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
old = """            add_custom_command(
                OUTPUT ${ASM}.${SUFFIX}
                COMMAND ${CMAKE_CXX_COMPILER}
                ARGS ${ARM_ARGS} -c ${ASM_SRC} -o ${ASM}.${SUFFIX}
                DEPENDS ${ASM_SRC})"""
new = """            add_custom_command(
                OUTPUT ${ASM}.${SUFFIX}
                COMMAND ${CMAKE_CXX_COMPILER}
                ARGS ${ARM_ARGS} -arch arm64 -target arm64-apple-ios16.0 -miphoneos-version-min=16.0 -isysroot ${CMAKE_OSX_SYSROOT} -c ${ASM_SRC} -o ${ASM}.${SUFFIX}
                DEPENDS ${ASM_SRC})"""
if text.count(old) != 1:
    raise SystemExit("x265 ARM assembly rule was not found")
path.write_text(text.replace(old, new, 1))
PY

build_x265() {
  local build="$SRC/x265-build"
  local asm_flag="$1"
  rm -rf "$build"
  local cmake_args=(
    -G Ninja
    -S "$SRC/x265/source"
    -B "$build"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DENABLE_SHARED=OFF
    -DENABLE_CLI=OFF
    -DENABLE_PIC=ON
    -DHIGH_BIT_DEPTH=ON
    -DEXPORT_C_API=ON
    -DENABLE_LIBNUMA=OFF
    -DENABLE_ASSEMBLY="$asm_flag"
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  )
  if [[ "$TARGET" == iphoneos ]]; then
    cmake_args+=(
      -DCMAKE_SYSTEM_NAME=iOS
      -DCMAKE_SYSTEM_PROCESSOR=aarch64
      -DCMAKE_OSX_ARCHITECTURES=arm64
      -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0
      -DCMAKE_OSX_SYSROOT=iphoneos
      -DCMAKE_C_COMPILER=clang
      -DCMAKE_CXX_COMPILER=clang++
    )
  else
    cmake_args+=(
      -DCMAKE_C_COMPILER=gcc
      -DCMAKE_CXX_COMPILER=g++
    )
  fi
  cmake "${cmake_args[@]}" || return 1
  cmake --build "$build" -j "$JOBS" || return 1
  cmake --install "$build" || return 1
}

rewrite_pc() {
  cat > "$PREFIX/lib/pkgconfig/x265.pc" << EOF
prefix=${PREFIX}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: x265
Description: DJATOM x265 3.4+70 10-bit
Version: 3.4+70
Libs: -L\${libdir} -lx265
Libs.private: -lc++
Cflags: -I\${includedir}
EOF
  if [[ "$TARGET" == linux ]]; then
    cat > "$PREFIX/lib/pkgconfig/x265.pc" << EOF
prefix=${PREFIX}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: x265
Description: DJATOM x265 3.4+70 10-bit
Version: 3.4+70
Libs: -L\${libdir} -lx265
Libs.private: -lstdc++ -lm -lpthread -ldl
Cflags: -I\${includedir}
EOF
  fi
}

ASM_ON=ON
if [[ "$TARGET" == linux ]]; then
  ASM_ON=ON
fi
set +e
build_x265 "$ASM_ON"
x265_status=$?
set -e
if [[ "$x265_status" != 0 && "$TARGET" == iphoneos ]]; then
  echo "x265 assembly build failed; retrying once without assembly"
  build_x265 OFF
  echo "assembly=off" > "$PREFIX/x265-assembly.txt"
else
  if [[ "$x265_status" != 0 ]]; then
    exit "$x265_status"
  fi
  echo "assembly=${ASM_ON}" > "$PREFIX/x265-assembly.txt"
fi
rewrite_pc
test -f "$PREFIX/lib/libx265.a"
nm "$PREFIX/lib/libx265.a" > "$PREFIX/x265-symbols.txt"
if ! grep -q 'x265_api_get_198' "$PREFIX/x265-symbols.txt"; then
  echo "libx265.a is missing x265_api_get_198"
  exit 1
fi

if [[ "$TARGET" == iphoneos ]]; then
  python3 - "$SRC/ffmpeg/configure" << 'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
old = "check_cflags -mdynamic-no-pic && add_asflags -mdynamic-no-pic;"
new = "check_cflags -mdynamic-no-pic;"
if old not in text:
    raise SystemExit("FFmpeg configure no longer has the mdynamic-no-pic assembler flag")
path.write_text(text.replace(old, new, 1))
PY
fi

configure_ffmpeg() {
  local asm_args="$1"
  local build="$SRC/ffmpeg"
  cd "$build"
  make distclean >/dev/null 2>&1 || true
  local extra_cflags="-I$PREFIX/include"
  local extra_ldflags="-L$PREFIX/lib"
  local extra_libs="-lstdc++"
  local arch_args=(--arch=x86_64)
  if [[ "$TARGET" == iphoneos ]]; then
    local sdk
    sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
    extra_cflags="-arch arm64 -miphoneos-version-min=16.0 -isysroot $sdk $extra_cflags"
    extra_ldflags="-arch arm64 -miphoneos-version-min=16.0 -isysroot $sdk $extra_ldflags"
    extra_libs="-lc++"
    arch_args=(--arch=aarch64 --cpu=armv8 --sysroot="$sdk" --cc=clang --cxx=clang++ --host-cc=clang --target-os=darwin --enable-cross-compile)
  else
    extra_ldflags="$extra_ldflags -lstdc++ -lm -lpthread -ldl"
    arch_args=(--arch="$(uname -m)" --target-os=linux --cc=gcc --cxx=g++)
  fi
  export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  if ! command -v pkg-config >/dev/null; then
    echo "pkg-config is required to find libx265"
    return 1
  fi
  echo "pkg-config x265: $(pkg-config --modversion x265) libs: $(pkg-config --libs --static x265)"
  ./configure \
    --prefix="$PREFIX" \
    "${arch_args[@]}" \
    --pkg-config-flags=--static \
    --enable-static \
    --disable-shared \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-avdevice \
    --disable-postproc \
    --disable-network \
    --disable-autodetect \
    --enable-gpl \
    --enable-pic \
    --enable-pthreads \
    --disable-securetransport \
    --disable-videotoolbox \
    --disable-audiotoolbox \
    --disable-avfoundation \
    --disable-appkit \
    --disable-coreimage \
    --disable-iconv \
    --disable-bzlib \
    --enable-zlib \
    --enable-libx265 \
    $asm_args \
    --disable-everything \
    --enable-avcodec \
    --enable-avformat \
    --enable-avfilter \
    --enable-swscale \
    --enable-swresample \
    --enable-avutil \
    --enable-decoders \
    --enable-demuxers \
    --enable-parsers \
    --enable-encoder=libx265,aac \
    --enable-muxer=mov,mp4,ipod \
    --enable-protocol=file \
    --enable-filter=scale,fps,format,aresample,transpose,pad,setsar,hflip,vflip,null \
    --enable-bsf=extract_extradata,hevc_mp4toannexb,aac_adtstoasc,null \
    --extra-cflags="$extra_cflags" \
    --extra-ldflags="$extra_ldflags" \
    --extra-libs="$extra_libs"
}

ffmpeg_asm="--enable-asm --enable-neon"
if [[ "$TARGET" == linux ]]; then
  ffmpeg_asm="--enable-asm"
fi
set +e
configure_ffmpeg "$ffmpeg_asm"
configure_status=$?
set -e
if [[ "$configure_status" != 0 ]]; then
  echo "FFmpeg configure with assembly failed; retrying once without assembly"
  configure_ffmpeg "--disable-asm"
fi
cd "$SRC/ffmpeg"
make -j "$JOBS"
make install

cat > "$PREFIX/BUILD-INFO.txt" << EOF
build-id=${BUILD_ID}
x265-tag=${X265_TAG}
x265-build=198
ffmpeg-tag=${FFMPEG_TAG}
target=${TARGET}
$(cat "$PREFIX/x265-assembly.txt")
EOF
echo "Installed FFmpeg and DJATOM x265 to $PREFIX"
