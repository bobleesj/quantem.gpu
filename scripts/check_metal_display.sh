#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --disable-sandbox --product metal-image-runtime-benchmark
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -I "$build_dir/Modules" \
  tests/metal/shared_display_check.swift \
  "$build_dir"/MetalImageRuntime.build/*.o \
  "$build_dir"/MetalDisplayKernels.build/*.o \
  -o "$build_dir/shared-display-check" -parse-as-library
"$build_dir/shared-display-check"
