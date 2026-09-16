#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 1 ]]; then
  echo 'Usage: bash scripts/check_metal_ssb_scan_sizes.sh <CUDA-reference-directory>' >&2
  exit 2
fi
swift build -c release --disable-sandbox --product metal-ssb-benchmark
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -I "$build_dir/Modules" tests/metal/ssb_scan_size_check.swift \
  "$build_dir"/MetalSSBKernels.build/*.o \
  -o "$build_dir/ssb-scan-size-check" -parse-as-library
"$build_dir/ssb-scan-size-check" "$1"
