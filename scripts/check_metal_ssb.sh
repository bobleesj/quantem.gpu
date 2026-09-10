#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --disable-sandbox --product metal-ssb-benchmark
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -I "$build_dir/Modules" tests/metal/ssb_workflow_check.swift \
  "$build_dir"/MetalSSBKernels.build/*.o \
  -o "$build_dir/ssb-workflow-check" -parse-as-library
"$build_dir/ssb-workflow-check"
