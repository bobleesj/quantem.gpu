#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${QGPU_PROBE_SKIP_BUILD:-0}" != 1 ]]; then
  swift build -c release --disable-sandbox --product metal-paired-runtime-tans-benchmark
fi
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -parse-as-library -I "$build_dir/Modules" \
  experiments/20260912-apple-m5-ans-adf-optimization/fine_polar_field_correctness_probe.swift \
  "$build_dir"/Metal4DSTEMKernels.build/*.o \
  -o "$build_dir/fine-polar-field-correctness-probe"
"$build_dir/fine-polar-field-correctness-probe"
