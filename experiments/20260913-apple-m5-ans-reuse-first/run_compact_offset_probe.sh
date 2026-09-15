#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${QGPU_PROBE_SKIP_BUILD:-0}" != 1 ]]; then
  swift build -c release --disable-sandbox \
    --product metal-paired-runtime-tans-series-benchmark
fi

build_dir=$(swift build -c release --show-bin-path)
objects=()
while IFS= read -r object; do
  case "$object" in
    */MetalPairedRuntimeTANSSeriesBenchmark.build/*) ;;
    *) objects+=("$object") ;;
  esac
done < "$build_dir/metal-paired-runtime-tans-series-benchmark.product/Objects.LinkFileList"

swiftc -O -parse-as-library -I "$build_dir/Modules" \
  -I "$build_dir/CNativeHDF5.build" \
  -I "$build_dir/CMetal4DSTEMInteractions.build" \
  -I src/quantem/gpu/swift/Sources/CNativeHDF5/include \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  experiments/20260913-apple-m5-ans-reuse-first/compact_offset_probe.swift \
  "${objects[@]}" \
  src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a \
  -lz -o "$build_dir/compact-offset-probe"

"$build_dir/compact-offset-probe"
