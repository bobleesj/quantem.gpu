#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${QGPU_PROBE_SKIP_BUILD:-0}" != 1 ]]; then
  swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
fi
build_dir=$(swift build -c release --show-bin-path)
# Use SwiftPM's live object list, not stale objects left by older builds.
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
  experiments/20260912-apple-m5-ans-adf-optimization/detector_correctness_probe.swift \
  src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSMacroTable.swift \
  "${objects[@]}" \
  src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a \
  -lz -o "$build_dir/detector-correctness-probe"
"$build_dir/detector-correctness-probe"
