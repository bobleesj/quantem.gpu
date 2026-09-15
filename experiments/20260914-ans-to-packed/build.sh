#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-benchmark
build_dir=$(swift build -c release --show-bin-path)
objects=()
while IFS= read -r object; do objects+=("$object"); done < <(
  rg -v '/MetalPairedRuntimeTANSBenchmark\.build/' "$build_dir/metal-paired-runtime-tans-benchmark.product/Objects.LinkFileList")
swiftc -O -I "$build_dir/Modules" -I "$build_dir/CNativeHDF5.build" \
  -I "$build_dir/CMetal4DSTEMInteractions.build" \
  -I src/quantem/gpu/swift/Sources/CNativeHDF5/include \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  experiments/20260914-ans-to-packed/probe.swift "${objects[@]}" \
  src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a \
  -lz -o "$build_dir/ans-to-packed-probe"
swiftc -O -I "$build_dir/Modules" -I "$build_dir/CNativeHDF5.build" \
  -I "$build_dir/CMetal4DSTEMInteractions.build" \
  -I src/quantem/gpu/swift/Sources/CNativeHDF5/include \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  experiments/20260914-ans-to-packed/synthetic.swift \
  src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSTables.swift \
  "${objects[@]}" \
  src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a \
  -lz -o "$build_dir/ans-to-packed-synthetic"
