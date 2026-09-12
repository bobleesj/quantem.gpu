#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
swift build -c release --disable-sandbox --product metal-original-hdf5-benchmark
swift build -c release --disable-sandbox --product metal-ssb-benchmark
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -I "$build_dir/Modules" \
  -I "$build_dir/CNativeHDF5.build" \
  -I "$build_dir/CMetal4DSTEMInteractions.build" \
  -I src/quantem/gpu/swift/Sources/CNativeHDF5/include \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  experiments/20260910-ssb-full-bf-scheduling/probe.swift \
  "$build_dir"/MetalSSBKernels.build/*.o \
  "$build_dir"/Metal4DSTEMStreamingIO.build/*.o \
  "$build_dir"/Metal4DSTEMKernels.build/*.o \
  "$build_dir"/Native4DSTEMIO.build/*.o \
  "$build_dir"/CNativeHDF5.build/*.o \
  "$build_dir"/CMetal4DSTEMInteractions.build/*.o \
  src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a \
  -lz -o "$build_dir/ssb-scheduling-probe"
