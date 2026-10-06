#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --product metal-runtime-ans-benchmark
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -parse-as-library -target arm64-apple-macosx15.0 \
  -I "$build_dir/Modules" -L "$build_dir" -lhdf5 -lz \
  -Xcc "-fmodule-map-file=$build_dir/CNativeHDF5.build/module.modulemap" \
  -Xcc "-fmodule-map-file=$build_dir/CMetal4DSTEMInteractions.build/module.modulemap" \
  -I native/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  tests/metal/scaled_qem_roundtrip.swift \
  "$build_dir"/Native4DSTEMIO.build/*.o \
  "$build_dir"/Metal4DSTEMStreamingIO.build/*.o \
  "$build_dir"/Metal4DSTEMKernels.build/*.o \
  "$build_dir"/MetalCountResources.build/*.o \
  "$build_dir"/CNativeHDF5.build/*.o \
  "$build_dir"/CMetal4DSTEMInteractions.build/*.o \
  -o "$build_dir/scaled-qem-roundtrip"
"$build_dir/scaled-qem-roundtrip" tests/data/qem-v2/u16-multiple-chunks.qem
