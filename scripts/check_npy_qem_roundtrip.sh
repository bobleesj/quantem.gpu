#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/check_npy_qem_roundtrip.sh counts.npy [--reject=unsupported.npy]" >&2
  exit 2
fi
swift build -c release --product metal-runtime-ans-benchmark
build_dir=$(swift build -c release --show-bin-path)
swiftc -O -parse-as-library -target arm64-apple-macosx15.0 \
  -I "$build_dir/Modules" -L "$build_dir" -lhdf5 -lz \
  -Xcc "-fmodule-map-file=$build_dir/CNativeHDF5.build/module.modulemap" \
  -Xcc "-fmodule-map-file=$build_dir/CMetal4DSTEMInteractions.build/module.modulemap" \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  tests/metal/npy_qem_roundtrip.swift \
  "$build_dir"/Native4DSTEMIO.build/*.o \
  "$build_dir"/Metal4DSTEMStreamingIO.build/*.o \
  "$build_dir"/Metal4DSTEMKernels.build/*.o \
  "$build_dir"/MetalCountResources.build/*.o \
  "$build_dir"/CNativeHDF5.build/*.o \
  "$build_dir"/CMetal4DSTEMInteractions.build/*.o \
  -o "$build_dir/npy-qem-roundtrip"
"$build_dir/npy-qem-roundtrip" "$@"
