#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --product metal-runtime-ans-benchmark
build_dir=$(swift build -c release --show-bin-path)
objects=()
while IFS= read -r object; do
  case "$object" in
    */Native4DSTEMIO.build/*|*/Metal4DSTEMStreamingIO.build/*|*/Metal4DSTEMKernels.build/*|*/MetalCountResources.build/*|*/CNativeHDF5.build/*|*/CMetal4DSTEMInteractions.build/*)
      objects+=("$object") ;;
  esac
done < "$build_dir/metal-runtime-ans-benchmark.product/Objects.LinkFileList"
swiftc -O -parse-as-library -target arm64-apple-macosx15.0 \
  -I "$build_dir/Modules" -L "$build_dir" -lhdf5 -lz \
  -Xcc "-fmodule-map-file=$build_dir/CNativeHDF5.build/module.modulemap" \
  -Xcc "-fmodule-map-file=$build_dir/CMetal4DSTEMInteractions.build/module.modulemap" \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  tests/metal/qem_reference_roundtrip.swift \
  "${objects[@]}" \
  -o "$build_dir/qem-reference"
"$build_dir/qem-reference" "$@"
