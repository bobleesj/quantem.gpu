#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --product metal-paired-runtime-tans-benchmark -Xswiftc -enable-testing
build_dir=$(swift build -c release --show-bin-path)
objects=()
for target in Native4DSTEMIO Metal4DSTEMStreamingIO Metal4DSTEMKernels MetalCountResources; do
  while IFS= read -r object; do objects+=("$object"); done < <(
    jq -r 'to_entries[] | select(.key != "") | .value.object // empty' "$build_dir/$target.build/output-file-map.json"
  )
done
swiftc -O -parse-as-library -target arm64-apple-macosx15.0 \
  -I "$build_dir/Modules" -L "$build_dir" -lhdf5 -lz \
  -Xcc "-fmodule-map-file=$build_dir/CNativeHDF5.build/module.modulemap" \
  -Xcc "-fmodule-map-file=$build_dir/CMetal4DSTEMInteractions.build/module.modulemap" \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  tests/metal/paired_mean_diffraction_check.swift "${objects[@]}" \
  "$build_dir"/CNativeHDF5.build/*.o "$build_dir"/CMetal4DSTEMInteractions.build/*.o \
  -o "$build_dir/paired-mean-check"
"$build_dir/paired-mean-check" "$@"
