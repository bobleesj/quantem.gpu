#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --disable-sandbox --disable-build-manifest-caching --target MetalScientificNumerics
build_dir=$(swift build -c release --show-bin-path)
tests_dir=src/quantem/gpu/swift/Tests/MetalScientificNumericsTests
# Use the current source list, not *.o: renamed files can leave stale objects.
objects=()
for target in MetalScientificNumerics Metal4DSTEMStreamingIO Metal4DSTEMKernels MetalCountResources Native4DSTEMIO; do
  while IFS= read -r source; do
    objects+=("$build_dir/$target.build/$(basename "$source").o")
  done < "$build_dir/$target.build/sources"
done
swiftc -O -parse-as-library -D SCIENTIFIC_NUMERICS_CHECK \
  -I "$build_dir/Modules" -L "$build_dir" -lhdf5 -lz \
  -Xcc "-fmodule-map-file=$build_dir/CNativeHDF5.build/module.modulemap" \
  -Xcc "-fmodule-map-file=$build_dir/CMetal4DSTEMInteractions.build/module.modulemap" \
  -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
  -framework Metal -framework MetalPerformanceShadersGraph \
  "$tests_dir/MetalImageReferenceTests.swift" \
  "$tests_dir/MetalCalibratedDetectorTests.swift" \
  src/quantem/gpu/swift/Tests/Native4DSTEMIOTests/MetalEncodedHDF5LoadingTests.swift \
  "${objects[@]}" \
  "$build_dir"/CNativeHDF5.build/*.o \
  "$build_dir"/CMetal4DSTEMInteractions.build/*.o \
  -o "$build_dir/scientific-numerics-check"
"$build_dir/scientific-numerics-check" "$tests_dir/Fixtures" \
  src/quantem/gpu/swift/Tests/Native4DSTEMIOTests/Fixtures
