#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
swift build -c release --disable-sandbox --product metal-original-hdf5-benchmark
swift build -c release --disable-sandbox --product metal-ssb-benchmark
build_dir=$(swift build -c release --show-bin-path)
objects=()
while IFS= read -r object; do
  objects+=("$object")
done < <(sort -u \
  "$build_dir/metal-original-hdf5-benchmark.product/Objects.LinkFileList" \
  "$build_dir/metal-ssb-benchmark.product/Objects.LinkFileList" \
  | grep -vE '/Metal(OriginalHDF5|SSB)Benchmark\.build/')
build() {
  swiftc -O -I "$build_dir/Modules" \
    -I "$build_dir/CNativeHDF5.build" \
    -I "$build_dir/CMetal4DSTEMInteractions.build" \
    -I src/quantem/gpu/swift/Sources/CNativeHDF5/include \
    -I src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/Headers \
    "$1" \
    "${objects[@]}" \
    src/quantem/gpu/swift/Vendor/CHDF5.xcframework/macos-arm64/libhdf5.a \
    -lz -o "$build_dir/$2"
}
build experiments/20260916-metal-ssb-objective-profile/probe2.swift ssb-objective-probe2
build experiments/20260916-metal-ssb-objective-profile/probe3.swift ssb-objective-probe3
build experiments/20260916-metal-ssb-objective-profile/probe5.swift ssb-objective-probe5
