#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${QGPU_TEST_BUILD_DIR:-}" ]]; then
  swift build -c release --disable-sandbox --product metal-ssb-benchmark
  build_dir=$(swift build -c release --show-bin-path)
else
  build_dir="$QGPU_TEST_BUILD_DIR"
fi
# Source renames can leave orphan objects in an incremental build directory.
# Link only the sources in the current SwiftPM manifest.
objects=()
while IFS= read -r source; do
  objects+=("$build_dir/MetalSSBKernels.build/$(basename "$source").o")
done < "$build_dir/MetalSSBKernels.build/sources"
swiftc -O -I "$build_dir/Modules" tests/metal/ssb_workflow_check.swift \
  "${objects[@]}" \
  -o "$build_dir/ssb-workflow-check" -parse-as-library
"$build_dir/ssb-workflow-check"
