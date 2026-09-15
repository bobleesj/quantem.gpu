#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
experiment="$repo_root/experiments/20260913-apple-m5-ans-selective-prefix-overlay"
planner="$repo_root/src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSPolarPlan.swift"
benchmark_masks="$repo_root/src/quantem/gpu/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark/main.swift"
frozen="$repo_root/experiments/20260912-apple-m5-ans-adf-optimization/polar16-radial1-aga.jsonl"

# Keep this prototype independent of opt-in production planner diagnostics.
export QGPU_PAIRED_RUNTIME_JOINT_PLAN=0
export QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN=0

swiftc "$planner" "$experiment/main.swift" -o /tmp/ans-selective-prefix-overlay-cpu
/tmp/ans-selective-prefix-overlay-cpu \
  "$frozen" \
  "$experiment/results/prefix-overlay.json"

printf '%s\n' "Mask geometry source: $benchmark_masks"
