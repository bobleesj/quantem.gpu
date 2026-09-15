#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
probe_dir=$(mktemp -d)
trap 'rm -rf "$probe_dir"' EXIT
swiftc -O -parse-as-library \
  src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSPolarPlan.swift \
  experiments/20260912-apple-m5-ans-adf-optimization/polar_plan_cache_probe.swift \
  -o "$probe_dir/polar-plan-cache-probe"
"$probe_dir/polar-plan-cache-probe"
