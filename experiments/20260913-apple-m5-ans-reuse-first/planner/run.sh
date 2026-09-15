#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/qgpu-paired-planner.XXXXXX")"
cleanup() {
  rm -f "$scratch/planner" "$scratch/greedy.out" "$scratch/joint.out" \
    "$scratch/greedy.parity" "$scratch/joint.parity" \
    "$scratch/greedy.cost" "$scratch/joint.cost"
  rmdir "$scratch"
}
trap cleanup EXIT

swiftc \
  "$repo_root/src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSPolarPlan.swift" \
  "$script_dir/main.swift" \
  -o "$scratch/planner"

env QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN=1 QGPU_PAIRED_RUNTIME_JOINT_PLAN=0 \
  "$scratch/planner" > "$scratch/greedy.out"
env QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN=1 QGPU_PAIRED_RUNTIME_JOINT_PLAN=1 \
  "$scratch/planner" > "$scratch/joint.out"

awk '$1 == "PARITY" { print }' "$scratch/greedy.out" > "$scratch/greedy.parity"
awk '$1 == "PARITY" { print }' "$scratch/joint.out" > "$scratch/joint.parity"
diff -u "$scratch/greedy.parity" "$scratch/joint.parity"

awk '$1 == "COST" { print $2, $3 }' "$scratch/greedy.out" \
  | LC_ALL=C sort -k1,1 > "$scratch/greedy.cost"
awk '$1 == "COST" { print $2, $3 }' "$scratch/joint.out" \
  | LC_ALL=C sort -k1,1 > "$scratch/joint.cost"
awk '
  NR == FNR { greedy[$1] = $2; next }
  { joint[$1] = $2 }
  !($1 in greedy) || $2 > greedy[$1] {
    printf "FAIL: joint cost for %s (%s) exceeds greedy (%s)\n", $1, $2, greedy[$1] > "/dev/stderr"
    failed = 1
  }
  END {
    for (label in greedy) {
      if (!(label in joint)) {
        printf "FAIL: missing joint cost for %s\n", label > "/dev/stderr"
        failed = 1
      }
    }
    exit failed
  }
' "$scratch/greedy.cost" "$scratch/joint.cost"

awk '$1 == "RANDOM_TOTAL" { print }' "$scratch/greedy.out"
awk '$1 == "RANDOM_TOTAL" { print }' "$scratch/joint.out"
awk '$1 == "PLAN" && $2 ~ /^tie-/ { print }' "$scratch/greedy.out"
awk '$1 == "PLAN" && $2 ~ /^tie-/ { print }' "$scratch/joint.out"
grep '^PASS ' "$scratch/greedy.out"
grep '^PASS ' "$scratch/joint.out"
printf 'PASS exact parity and seeded cost non-regression (%s cases)\n' \
  "$(wc -l < "$scratch/joint.cost" | tr -d ' ')"
