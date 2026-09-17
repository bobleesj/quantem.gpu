#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=./.build/arm64-apple-macosx/release/ssb-objective-probe3
H5="${SSB_AUDIT_H5:?set SSB_AUDIT_H5 to the ARINA original master.h5}"
OUT="${SSB_AUDIT_RUNS:?set SSB_AUDIT_RUNS to the run output directory}"
i=0
for n in 8 4 8 4 8 4; do
  i=$((i+1))
  env SSB_PHASE_BATCH="$n" "$BIN" "$H5" "$OUT/pbaba-$i-$n" >/dev/null 2>&1
done
echo pb-aba-done
