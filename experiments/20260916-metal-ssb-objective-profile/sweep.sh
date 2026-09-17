#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=./.build/arm64-apple-macosx/release/ssb-objective-probe3
H5="${SSB_AUDIT_H5:?set SSB_AUDIT_H5 to the ARINA original master.h5}"
OUT="${SSB_AUDIT_RUNS:?set SSB_AUDIT_RUNS to the run output directory}"
for n in 8 16 32 4 2 64 8; do
  env SSB_PHASE_BATCH="$n" "$BIN" "$H5" "$OUT/pb-$n-$(date +%s)" >/dev/null 2>&1
  mv "$OUT"/pb-$n-* "$OUT/pb-$n" 2>/dev/null || true
done
echo sweep-done
