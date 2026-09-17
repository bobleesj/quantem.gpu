#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
H5="${SSB_AUDIT_H5:?set SSB_AUDIT_H5 to the ARINA original master.h5}"
OUT="${SSB_AUDIT_RUNS:?set SSB_AUDIT_RUNS to the run output directory}"
B=.build/arm64-apple-macosx/release/ssb-probe2-base
C=.build/arm64-apple-macosx/release/ssb-probe2-cand
"$B" "$H5" "$OUT/aba-b1" >/dev/null 2>&1
"$C" "$H5" "$OUT/aba-c1" >/dev/null 2>&1
"$B" "$H5" "$OUT/aba-b2" >/dev/null 2>&1
"$C" "$H5" "$OUT/aba-c2" >/dev/null 2>&1
