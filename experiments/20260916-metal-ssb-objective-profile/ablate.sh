#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=./.build/arm64-apple-macosx/release/ssb-objective-probe3
H5="${SSB_AUDIT_H5:?set SSB_AUDIT_H5 to the ARINA original master.h5}"
OUT="${SSB_AUDIT_RUNS:?set SSB_AUDIT_RUNS to the run output directory}"
run() {
  local name="$1"; shift
  echo "--- arm $name : $* ---"
  env "$@" "$BIN" "$H5" "$OUT/ablate-$name" >/dev/null 2>&1
}
run full
run no-column SSB_PROFILE_SKIP_COLUMN=1
run no-rows SSB_PROFILE_SKIP_ROWS=1
run no-nyquist SSB_PROFILE_SKIP_NYQUIST=1
run no-column-no-rows SSB_PROFILE_SKIP_COLUMN=1 SSB_PROFILE_SKIP_ROWS=1
run full-2
