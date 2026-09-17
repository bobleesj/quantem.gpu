#!/bin/bash
# Reproduce the traffic attribution.
#   1. apply traffic-gates.patch (experimental instrumentation; not production)
#   2. run_traffic.sh <master.h5> <out-dir> [reps]
# All arms are cycled round-robin inside one process so host load is common-mode.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${SSB_TRAFFIC_BIN:-./.build/arm64-apple-macosx/release/ssb-traffic-probe}
GPU_RUN_LABEL=${GPU_RUN_LABEL:-traffic}
"$HOME/perf-lab/ssb-audit/gpurun" \
  env GPU_RUN_LABEL="$GPU_RUN_LABEL" "$BIN" "$1" "$2" "${3:-4}" "${4:-session}"
