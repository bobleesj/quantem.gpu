#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bin=.build/arm64-apple-macosx/release/ssb-candidate-batch-probe
h5="/path/to/local/data/Live4DSTEM Testing/ARINA/arina-fixture-a_master.h5"
out=/path/to/local/perf-lab/ssb-audit/candbatch-runs
for spec in "4 1" "4 2" "4 4" "2 2" "2 4" "16 2"; do
  set -- $spec
  pb=$1; k=$2
  echo "=== pb=$pb k=$k $(date '+%H:%M:%S') load=$(uptime | sed 's/.*averages: //') ==="
  GPU_RUN_LABEL=candbatch \
  SSB_CANDBATCH_K=$k SSB_CANDBATCH_ORDER=groupedPasses SSB_CANDBATCH_PLANES=$pb SSB_CANDBATCH_REPS=3 \
  SSB_CANDBATCH_POINTS=16 SSB_CANDBATCH_OUT=pb${pb}-k${k}.json \
    /path/to/local/perf-lab/ssb-audit/gpurun "$bin" points "$h5" "$out"
  echo "=== done pb=$pb k=$k $(date '+%H:%M:%S') load=$(uptime | sed 's/.*averages: //') ==="
done
