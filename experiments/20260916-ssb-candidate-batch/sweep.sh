#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bin=.build/arm64-apple-macosx/release/ssb-candidate-batch-probe
h5="/path/to/local/data/Live4DSTEM Testing/ARINA/arina-fixture-a_master.h5"
out=/path/to/local/perf-lab/ssb-audit/candbatch-runs
for k in 1 2 4 8; do
  for order in groupedPasses interleaved; do
    echo "=== k=$k order=$order $(date '+%H:%M:%S') load=$(uptime | sed 's/.*averages: //') ==="
    GPU_RUN_LABEL=candbatch \
    SSB_CANDBATCH_K=$k SSB_CANDBATCH_ORDER=$order SSB_CANDBATCH_REPS=3 \
    SSB_CANDBATCH_POINTS=16 SSB_CANDBATCH_OUT=sweep-k${k}-${order}.json \
      /path/to/local/perf-lab/ssb-audit/gpurun "$bin" points "$h5" "$out"
    echo "=== done k=$k $order $(date '+%H:%M:%S') load=$(uptime | sed 's/.*averages: //') ==="
  done
done
