#!/bin/bash
set -uo pipefail
WT=~/perf-lab/ssb-audit/interleave
RUNS=~/perf-lab/ssb-audit/interleave-runs
H5="/path/to/local/data/Live4DSTEM Testing/ARINA/arina-fixture-a_master.h5"
BIN=$WT/.build/arm64-apple-macosx/release/ssb-objective-probe3
echo "load_start $(uptime | sed 's/.*load averages: //')"
rep=1
for order in "8 4 16 2 32" "32 16 4 8 2" "4 8 2 32 16"; do
  for n in $order; do
    out=$RUNS/pb-$n-r$rep
    rm -rf "$out"; mkdir -p "$out"
    echo "arm pb=$n rep=$rep start $(date +%H:%M:%S) load $(uptime | sed 's/.*load averages: //' | cut -d, -f1)"
    GPU_RUN_LABEL=interleave-pb$n SSB_PHASE_BATCH=$n \
      ~/perf-lab/ssb-audit/gpurun "$BIN" "$H5" "$out" >/dev/null 2>&1
    echo "arm pb=$n rep=$rep done  $(date +%H:%M:%S)"
  done
  rep=$((rep+1))
done
echo "load_end $(uptime | sed 's/.*load averages: //')"
echo pb-aba-done
