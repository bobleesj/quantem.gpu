#!/bin/bash
# Paired runs under the shared GPU lock: instrumented fit first, then the
# identical fit with the recorder disabled (control for instrumentation
# overhead and for the bit-exact loss/optimum check).
set -euo pipefail
cd "$(dirname "$0")/../.."
input="${1:-/path/to/local/data/Live4DSTEM Testing/ARINA/arina-fixture-a_master.h5}"
out="${2:-$HOME/perf-lab/ssb-audit/metal-runs/timeline-fit}"
bin="$(swift build -c release --show-bin-path)/ssb-timeline-probe"
mkdir -p "$out"
echo "load_before=$(uptime | sed 's/.*load averages: //')"
echo "--- arm 1: timeline on ---"
GPU_RUN_LABEL=timeline SSB_TIMELINE=1 "$HOME/perf-lab/ssb-audit/gpurun" "$bin" "$input" "$out" 2>&1 | tail -4
echo "load_mid=$(uptime | sed 's/.*load averages: //')"
echo "--- arm 2: timeline off (control) ---"
GPU_RUN_LABEL=timeline SSB_TIMELINE=0 "$HOME/perf-lab/ssb-audit/gpurun" "$bin" "$input" "$out" 2>&1 | tail -4
echo "load_after=$(uptime | sed 's/.*load averages: //')"
