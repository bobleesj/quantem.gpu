#!/bin/bash
# Trial-budget cross-check on the native Metal path.
#
#   run_metal_budget.sh <basis> <seed> [<seed> ...]
#
#   basis A : parity-runs/strict-parity/arina-512-full-disk (historical
#             documented sampling, 2464 aperture-active BF of 8937 selected)
#             warm start = the recorded production optimum file
#   basis B : budget2-runs/strict-parity/arina-512-aperture-matched-8937
#             (all 8937 selected BF aperture-active, the production selection)
#             warm start = "-" (derived in-process from the 200-trial arm)
#   basis C : budget2-runs/strict-parity/arina-512-17p0x-production-8937
#             (the production -17.0x acquisition, all 8937 active) with the
#             recorded production-fit optimum as the refinement-only start
#   basis Cauto : same case, warm start derived in-process
#
# Every arm set runs the production SSBOptimizer.run + MetalSSBEngine.phaseVariance
# closure; only globalTrials and the TPE seed vary. One process per seed, both
# forward and reverse passes inside the harness (ABBA pairing for wall time).
set -u
BASIS=${1:?basis A or B}
shift
HARNESS_DIR=/path/to/local/perf-lab/ssb-audit/budget2/experiments/20260916-ssb-budget2
BUDGET2=/path/to/local/perf-lab/ssb-audit/budget2
RUNS=/path/to/local/perf-lab/ssb-audit/budget2-runs
PY=~/miniforge3/bin/python3.12

case "$BASIS" in
  A)
    CASE_DIR=/path/to/local/perf-lab/ssb-audit/parity-runs/strict-parity/arina-512-full-disk
    WARM=/path/to/local/perf-lab/ssb-audit/parity-runs/fit-arina-512-full-disk.json
    ;;
  B)
    CASE_DIR=$RUNS/strict-parity/arina-512-aperture-matched-8937
    WARM=-
    ;;
  C)
    CASE_DIR=$RUNS/strict-parity/arina-512-17p0x-production-8937
    WARM=/path/to/local/perf-lab/ssb-audit/budget2-runs/production-recorded-warm.json
    ;;
  Cauto)
    CASE_DIR=$RUNS/strict-parity/arina-512-17p0x-production-8937
    WARM=-
    ;;
  *) echo "unknown basis $BASIS" >&2; exit 2 ;;
esac

for SEED in "$@"; do
  OUT=$RUNS/metal-budget-$BASIS-seed$SEED
  LOG=$RUNS/metal-budget-$BASIS-seed$SEED.log
  mkdir -p "$OUT"
  echo "[budget2] basis=$BASIS seed=$SEED start=$(date -u +%H:%M:%S) load=$(uptime | sed 's/.*load averages: //')"
  GPU_RUN_LABEL=budget2-$BASIS-s$SEED /path/to/local/perf-lab/ssb-audit/gpurun \
    "$BUDGET2/build/ssb-fit-budget" "$CASE_DIR" "$OUT" "$WARM" "$SEED" 2>&1 | tee "$LOG"
  echo "[budget2] basis=$BASIS seed=$SEED done=$(date -u +%H:%M:%S) load=$(uptime | sed 's/.*load averages: //')"
done
