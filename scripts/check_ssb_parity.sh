#!/usr/bin/env bash
# GPU-first SSB parity gate.
#
#   scripts/check_ssb_parity.sh                      # native agreement-search gate
#   scripts/check_ssb_parity.sh --cpu-oracle         # expensive opt-in: MPS against the CPU oracle
#   scripts/check_ssb_parity.sh --cpu-oracle --full  # every case, the 512x512 one included
#
# The optional CPU oracle reports the MPS findings of this experiment as failures.
# The native engine's non-model reconstruction and phase-variance search, which
# this gate also measured, are retired.
#
# Every GPU command runs under the shared gpurun lock so concurrent agents do
# not corrupt each other's measurements.
#
# QUANTEM_SSB_PARITY_REPORT_PATH overrides where the report JSON is written, so
# a confirmation run cannot overwrite a recorded artifact in place.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${QUANTEM_SSB_PARITY_PYTHON:-$HOME/miniforge3/bin/python3.12}"
RUNS_ROOT="${QUANTEM_SSB_PARITY_RUNS:-$HOME/perf-lab/ssb-audit/parity-runs}"
GPURUN="${QUANTEM_SSB_PARITY_GPURUN:-$HOME/perf-lab/ssb-audit/gpurun}"
REPORT_OVERRIDE="${QUANTEM_SSB_PARITY_REPORT_PATH:-}"
FULL=0
CPU_ORACLE=0
EXTRA=()
for argument in "$@"; do
  case "$argument" in
    --full) FULL=1 ;;
    --cpu-oracle) CPU_ORACLE=1 ;;
    --export|--force-export|--all-cases) EXTRA+=("$argument") ;;
    *) echo "check_ssb_parity: unknown argument $argument" >&2; exit 2 ;;
  esac
done

if [ "$CPU_ORACLE" = 0 ]; then
  if [ "${#EXTRA[@]}" -gt 0 ] || [ "$FULL" = 1 ]; then
    echo "check_ssb_parity: oracle-specific flags require explicit --cpu-oracle; the default is the GPU agreement gate" >&2
    exit 2
  fi
  echo "GPU acceptance: native agreement search against the Python resident search; no CPU oracle."
  exec bash scripts/check_ssb_fit_trajectory.sh
fi

if [ "$FULL" = 1 ]; then
  CASE_ARGS=(--all-cases)
  REPORT="$RUNS_ROOT/gate-full.json"
else
  CASE_ARGS=(--case arina-128-full-disk --case arina-128-inner-disk \
    --case arina-128-recorded-c10)
  REPORT="$RUNS_ROOT/gate-fast.json"
fi
if [ -n "$REPORT_OVERRIDE" ]; then
  REPORT="$REPORT_OVERRIDE"
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export QUANTEM_SSB_PARITY_RUNS="$RUNS_ROOT"
export GPU_RUN_LABEL=parity
mkdir -p "$RUNS_ROOT"

if [ "${#EXTRA[@]}" -gt 0 ]; then
  GPU_RUN_LABEL=parity "$GPURUN" "$PYTHON" tests/parity/ssb_parity_gate.py \
    --cpu-oracle "${CASE_ARGS[@]}" "${EXTRA[@]}" --json "$REPORT"
else
  GPU_RUN_LABEL=parity "$GPURUN" "$PYTHON" tests/parity/ssb_parity_gate.py \
    --cpu-oracle "${CASE_ARGS[@]}" --json "$REPORT"
fi
