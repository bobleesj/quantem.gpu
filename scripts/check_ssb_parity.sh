#!/usr/bin/env bash
# GPU-first SSB parity gate.
#
#   scripts/check_ssb_parity.sh                 # frozen GPU fit and repeat
#   scripts/check_ssb_parity.sh --full          # adds the 512x512 GPU fit
#   scripts/check_ssb_parity.sh --build-metal   # build the native harness first
#   scripts/check_ssb_parity.sh --cpu-oracle --metal-only  # expensive opt-in
#
# The optional CPU oracle reports the MPS findings of this experiment as failures,
# so its exit status is not the native Metal acceptance signal. Use
# `--metal-only` when the change under test is in the Metal path: every bound is
# unchanged, only the MPS measurement is skipped.
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
BUILD_METAL=0
FULL=0
CPU_ORACLE=0
EXTRA=()
for argument in "$@"; do
  case "$argument" in
    --build-metal) BUILD_METAL=1 ;;
    --full) FULL=1 ;;
    --cpu-oracle) CPU_ORACLE=1 ;;
    --export|--force-export|--no-metal|--metal-only|--all-cases) EXTRA+=("$argument") ;;
    *) echo "check_ssb_parity: unknown argument $argument" >&2; exit 2 ;;
  esac
done

if [ "$CPU_ORACLE" = 0 ]; then
  if [ "${#EXTRA[@]}" -gt 0 ]; then
    echo "check_ssb_parity: oracle-specific flags require explicit --cpu-oracle; the default is frozen GPU parity" >&2
    exit 2
  fi
  GPU_ARGS=(scripts/check_ssb_fit_trajectory.sh)
  if [ "$BUILD_METAL" = 1 ]; then GPU_ARGS+=(--build-metal); fi
  if [ "$FULL" = 1 ]; then GPU_ARGS+=(--full); fi
  echo "GPU acceptance: frozen fit trajectory, repeat determinism, and MPS GPU comparison; no CPU oracle."
  exec bash "${GPU_ARGS[@]}"
fi

# The native harness is a standalone binary: `swift test` needs Xcode, which
# this host does not have.
if [ "$BUILD_METAL" = 1 ]; then
  swift build -c release --disable-sandbox --product metal-ssb-benchmark
  build_dir=$(swift build -c release --show-bin-path)
  mkdir -p build
  swiftc -O -I "$build_dir/Modules" tests/metal/ssb_parity_check.swift \
    "$build_dir"/MetalSSBKernels.build/*.o -o build/ssb-parity-check -parse-as-library
  printf '{"binary": "%s"}\n' "$PWD/build/ssb-parity-check" > build/ssb-parity-check.json
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
