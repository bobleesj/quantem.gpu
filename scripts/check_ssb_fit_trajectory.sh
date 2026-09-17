#!/usr/bin/env bash
# SSB fit-trajectory parity gate.
#
#   scripts/check_ssb_fit_trajectory.sh              # fast: 128x128 real crop
#   scripts/check_ssb_fit_trajectory.sh --full       # adds the 512x512 acquisition
#   scripts/check_ssb_fit_trajectory.sh --build-metal
#
# A single objective evaluation can agree with an oracle while the *fit* still
# moves, because the search draws candidate t+1 from a history that may or may
# not already contain candidate t. This gate records the unbatched production
# trajectory (what `MetalSSBEngine.optimize` runs today), the batched pair
# trajectory, the Python/QuantEM reference trajectory on the same artifact and
# seed, a second process run for determinism, and compares the unbatched
# trajectory against its frozen pin. Every GPU command runs under the shared
# gpurun lock.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${QUANTEM_SSB_PARITY_PYTHON:-$HOME/miniforge3/bin/python3.12}"
RUNS_ROOT="${QUANTEM_SSB_PARITY_RUNS:-$HOME/perf-lab/ssb-audit/parity-runs}"
GPURUN="${QUANTEM_SSB_PARITY_GPURUN:-$HOME/perf-lab/ssb-audit/gpurun}"
BUILD_METAL=0
FULL=0
for argument in "$@"; do
  case "$argument" in
    --build-metal) BUILD_METAL=1 ;;
    --full) FULL=1 ;;
    *) echo "check_ssb_fit_trajectory: unknown argument $argument" >&2; exit 2 ;;
  esac
done

# The native batch objective is newer than the harness: compile it in only when
# the tree actually exposes it, so this gate runs on both trees.
# Keep this array nonempty: macOS Bash 3 treats an empty array as unset under -u.
SWIFT_FLAGS=(-O)
BATCH_MODE=closure
if grep -q "func phaseVarianceBatch" \
  src/quantem/gpu/swift/Sources/MetalSSBKernels/MetalSSBKernels.swift 2>/dev/null; then
  SWIFT_FLAGS+=(-D SSB_HAS_BATCH_OBJECTIVE)
  BATCH_MODE=both
fi

if [ "$BUILD_METAL" = 1 ]; then
  swift build -c release --disable-sandbox --product metal-ssb-benchmark
  build_dir=$(swift build -c release --show-bin-path)
  mkdir -p build
  swiftc "${SWIFT_FLAGS[@]}" -I "$build_dir/Modules" tests/metal/ssb_fit_trajectory.swift \
    "$build_dir"/MetalSSBKernels.build/*.o -o build/ssb-fit-trajectory -parse-as-library
fi
if [ ! -x build/ssb-fit-trajectory ]; then
  echo "check_ssb_fit_trajectory: build/ssb-fit-trajectory is missing; pass --build-metal" >&2
  exit 2
fi

CASE_DIRS=("$RUNS_ROOT/strict-parity/arina-128-full-disk")
CASES=(arina-128-full-disk)
if [ "$FULL" = 1 ]; then
  CASE_DIRS+=("$RUNS_ROOT/strict-parity/arina-512-full-disk")
  CASES+=(arina-512-full-disk)
fi
for directory in "${CASE_DIRS[@]}"; do
  if [ ! -f "$directory/case.json" ]; then
    echo "check_ssb_fit_trajectory: $directory/case.json is missing; run scripts/check_ssb_parity.sh --export" >&2
    exit 2
  fi
done

mkdir -p "$RUNS_ROOT"
export GPU_RUN_LABEL=parity

commands=""
for index in "${!CASES[@]}"; do
  case_name="${CASES[$index]}"
  commands+="build/ssb-fit-trajectory '${CASE_DIRS[$index]}' '$RUNS_ROOT/fit-$case_name' $BATCH_MODE && "
  commands+="cp '$RUNS_ROOT/fit-$case_name/fit-trajectory.json' '$RUNS_ROOT/fit-$case_name.json' && "
done
# A pin is only meaningful if the same command twice gives the same trajectory.
commands+="build/ssb-fit-trajectory '${CASE_DIRS[0]}' '$RUNS_ROOT/fit-${CASES[0]}-repeat' $BATCH_MODE && "
commands+="cp '$RUNS_ROOT/fit-${CASES[0]}-repeat/fit-trajectory.json' '$RUNS_ROOT/fit-${CASES[0]}-repeat.json' && "
commands+="PYTHONPATH=src $PYTHON scripts/ssb_fit_reference_trajectory.py "
commands+="--case arina-128-full-disk --json '$RUNS_ROOT/fit-reference-128.json'"

"$GPURUN" bash -c "set -euo pipefail; cd '$PWD'; $commands"

PIN="tests/parity/fixtures/ssb_fit_trajectory_128.json"
status=0
PYTHONPATH=src "$PYTHON" scripts/ssb_fit_trajectory_report.py \
  --native "$RUNS_ROOT/fit-arina-128-full-disk.json" \
  --repeat "$RUNS_ROOT/fit-arina-128-full-disk-repeat.json" \
  --reference "$RUNS_ROOT/fit-reference-128.json" \
  --pin "$PIN" || status=1
if [ "$FULL" = 1 ]; then
  PYTHONPATH=src "$PYTHON" scripts/ssb_fit_trajectory_report.py \
    --native "$RUNS_ROOT/fit-arina-512-full-disk.json" || status=1
fi
exit "$status"
