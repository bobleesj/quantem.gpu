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
# trajectory against its frozen pin. GPU commands share a per-user lock; an
# optional site GPU launcher can wrap the whole invocation.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${QUANTEM_SSB_PARITY_PYTHON:-python3}"
RUNS_ROOT="${QUANTEM_SSB_PARITY_RUNS:-$HOME/perf-lab/ssb-audit/parity-runs}"
GPURUN="${QUANTEM_SSB_PARITY_GPURUN:-}"
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
  native/swift/Sources/MetalSSBKernels/MetalSSBKernels.swift 2>/dev/null; then
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
if [ "$FULL" = 1 ]; then
  CASE_DIRS+=("$RUNS_ROOT/strict-parity/arina-512-full-disk")
fi
for directory in "${CASE_DIRS[@]}"; do
  if [ ! -f "$directory/case.json" ]; then
    echo "check_ssb_fit_trajectory: $directory/case.json is missing; restore the frozen fixture and set QUANTEM_SSB_PARITY_RUNS. Do not recapture its pin." >&2
    exit 2
  fi
done

ARGS=(scripts/ssb_fit_gate.py --runs "$RUNS_ROOT" --batch-mode "$BATCH_MODE")
if [ "$FULL" = 1 ]; then
  ARGS+=(--full)
fi
export GPU_RUN_LABEL=parity
if [ -n "$GPURUN" ]; then
  exec "$GPURUN" "$PYTHON" "${ARGS[@]}"
fi
exec "$PYTHON" "${ARGS[@]}"
