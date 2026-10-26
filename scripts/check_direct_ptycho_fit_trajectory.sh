#!/usr/bin/env bash
# Direct Ptycho fit-trajectory parity gate.
#
#   scripts/check_direct_ptycho_fit_trajectory.sh
#
# The agreement objective, which the default fit and tilt=False use: the native DirectPtychoAgreementSearch must
# draw the search trials of the Python resident search and reach its exact-derivative Newton's fits
# (scripts/direct_ptycho_agreement_gate.py; measured parity too when SSB_PARITY_FIXTURE and SSB_PARITY_REFERENCE are
# set). GPU commands share a per-user lock; an optional site GPU launcher can wrap the invocation.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${QUANTEM_SSB_PARITY_PYTHON:-python3}"
GPURUN="${QUANTEM_SSB_PARITY_GPURUN:-}"
for argument in "$@"; do
  echo "check_direct_ptycho_fit_trajectory: unknown argument $argument" >&2
  exit 2
done

export GPU_RUN_LABEL=parity
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" exec ${GPURUN:+"$GPURUN"} "$PYTHON" scripts/direct_ptycho_agreement_gate.py
