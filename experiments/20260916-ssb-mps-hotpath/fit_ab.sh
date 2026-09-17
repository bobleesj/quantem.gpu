#!/bin/bash
# Paired fit A/B at 8937 logical BF under one GPU lock hold.
#
# Alternates the frozen 32ba29b engine and HEAD inside a single lock hold so
# host-load drift is shared rather than attributed to the change.  The fit is
# deterministic (seed 42), so the A/B is a timing comparison at a fixed answer:
# loss, phase sha256 and object sha256 must match exactly between engines.
#
# REPS   how many A/B pairs (default 2)
# TRIALS optuna trials per fit (default 200; smaller values are the cheap
#        small-scale variant because every call still runs the full 8937-BF
#        pack structure, so per-call times stay comparable)
# PROBE  1 to also re-run the intermediate redundancy probe (default 0)
# PREFIX label prefix (default fit-ab)
# REVERSE 1 runs head before the frozen engine each pair (ABBA control: the
#        first fit of a pair settles after the previous process released its
#        ~12 GB, so order has to be balanced rather than fixed)
set -u
REPS=${REPS:-2}
TRIALS=${TRIALS:-200}
PROBE=${PROBE:-0}
PREFIX=${PREFIX:-fit-ab}
REVERSE=${REVERSE:-0}
MPS=/path/to/local/perf-lab/ssb-audit/mps
RUNS=/path/to/local/perf-lab/ssb-audit/mps-runs/20260916-ssb-mps-hotpath
REF=/path/to/local/perf-lab/ssb-audit/mps-runs/ref-src-32ba29b/src
HARNESS=$MPS/experiments/20260916-ssb-mps-hotpath
PY=~/miniforge3/bin/python3.12
OUT=$RUNS/fit_ab.jsonl
PROBE_OUT=$RUNS/intermediate_probe.jsonl

set -e
echo "[fit_ab] pairs=$REPS trials=$TRIALS prefix=$PREFIX"
if [ "$PROBE" = 1 ]; then
  for engine in ref head; do
    if [ "$engine" = ref ]; then SRC=$REF; else SRC=$MPS/src; fi
    echo "[fit_ab] intermediate probe engine=$engine"
    SSB_SRC=$SRC PYTHONPATH=$SRC $PY $HARNESS/intermediate_probe.py \
      --terms 1 --json-out "$PROBE_OUT" --label "$PREFIX-probe-$engine" >/dev/null
  done
fi

for rep in $(seq 1 "$REPS"); do
  if [ "$REVERSE" = 1 ]; then engines="head ref"; else engines="ref head"; fi
  for engine in $engines; do
    if [ "$engine" = ref ]; then SRC=$REF; else SRC=$MPS/src; fi
    echo "[fit_ab] rep=$rep engine=$engine label=$PREFIX-$engine-$rep"
    SSB_SRC=$SRC PYTHONPATH=$SRC $PY $HARNESS/profile_mps.py \
      --stage fit --trials "$TRIALS" --seed 42 \
      --json-out "$OUT" --label "$PREFIX-$engine-$rep" >/dev/null
  done
done
echo "[fit_ab] done"
