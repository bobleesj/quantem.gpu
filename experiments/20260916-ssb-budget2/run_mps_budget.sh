#!/bin/bash
# Trial-budget cross-check on the MPS/MLX path (8937 aperture-active BF).
#
#   run_mps_budget.sh <seed>
#
# Same protocol as the Metal arms:
#   tpe200  : 200 TPE trials + Nelder-Mead refinement
#   nmWarm  : refinement only, started at the 200-trial arm's own optimum
#   tpe50   : 50 TPE trials + Nelder-Mead refinement
#
# Each arm is its own process (one prepare per arm, as the MPS fit evidence
# always was) and each holds the shared GPU lock. The NM-only arm consumes the
# optimum the 200-trial arm just wrote, so it is the refinement-only arm of the
# same protocol rather than a different search.
set -u
SEED=${1:?seed}
PY=~/miniforge3/bin/python3.12
HARNESS=/path/to/local/perf-lab/ssb-audit/budget2/experiments/20260916-ssb-budget2
BUDGET2=/path/to/local/perf-lab/ssb-audit/budget2
# origin/main cannot run this path at all (see README: the 512-wide exact pair
# pack raises against the 320-plane storage class), so the MPS arms run on the
# only tree where the 512^2 fit executes; SSB_SRC records which one.
SSB_SRC=${SSB_SRC:-/path/to/local/perf-lab/ssb-audit/mps/src}
echo "[budget2-mps] SSB_SRC=$SSB_SRC git=$(git -C "$(dirname "$SSB_SRC")" rev-parse --short HEAD 2>/dev/null)"
RUNS=/path/to/local/perf-lab/ssb-audit/budget2-runs
OUT=$RUNS/mps-fit-arm.jsonl
LOGDIR=$RUNS/mps-budget-seed$SEED
mkdir -p "$LOGDIR"

run_arm () {
  local label=$1 trials=$2 warmjson=${3:-}
  local extra=""
  if [ -n "$warmjson" ]; then extra="--warm-json $warmjson"; fi
  echo "[budget2-mps] seed=$SEED arm=$label trials=$trials start=$(date -u +%H:%M:%S) load=$(uptime | sed 's/.*load averages: //')"
  GPU_RUN_LABEL=budget2-mps-s$SEED-$label /path/to/local/perf-lab/ssb-audit/gpurun \
    env SSB_SRC=$SSB_SRC PYTHONPATH=$SSB_SRC $PY $HARNESS/mps_fit_arm.py \
      --stage fit --trials "$trials" --seed "$SEED" $extra \
      --json-out "$OUT" --label "seed$SEED-$label" > "$LOGDIR/$label.log" 2>&1
  echo "[budget2-mps] seed=$SEED arm=$label exit=$? done=$(date -u +%H:%M:%S) load=$(uptime | sed 's/.*load averages: //')"
}

if [ "${TRACE_ONLY:-0}" = "1" ]; then
  # trace-capture pass: rerun the two trial-count arms (deterministic, so the
  # losses must match the first pass) with the full TPE/refine trace recorded.
  run_arm tpe200 200
  run_arm tpe50 50
  exit 0
fi
run_arm tpe200 200
# The NM-only arm starts where the 200-trial arm finished: extract its optimum.
WARM=$LOGDIR/warm.json
$PY - "$OUT" "seed$SEED-tpe200" "$WARM" <<'PY'
import json, sys
path, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
best = None
for line in open(path, encoding="utf-8"):
    record = json.loads(line)
    if record.get("label") == label:
        best = record["stages"]["fit"]["aberrations"]
if best is None:
    raise SystemExit(f"no record labelled {label} in {path}")
json.dump({"aberrations": {k: float(v) for k, v in best.items()}}, open(out, "w"), indent=2)
print(f"warm start for {label}: {best}")
PY
run_arm nmWarm 0 "$WARM"
run_arm tpe50 50
