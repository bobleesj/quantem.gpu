# Independent trial-budget validation (cross-seed, cross-backend)

Ticket: confirm or falsify, on conditions different from the first measurement,
that at 512^2 / 8937 BF on Metal the TPE trial budget is not load-bearing
(50 ≡ 100 ≡ 200 trials, bit-identical optimum), that the extra trials buy 0.0,
that Nelder-Mead rather than the trial count is the binding constraint, and that
an NM-only arm warm-started from the best point beats the from-scratch search.

Branch `ssb-budget2`, worktree `~/perf-lab/ssb-audit/budget2`, base
`ssb-audit-parity@7372d93`. Nothing pushed; `quantem.gpu` untouched.

## The one thing that changed the test design

"8937 BF" names two different selections on the same ARINA acquisition, and
they cost 3.6x different amounts per objective evaluation:

| basis | BF selection | aperture-active terms | ms/eval | who uses it |
| --- | --- | ---: | ---: | --- |
| A: `parity-runs/strict-parity/arina-512-full-disk` | historical documented sampling (1.0909090909090908 mrad) | 2464 | ~78 | the original trial-budget finding, the strict parity gate |
| B: `budget2-runs/strict-parity/arina-512-aperture-matched-8937` | aperture-matched sampling (0.5622196476170719 mrad) | 8937 | ~283 | the production engine after calibration matching, and the recorded 62 s fit |

The original finding was measured on basis A; the deployment question ("can 200
trials be cut") is about basis B, where 200 trials costs ~57 s instead of ~16 s.
Both bases are therefore run here. Basis B reuses basis A's verified
`bf_columns.u16` by symlink and regenerates only the angular geometry with the
case module's own functions, at the sampling the native engine's
`matchApertureToBrightfieldDisk()` produces (asserted verbatim in
`tests/metal/ssb_workflow_check.swift`). Construction checks: 8937 selected,
8937 aperture-active, all aperture weights > 0, counts max/sum and the pixel set
identical to the owner case.

## Protocol

`tests/metal/ssb_fit_budget.swift` runs the production `SSBOptimizer.run` +
`MetalSSBEngine.phaseVariance` closure over the real acquisition. Only
`globalTrials` and the TPE seed vary; the objective, evaluator, summation order,
start point, search ranges and refinement are untouched. Two harness-only
changes, both disclosed:

1. the seed is now `argv[4]` (default 42, so seed 42 is unchanged);
2. `-` as the warm-start argument derives the refinement-only arm from the
   forward pass's own 200-trial best point. Basis B has no recorded production
   optimum, so this is the only self-consistent warm start there; basis A keeps
   the recorded production optimum file.

Controls: seeds {42, 7, 123} on basis A; the seed-42 run is compared row by row
against the original recorded `parity-runs/budget-arina-512-full-disk/fit-budget.json`
and must be bit-identical (it is). Every arm runs forward and reverse inside one
process (ABBA pairing for wall time, plus an order-independence control).

## Artifacts

| item | sha256 |
| --- | --- |
| `build/ssb-fit-budget` as used for every run here | `81e019e4cd6904c28507e7ff6a3d667b3b168e6c2a41f53d7ebfb5bb840dfaa9` |
| `build/ssb-fit-budget` rebuilt from the committed source | `4d441b6d6f13f9889f00c67f18daabd4e834013764828619564604e11a211578` |
| `parity/build/ssb-fit-budget` (pristine, original finding) | `5e0575a067888bbad8a81d348c71a5419b5db61dba67972bd3848f325f6c114c` |
| basis A `case.json` | `f4a2a7a0ac208f0f0c391282db6716319f97fb38726438cdf51e08ff6f71af9b` |
| basis B `case.json` | `69e2ca09e17bca434bfb4333922842e72fcc91cafac6a40a636275a919444f4b` |
| basis A warm start (`parity-runs/fit-arina-512-full-disk.json`) | `703f56ed21d35923bbf20990b73607f7636c1ba5e22a26650b7ee1d002da8798` |
| shared BF payload `bf_columns.u16` | `6b8b6f2df8c65bbde8feb666289a22374dbb364e48f2fa9b7be011a067d1f3d5` |

### Re-derivation checks run against this record

The tables below were regenerated from the raw JSON (`budget_report.py`,
`mps_report.py`) rather than copied from the run logs, and four checks were run
against the artifacts themselves:

1. The basis-A seed-42 replay was compared row by row with the original recorded
   run: all 10 arms, all 1548 evaluations (points, losses, stages, indices) are
   bit-identical, so the harness reproduces the original finding exactly.
2. Both basis-A artifacts (the original recorded run and the replay) report the
   same `activeBrightfieldCount = 2464` of `logicalBrightfieldCount = 8937`, and
   the case file's sampling is `1.0909090909090908 mrad`, confirming the original
   finding was made on a 2464-active loss.
3. The MPS blocker was reproduced in this session on the clean pristine tree
   (`qgpu` @597b566) with a 2-trial fit: identical `ValueError` and identical
   `engine.py:4973 -> engine.py:199` frames.
4. The MPS `tpe200` arms for both seeds were each recorded twice; both repeats
   reproduced the loss bit-for-bit.
5. The committed harness source was rebuilt with the repository's own recipe
   (`swiftc -O -I "$(swift build -c release --show-bin-path)/Modules" ...`,
   `scripts/check_ssb_parity.sh`), which produces a byte-different binary
   (code layout only). That rebuilt binary was re-run on basis A seed 42 and
   reproduced the original recorded run and the earlier replay bit-for-bit
   across all 1548 evaluations, so the committed source is the source of the
   measurements.
6. The blocker below was reproduced a second time in this session on the clean
   pristine tree with the same traceback; the captured output with its
   provenance header is at `budget2-runs/blocker-probe-qgpu-pristine.log`.

## Results

Filled in below from `budget2-runs/`.

### Metal (production `SSBOptimizer.run` + `MetalSSBEngine.phaseVariance`)

## Per-arm results (forward pass; reverse is bit-identical)

| basis | seed | arm | TPE trials | evals | NM evals | loss | NM gain (ulp) | ms/eval | load |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C production 8937-active (-17.0x file) | 7 | tpe25 | 25 | 96 | 69 | 0.13769800961017609 | 38468 | 282.7 | 2.68->2.12 |
| C production 8937-active (-17.0x file) | 7 | tpe50 | 50 | 116 | 65 | 0.13769800961017609 | 7757 | 273.7 | 2.12->2.04 |
| C production 8937-active (-17.0x file) | 7 | tpe100 | 100 | 172 | 71 | 0.13769800961017609 | 4996 | 273.5 | 2.04->2.02 |
| C production 8937-active (-17.0x file) | 7 | tpe200 | 200 | 262 | 61 | 0.13769800961017609 | 2604 | 280.2 | 2.02->2.48 |
| C production 8937-active (-17.0x file) | 7 | nmWarm | 0 | 28 | 27 | 0.13769800961017609 |  | 288.8 | 2.48->3.13 |
| C production 8937-active (-17.0x file) | 42 | tpe25 | 25 | 82 | 56 | 0.13769800961017609 | 5095 | 270.4 | 2.57->2.06 |
| C production 8937-active (-17.0x file) | 42 | tpe50 | 50 | 79 | 28 | 0.13769800961017609 | 14 | 266.3 | 2.06->1.82 |
| C production 8937-active (-17.0x file) | 42 | tpe100 | 100 | 129 | 27 | 0.13769799470901489 | 1 | 265.5 | 1.82->1.38 |
| C production 8937-active (-17.0x file) | 42 | tpe200 | 200 | 229 | 27 | 0.13769799470901489 | 1 | 272.3 | 1.38->1.13 |
| C production 8937-active (-17.0x file) | 42 | nmWarm | 0 | 26 | 25 | 0.13769799470901489 |  | 287.1 | 1.13->1.04 |
| C production 8937-active (-17.0x file) | 123 | tpe25 | 25 | 95 | 69 | 0.13769800961017609 | 28807 | 286.4 | 1.38->1.33 |
| C production 8937-active (-17.0x file) | 123 | tpe50 | 50 | 127 | 76 | 0.13769800961017609 | 26305 | 274.7 | 1.33->1.28 |
| C production 8937-active (-17.0x file) | 123 | tpe100 | 100 | 179 | 78 | 0.13769800961017609 | 25618 | 287.3 | 1.28->0.94 |
| C production 8937-active (-17.0x file) | 123 | tpe200 | 200 | 372 | 169 | 0.13272973895072937 | 260065 | 282.2 | 0.94->1.20 |
| C production 8937-active (-17.0x file) | 123 | nmWarm | 0 | 128 | 127 | 0.13272969424724579 |  | 267.9 | 1.20->0.87 |
| B matched 8937-active (0.0x file) | 7 | tpe25 | 25 | 85 | 59 | 0.15314754843711853 | 22288 | 259.2 | 2.19->1.97 |
| B matched 8937-active (0.0x file) | 7 | tpe50 | 50 | 126 | 75 | 0.15314754843711853 | 15639 | 261.7 | 1.97->1.62 |
| B matched 8937-active (0.0x file) | 7 | tpe100 | 100 | 173 | 72 | 0.15314754843711853 | 11700 | 263.0 | 1.62->0.87 |
| B matched 8937-active (0.0x file) | 7 | tpe200 | 200 | 272 | 71 | 0.15314754843711853 | 6527 | 264.0 | 0.87->0.82 |
| B matched 8937-active (0.0x file) | 7 | nmWarm | 0 | 30 | 29 | 0.15314754843711853 |  | 264.0 | 0.82->0.91 |
| B matched 8937-active (0.0x file) | 42 | tpe25 | 25 | 85 | 59 | 0.15314704179763794 | 2310 | 286.0 | 1.29->1.51 |
| B matched 8937-active (0.0x file) | 42 | tpe50 | 50 | 85 | 34 | 0.15314710140228271 | 254 | 273.9 | 1.51->1.89 |
| B matched 8937-active (0.0x file) | 42 | tpe100 | 100 | 128 | 26 | 0.15314711630344391 | 30 | 270.2 | 1.89->1.80 |
| B matched 8937-active (0.0x file) | 42 | tpe200 | 200 | 213 | 12 | 0.15314754843711853 | 0 | 268.3 | 1.80->1.10 |
| B matched 8937-active (0.0x file) | 42 | nmWarm | 0 | 13 | 12 | 0.15314754843711853 |  | 266.7 | 1.10->1.10 |
| B matched 8937-active (0.0x file) | 123 | tpe25 | 25 | 91 | 65 | 0.15314754843711853 | 27320 | 270.3 | 1.45->2.53 |
| B matched 8937-active (0.0x file) | 123 | tpe50 | 50 | 127 | 76 | 0.15314754843711853 | 22980 | 264.4 | 2.53->1.57 |
| B matched 8937-active (0.0x file) | 123 | tpe100 | 100 | 169 | 68 | 0.15314754843711853 | 21275 | 261.9 | 1.57->1.44 |
| B matched 8937-active (0.0x file) | 123 | tpe200 | 200 | 276 | 75 | 0.15314754843711853 | 14139 | 262.1 | 1.44->0.73 |
| B matched 8937-active (0.0x file) | 123 | nmWarm | 0 | 30 | 29 | 0.15314754843711853 |  | 261.8 | 0.73->0.70 |
| A historical 2464-active | 7 | tpe25 | 25 | 93 | 63 | 0.097227625548839569 | 114746 | 84.1 | 2.94->2.87 |
| A historical 2464-active | 7 | tpe50 | 50 | 118 | 63 | 0.097227625548839569 | 61586 | 85.7 | 2.87->4.14 |
| A historical 2464-active | 7 | tpe100 | 100 | 168 | 63 | 0.097227625548839569 | 61586 | 94.1 | 4.14->3.89 |
| A historical 2464-active | 7 | tpe200 | 200 | 268 | 63 | 0.097227625548839569 | 61586 | 91.0 | 3.89->3.85 |
| A historical 2464-active | 7 | nmWarm | 0 | 145 | 140 | 0.094526410102844238 |  | 83.9 | 3.85->5.07 |
| A historical 2464-active | 42 | tpe25 | 25 | 93 | 63 | 0.097227625548839569 | 112253 | 76.2 | 6.60->6.23 |
| A historical 2464-active | 42 | tpe50 | 50 | 112 | 58 | 0.096058391034603119 | 150405 | 76.4 | 6.23->5.42 |
| A historical 2464-active | 42 | tpe100 | 100 | 162 | 58 | 0.096058391034603119 | 150405 | 78.8 | 5.42->4.52 |
| A historical 2464-active | 42 | tpe200 | 200 | 262 | 58 | 0.096058391034603119 | 150405 | 80.6 | 4.52->4.29 |
| A historical 2464-active | 42 | nmWarm | 0 | 145 | 140 | 0.094526410102844238 |  | 79.0 | 4.29->3.71 |
| A historical 2464-active | 123 | tpe25 | 25 | 93 | 63 | 0.097227625548839569 | 107614 | 71.2 | 2.53->2.33 |
| A historical 2464-active | 123 | tpe50 | 50 | 118 | 63 | 0.097227625548839569 | 20189 | 70.8 | 2.33->2.14 |
| A historical 2464-active | 123 | tpe100 | 100 | 168 | 63 | 0.097227625548839569 | 20189 | 72.0 | 2.14->1.67 |
| A historical 2464-active | 123 | tpe200 | 200 | 268 | 63 | 0.097227625548839569 | 20189 | 74.3 | 1.67->1.35 |
| A historical 2464-active | 123 | nmWarm | 0 | 145 | 140 | 0.094526410102844238 |  | 71.3 | 1.35->1.14 |

## 50 trials vs 200 trials

| basis | seed | loss(50) | loss(200) | delta(200-50) | delta (ulp) | point identical | optima |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| C production 8937-active (-17.0x file) | 7 | 0.13769800961017609 | 0.13769800961017609 | 0 | 0.0 | no (rev no) | 50: (6.739, 0, -0.2689) / 200: (6.66, 0, -0.7236) |
| C production 8937-active (-17.0x file) | 42 | 0.13769800961017609 | 0.13769799470901489 | -1.49012e-08 | -1.0 | no (rev no) | 50: (6.485, 0, 1.066) / 200: (6.604, 0.09849, 1.134) |
| C production 8937-active (-17.0x file) | 123 | 0.13769800961017609 | 0.13272973895072937 | -0.00496827 | -333415.0 | no (rev no) | 50: (6.722, 0, -0.06099) / 200: (49.08, 49.09, 0.00762) |
| B matched 8937-active (0.0x file) | 7 | 0.15314754843711853 | 0.15314754843711853 | 0 | 0.0 | no (rev no) | 50: (5.895, 0, 0.2131) / 200: (5.896, 0, -0.3831) |
| B matched 8937-active (0.0x file) | 42 | 0.15314710140228271 | 0.15314754843711853 | 4.47035e-07 | 30.0 | no (rev no) | 50: (5.856, 0.4918, 1.3) / 200: (5.761, 0, 1.231) |
| B matched 8937-active (0.0x file) | 123 | 0.15314754843711853 | 0.15314754843711853 | 0 | 0.0 | no (rev no) | 50: (5.928, 0, -0.2196) / 200: (5.887, 0, -0.5986) |
| A historical 2464-active | 7 | 0.097227625548839569 | 0.097227625548839569 | 0 | 0.0 | yes (rev yes) | 50: (0, 51.22, -5.96e-10) / 200: (0, 51.22, -5.96e-10) |
| A historical 2464-active | 42 | 0.096058391034603119 | 0.096058391034603119 | 0 | 0.0 | yes (rev yes) | 50: (52.87, 52.88, 0.002157) / 200: (52.87, 52.88, 0.002157) |
| A historical 2464-active | 123 | 0.097227625548839569 | 0.097227625548839569 | 0 | 0.0 | yes (rev yes) | 50: (0, 51.22, -5.96e-10) / 200: (0, 51.22, -5.96e-10) |

## Verdict per basis/seed

- basis C production 8937-active (-17.0x file) seed 7: loss identical (0.0 ulp), point differs, 50 trials 31.8s vs 200 trials 73.4s
- basis C production 8937-active (-17.0x file) seed 42: loss differs (-1.0 ulp), point differs, 50 trials 21.0s vs 200 trials 62.3s
- basis C production 8937-active (-17.0x file) seed 123: loss differs (-333415.0 ulp), point differs, 50 trials 34.9s vs 200 trials 105.0s
- basis B matched 8937-active (0.0x file) seed 7: loss identical (0.0 ulp), point differs, 50 trials 33.0s vs 200 trials 71.8s
- basis B matched 8937-active (0.0x file) seed 42: loss differs (30.0 ulp), point differs, 50 trials 23.3s vs 200 trials 57.1s
- basis B matched 8937-active (0.0x file) seed 123: loss identical (0.0 ulp), point differs, 50 trials 33.6s vs 200 trials 72.3s
- basis A historical 2464-active seed 7: loss identical (0.0 ulp), point identical, 50 trials 10.1s vs 200 trials 24.4s
- basis A historical 2464-active seed 42: loss identical (0.0 ulp), point identical, 50 trials 8.6s vs 200 trials 21.1s
- basis A historical 2464-active seed 123: loss identical (0.0 ulp), point identical, 50 trials 8.4s vs 200 trials 19.9s


Every arm's forward and reverse pass is bit-identical (all 6 runs), so the
wall-time pairing hides no state leak. The seed-42 basis-A run is bit-identical
to the original recorded run, row by row.

### MPS/MLX (`optimizer.optimize`, 8937 aperture-active BF)

Runs on `ssb-audit-mps@f7a2bb6` because `origin/main` cannot run this path at
all (see the blocker below). `objective evals` counts the optimizer's objective
calls; `NM evals` is `refine_nfev`.

| arm | TPE trials | objective evals | NM evals | loss | NM gain (ulp) | p50 ms | wall s | peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
<!-- determinism control: tpe50 seed 7 recorded 2x, distinct losses [0.137724593282] -->
| seed 7 tpe50 | 50 | 68 | 41 | 0.13772459328174591 | 150 | 444.4 | 37.83 | 14.78 |
<!-- determinism control: tpe200 seed 7 recorded 2x, distinct losses [0.137698337436] -->
| seed 7 tpe200 | 200 | 107 | 5 | 0.13769833743572235 | 0 | 649.4 | 68.63 | 14.78 |
| seed 7 nmWarm | 0 | 7 | 5 | 0.13769833743572235 |  | 354.7 | 2.80 | 14.78 |
<!-- determinism control: tpe50 seed 123 recorded 2x, distinct losses [0.137697562575] -->
| seed 123 tpe50 | 50 | 80 | 53 | 0.13769756257534027 | 9432 | 372.0 | 37.10 | 14.78 |
<!-- determinism control: tpe200 seed 123 recorded 2x, distinct losses [0.137697443366] -->
| seed 123 tpe200 | 200 | 116 | 14 | 0.13769744336605072 | 38 | 665.3 | 78.20 | 14.78 |
| seed 123 nmWarm | 0 | 18 | 16 | 0.13769744336605072 |  | 376.1 | 7.21 | 14.78 |

50 vs 200 trials (MPS):

- seed 7: loss(50)=0.13772459328174591 loss(200)=0.13769833743572235 delta=-2.62558e-05 (-1762 ulp, relative -0.000191); wall 31.8s -> 67.7s
- seed 123: loss(50)=0.13769756257534027 loss(200)=0.13769744336605072 delta=-1.19209e-07 (-8 ulp, relative -8.66e-07); wall 39.2s -> 83.6s


Both seeds carry a determinism control: the `tpe200` and `tpe50` arms were each
recorded twice in separate processes, and the second process reproduced the
first process's loss bit-for-bit in all four cases. The `NM gain` column now
comes from a trace on both seeds (`TRACE_ONLY=1 run_mps_budget.sh <seed>`), so
the near-inert refinement at 200 trials is measured, not inferred: 0 ulp
(seed 7) and 38 ulp (seed 123), against 150 and 9432 ulp at 50 trials.

MPS p50 is the median wall time of one *batched objective call* (the harness's
`objective_calls`), not of one single-candidate loss: each recorded point costs
about 10.3 such calls (`evals_per_objective`), which is why the p50 column
(370-670 ms) is larger than `wall / objective evals` would suggest.

## What the numbers say

1. **On the historical 2464-active selection (the original finding's basis) the
   claim reproduces exactly and generalises across seeds.** 50 ≡ 100 ≡ 200
   trials give bit-identical losses *and* bit-identical optima for seeds 42, 7
   and 123; forward and reverse agree; the seed-42 run is bit-identical to the
   recorded original. Nothing about that measurement is in question.

2. **It does not generalise to the production 8937-active selection.** There the
   returned optimum *point* differs between 50 and 200 trials in **every** seed
   on **both** backends, and the loss difference is seed-dependent:

   | basis | seed | delta loss (200 - 50) | in ulp | what it is |
   | --- | ---: | ---: | ---: | --- |
   | C production | 42 | -1.49e-08 | -1 | 200 trials hits the recorded production optimum exactly; 50 trials is 1 ulp above it |
   | C production | 7 | 0 | 0 | both sit 1 ulp above the recorded optimum, different points |
   | C production | 123 | **-4.97e-03** | -333415 | 200 trials reaches a *different basin* (C10 = C12 = 49 nm family); 50/100 trials never enter it |
   | B matched (0.0x) | 7 / 123 | 0 | 0 | identical float32 loss, different points |
   | B matched (0.0x) | 42 | +4.47e-07 | +30 | 25 trials is the best arm; the loss *rises* with the budget |
   | A historical | 7 / 42 / 123 | 0 | 0 | identical loss *and* point |

3. **"Nelder-Mead is the binding constraint" is basis-dependent.** On the
   historical selection NM carries the search (gains of 19,000-150,000 ulp,
   140-144 evals in the refinement-only arm). On the production selection it is
   frequently inert: 0 ulp gain from the 200-trial optimum in Metal seeds 42/7
   and both MPS seeds (5-16 NM evals, immediate convergence). But when NM is
   inert the *loss* is at the plateau, and when TPE lands somewhere productive
   NM is again the thing that converts it: on basis C seed 123 the 200-trial
   TPE best is 0.13660500943660736 (trial 194), and NM descends it to
   0.13272973895072937 - 2.9% of relative loss, the entire meaningful gain of
   that arm.

4. **The reason "50 ≡ 200" fails is basin selection inside the TPE phase.** On
   basis C seed 123, TPE's last useful sample arrives at trial 194; the arms
   that stopped at 25/50/100 trials all end on the 0.1376980 plateau, and only
   the 200-trial arm pays for the transit into the (49, 49) basin. Trial-budget
   cuts remove exactly the part of the search that discriminates basins, so the
   cost of cutting is not visible as a small loss delta - it is visible as a
   3.6% loss difference when it happens.

5. **Plateau resolution.** Where arms disagree by ~0 ulp they are
   indistinguishable at the objective's own float32 resolution
   (`spacing(0.1531475) = 1.49e-08`, 9.7e-08 relative): on basis B seeds 7/123
   *every* arm returns the identical float32 loss `0.15314754843711853` while
   the points span C10 5.68-5.93 nm, C12 0-0.56 nm and phi12 -0.60 to +0.21 rad.
   The 3-parameter fit is underdetermined by the objective on the plateau; the
   point is not the measurement, the loss is.

## Blocking defect found on the way (report, not fixed here)

`origin/main` (597b566) cannot fit a 512^2 acquisition on the MPS backend:

```
ValueError: Exact 512 pair pack of 512 BF planes exceeds the 320-plane storage class.
  engine.py:4973 in _reconstruct_prepared_batch_exact_loss
  engine.py:199  in _exact_pair_row_storage_bf_512
```

Reproduced on the pristine tree (`qgpu` worktree @597b566, clean, HEAD
`597b5664f22347771f6c385390602f9750fd9d75`) with a 2-trial fit, and in this
worktree. The failure is geometric rather than timing-dependent: the packer
accumulates a whole sparse boundary without splitting it, and on an all-active
8937 selection a 512-wide logical boundary cannot compact below 512, which
exceeds the (288, 320) storage class this machine's policy selects (base M5,
24 GB, `_exact_pair_row_policy_512() -> (300, (288, 320))`). On the 2464-active
selection the same boundary compacts into the class, which is why the path
works there and fails here. `ssb-audit-mps` carries the fix
(`32ba29b`, "allocate wide exact pair packs at their own row width"); the whole
MPS-branch diff against `origin/main` for `src/` is that one file (+52/-3),
including the `tiled_input`/`tiled_output` plumbing. Any MPS 512^2 fit number
depends on that unmerged commit; the MPS rows above are on that tree.

## Cost basis (why the ticket's "8937 BF" needed two answers)

The original finding was measured on the historical selection, where only 2464
of the 8937 selected BF pixels have a non-zero aperture weight, so the loss
compacts to 2464 terms and an evaluation costs ~78 ms. The production engine
matches the aperture to the disk (detector sampling 0.5622196476170719 mrad),
which makes all 8937 terms active and an evaluation cost ~270 ms. A 200-trial
arm therefore costs ~16 s on the first basis and ~57-100 s on the second. The
budget question had to be re-asked on the second basis; that is where the claim
breaks.

## Limitations

* One machine, one GPU, one acquisition family (ARINA). Seeds tested: 42, 7, 123.
* Basis B and basis C are the same acquisition family at different tilts, so the
  basin structure may be acquisition-specific; the mechanism (TPE basin transit
  between trials ~164 and ~194) is measured on one trace.
* MPS rows depend on the unmerged `32ba29b` storage-class fix and run on the
  MPS branch tree, not on `origin/main`.
* No `.qem`, histogram, or UI work: this ticket is the trial-budget claim only.
* MPS `NM evals` is `refine_nfev`; both seeds' `NM gain` values are trace-backed
  (the seed-7 arms from the first pass, the seed-123 arms from a rerun).
* This machine is a base Apple M5 with 24 GB, so the MPS storage-class policy is
  the 288/320-plane fallback, not the M5 Max profiles the engine also carries.
  The blocker above is therefore this hardware class's failure; the fix commit
  is what makes the 8937-active fit runnable here at all.
